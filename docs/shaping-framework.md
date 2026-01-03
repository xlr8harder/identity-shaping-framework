# Shaping Framework Design

A design for consolidating the Aria codebase into a reusable identity-shaping toolkit.

Related issue: `existence-claude-e6c` (Design generalized identity-shaping framework)

---

## Goals

1. **Consolidate scattered code** into a proper Python package
2. **Enable testing** - catch subtle bugs like API changes
3. **Make reusable** - separate generic toolkit from Aria-specific content
4. **Adopt dispatcher** for complex multi-step pipelines

---

## Package Layout

```
shaping/                        # Generic identity-shaping toolkit
├── __init__.py
├── data/                       # Training data utilities
│   ├── think_tags.py          # Validation, stripping, extraction
│   └── format.py              # train.jsonl format handling
├── eval/                       # Evaluation infrastructure
│   ├── parsers.py             # XML parsing for structured responses
│   ├── rubrics.py             # Rubric definitions (generic base)
│   └── judge.py               # Assessment/judging logic
├── modeling/                   # Model backends and renderers
│   ├── clients.py             # LLMClient for mq-registered models
│   ├── backends.py            # BackendManager implementations
│   ├── model_formats.py       # Model format configuration
│   └── tinker/                # Tinker-specific wrappers
│       ├── client.py          # TinkerClient for trained models
│       └── renderers.py       # Renderer wrappers with HF compatibility
└── pipeline/                   # Dispatcher-based pipeline infrastructure
    ├── tasks.py               # GeneratorTask base classes
    └── runner.py              # CLI/runner for local execution

tests/
├── test_think_tags.py
├── test_parsers.py
├── test_inference.py          # With mocks
└── conftest.py                # Fixtures

# Aria-specific content stays in place:
design/                         # Aria's identity docs, narratives
artifacts/                      # Aria's philosophical framework
augmentation/                   # Aria's data pipelines (gradually adopt shaping.pipeline)
training/                       # Training configs and experiments
```

---

## Migration Strategy

### Phase 1: Foundation (current)
- [x] Create `shaping/` package structure
- [x] Extract `shaping.data.think_tags` from eval_lib
- [x] Extract `shaping.eval.parsers` from eval_lib
- [ ] Add tests for extracted modules
- [ ] Update eval_lib to import from shaping

### Phase 2: Modeling Backends
- [x] Implement `shaping.modeling.backends.LLMClientBackend`
- [x] Implement `shaping.modeling.backends.TinkerBackend`
- [x] Implement `shaping.modeling.tinker.renderers` with HF compatibility
- [ ] Test with dispatcher's local file mode

### Phase 3: Pipeline Adoption
- [ ] Convert one simple pipeline to dispatcher (e.g., identity)
- [ ] Validate approach works
- [ ] Gradually convert complex pipelines (narrative, misconception)

### Phase 4: Full Migration
- [ ] Move remaining eval_lib code to shaping.eval
- [ ] Update all imports
- [ ] Deprecate training/code/eval_lib.py

---

## Dispatcher Integration

### Why dispatcher?

Current shell-based pipelines are awkward for:
- Multi-step conditional flows (generate → judge → maybe retry)
- Parallel requests within a row
- Structured error handling
- Complex state management

Dispatcher's generator pattern is cleaner:

```python
class ReviseAndJudgeTask(GeneratorTask):
    def task_generator(self):
        # Step 1: Generate initial response
        resp = yield Request({"messages": self.data["messages"], "temperature": 0.7})

        # Step 2: Judge it
        judge_resp = yield Request({"messages": build_judge_prompt(resp.get_text())})
        score = parse_score(judge_resp.get_text())

        # Step 3: Revise if needed
        if score < 3:
            resp = yield Request({"messages": build_revision_prompt(resp.get_text())})

        return {"response": resp.get_text(), "score": score}
```

### Architecture

```
FileTaskSource                  LLMClientBackend / TinkerBackend
(reads JSONL)                   (calls mq/llm_client or tinker)
      │                                │
      ▼                                ▼
┌─────────────────────────────────────────┐
│            TaskManager                   │
│  - ThreadPoolExecutor(num_workers)       │
│  - Schedules requests from active tasks  │
│  - Handles completion, saves results     │
└─────────────────────────────────────────┘
                     │
                     ▼
           GeneratorTask subclass
   - yield Request → receive Response
   - yield [Req, Req] → receive [Resp, Resp]
   - return result_dict
```

### LLMClientBackend Implementation

```python
from dispatcher.taskmanager.backend.base import BackendManager
from dispatcher.taskmanager.backend.request import Request, Response
from llm_client import get_provider
from llm_client.retry import retry_request
from mq import store as mq_store

class LLMClientBackend(BackendManager):
    """Backend for mq-registered models via llm_client."""

    def __init__(self, shortname: str, max_retries: int = 5):
        model_info = mq_store.get_model(shortname)
        self.provider = get_provider(model_info["provider"])
        self.model_id = model_info["model"]
        self.sysprompt = model_info.get("sysprompt")
        self.max_retries = max_retries

    def process(self, request: Request) -> Response:
        messages = list(request.content.get("messages", []))

        # Inject sysprompt if configured
        if self.sysprompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self.sysprompt}] + messages

        # Extract options (temperature, max_tokens, etc.)
        options = {k: v for k, v in request.content.items() if k != "messages"}

        # Use retry_request for rate limit handling
        result = retry_request(
            self.provider,
            messages=messages,
            model_id=self.model_id,
            max_retries=self.max_retries,
            **options
        )

        if result.success:
            # Format for Response.get_text() compatibility
            content = {"choices": [{"message": {"content": result.standardized_response["content"]}}]}
            return Response(request, content=content, model_name=self.model_id)

        return Response.from_error(request, Exception(str(result.error_info)), model_name=self.model_id)

    def is_healthy(self) -> bool:
        return True
```

### TinkerBackend Implementation

```python
class TinkerBackend(BackendManager):
    """Backend for trained models via tinker."""

    def __init__(self, spec: str):
        # Reuse model resolution from eval_lib
        self.model_name, self.renderer_name, self.model_path = resolve_model_spec(spec)

        self.service_client = tinker.ServiceClient()
        self.sampling_client = self.service_client.create_sampling_client(
            model_path=self.model_path,
            base_model=self.model_name,
        )
        self.tokenizer = get_tokenizer(self.model_name)
        self.renderer = renderers.get_renderer(self.renderer_name, self.tokenizer)

        self.sampling_params = t_types.SamplingParams(
            max_tokens=2048,
            temperature=0.7,
            stop=self.renderer.get_stop_sequences(),
        )

    def process(self, request: Request) -> Response:
        messages = request.content.get("messages", [])

        # Build prompt via renderer
        model_input = self.renderer.build_generation_prompt(messages)

        # Sample (sync version)
        resp = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=self.sampling_params,
        )

        # Parse response
        for seq in resp.sequences:
            parsed = self.renderer.parse_response(seq.tokens)
            if parsed:
                content = parsed[0]["content"]
                # Format for Response.get_text()
                return Response(
                    request,
                    content={"choices": [{"message": {"content": content}}]},
                    model_name=self.model_name
                )

        return Response.from_error(request, Exception("No response generated"))

    def is_healthy(self) -> bool:
        return True
```

### Running Pipelines

Local file mode (no server needed):

```bash
python -m shaping.pipeline.runner \
    --task augmentation.narrative.tasks.NarrativeTask \
    --input augmentation/narrative/prompts.jsonl \
    --output augmentation/narrative/stage1-output.jsonl \
    --backend llm_client \
    --model aria-v0.9-full \
    --workers 16
```

Or via Python:

```python
from shaping.pipeline.runner import run
from shaping.modeling import LLMClientBackend
from augmentation.narrative.tasks import NarrativeTask

backend = LLMClientBackend("aria-v0.9-full")
run(
    task_cls=NarrativeTask,
    input_path="prompts.jsonl",
    output_path="output.jsonl",
    backend=backend,
    workers=16,
)
```

---

## Testing Strategy

| Layer | What | How |
|-------|------|-----|
| `shaping.data` | Think tag validation, stripping | Pure unit tests |
| `shaping.eval` | XML parsing, assessment parsing | Unit tests with fixtures |
| `shaping.modeling` | Backend API calls | Mock provider, integration tests |
| `shaping.pipeline` | Task execution | Mock backend, sample data |

### Example: Think Tags Test

```python
import pytest
from shaping.data import validate_think_tags, strip_thinking

class TestValidateThinkTags:
    def test_no_tags_valid(self):
        assert validate_think_tags("Just a response") == (True, None)

    def test_matched_pair_valid(self):
        assert validate_think_tags("<think>reasoning</think>response") == (True, None)

    def test_unclosed_tag_invalid(self):
        is_valid, error = validate_think_tags("<think>stuck in loop...")
        assert not is_valid
        assert error == "unclosed_think_tag"

    def test_orphaned_close_invalid(self):
        is_valid, error = validate_think_tags("response</think>")
        assert not is_valid
        assert error == "orphaned_close_tags"

class TestStripThinking:
    def test_strips_complete_block(self):
        assert strip_thinking("<think>hidden</think>visible") == "visible"

    def test_handles_no_tags(self):
        assert strip_thinking("just text") == "just text"

    def test_strips_orphaned_close(self):
        # When <think> was in prompt
        assert strip_thinking("reasoning</think>response") == "response"
```

### Smoke Test for Tinker API

```python
def test_tinker_renderer_api():
    """Catch breaking changes in tinker_cookbook renderer API."""
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = renderers.get_renderer("qwen3", tokenizer)

    # Verify expected methods exist
    assert hasattr(renderer, "build_generation_prompt")
    assert hasattr(renderer, "parse_response")
    assert hasattr(renderer, "get_stop_sequences")

    # Verify build_generation_prompt works
    messages = [{"role": "user", "content": "Hello"}]
    result = renderer.build_generation_prompt(messages)
    assert result is not None
```

---

## Dependencies

Add to pyproject.toml:

```toml
[project]
dependencies = [
    # ... existing ...
    "dispatcher @ git+https://github.com/LumiOpen/dispatcher",
]
```

---

## Open Questions

1. **DVC integration** - Do pipelines still use DVC for dependency tracking, or does dispatcher replace that?
   - Likely: Keep DVC for pipeline-level deps (stable.sh changes), use dispatcher within pipelines

2. **Async vs sync** - Dispatcher uses sync ThreadPoolExecutor. Some of our code is async.
   - Likely fine: Dispatcher tasks shouldn't block, async calls can be wrapped

3. **Error aggregation** - How to surface errors from multi-step tasks?
   - TaskFailed provides structured errors, can aggregate in output

4. **Gradual adoption** - Can new and old pipelines coexist?
   - Yes: New pipelines use shaping.pipeline, old pipelines keep working

---

## Next Steps

1. Add tests for `shaping.data.think_tags` and `shaping.eval.parsers`
2. Update eval_lib to import from shaping (backwards compat)
3. Implement LLMClientBackend
4. Test dispatcher local mode with simple task
5. Convert one pipeline (identity?) as proof of concept
