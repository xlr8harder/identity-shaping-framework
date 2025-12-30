# Design Notes: Framework Architecture

*Open questions and architectural decisions for the identity-shaping-framework*

**Status**: Early design phase - gathering requirements and exploring options

---

## Core Architectural Questions

### 1. Project Initialization Pattern

**Question**: How do users start a new identity-shaping project?

**Options**:

#### Option A: Fork the framework repository
```
User forks: xlr8harder/identity-shaping-framework
User modifies templates in place
User's work lives in their fork
```

**Problems**:
- Couples user's identity work to framework evolution
- Merge conflicts when framework updates
- User's content mixed with framework code
- Hard to incorporate upstream improvements
- Multiple projects require multiple forks

#### Option B: Framework as installable tool + generated projects ✅ (Preferred)
```
Framework repo: Core methodology, templates, tools
User project: Generated from templates, separate repo
Tool: `isf` CLI for init/upgrade/migrate
```

**Structure**:
```
# Framework (upstream, evolves independently)
/home/user/git/identity-shaping-framework/
  ├── docs/              # Methodology, guides
  ├── templates/         # Document templates
  ├── src/              # Framework tools
  └── pyproject.toml    # Installable package

# User project (generated, separate)
/home/user/git/my-ai-identity/
  ├── identity/         # Generated from templates
  │   ├── identity-document.md
  │   ├── narrative.md
  │   └── practice-guide.md
  ├── training/         # Training data structure
  ├── .env             # User's API keys (gitignored)
  ├── .env.example     # Template for keys
  ├── framework.yaml   # Which framework version/features
  └── README.md        # Generated project docs
```

**Workflow**:
```bash
# Install framework
pip install identity-shaping-framework

# Initialize new project
isf init my-ai-identity --template=minimal
# or
isf init my-ai-identity --template=full-featured

# Work in project
cd my-ai-identity
# Agent operates within this directory structure

# Later: upgrade framework
isf upgrade --from=0.1.0 --to=0.2.0
# Migrates data formats, updates templates
```

**Benefits**:
- Clean separation: framework evolves, user projects stable
- No merge conflicts
- Multiple projects from one framework install
- Standard project structure
- Migration paths for data format changes

**Open questions**:
- Should `isf` be Python-based or language-agnostic?
- How minimal can "minimal" template be?
- What gets version-controlled in user project vs. generated?

---

### 2. Secret Management

**Current state**: Likely scattered across shell scripts or config files in Aria project

**Proposed**: Standard dotenv pattern

**Design**:

```bash
# .env.example (committed, shows structure)
# API Keys for model access
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DEEPINFRA_API_KEY=
TOGETHER_API_KEY=

# Model endpoints (optional overrides)
# OPENAI_BASE_URL=
# ANTHROPIC_BASE_URL=

# Project settings
DEFAULT_MODEL=gpt-4
CONTEXT_WINDOW=128000

# .env (gitignored, user creates locally)
OPENAI_API_KEY=sk-actual-key-here
ANTHROPIC_API_KEY=sk-ant-actual-key
DEFAULT_MODEL=claude-opus-4
```

**Implementation**:
- Python: `python-dotenv` library
- Shell scripts: `source .env` or `dotenv` tool
- Framework generates `.env.example` during `isf init`
- User copies to `.env` and fills in actual keys

**Benefits**:
- Standard practice developers expect
- Easy onboarding (copy example, fill in)
- Secrets never committed
- Clear documentation of required keys

**Open questions**:
- Should framework provide key validation (check if keys work)?
- How to handle optional vs. required keys?
- Different .env files for different environments (dev/prod)?

---

### 3. Data Format Versioning

**Why needed**:
- Templates will evolve
- New required sections may be added
- Validation criteria will change
- Migration paths needed

**Proposed structure**:

```yaml
# framework.yaml (in user project)
framework_version: "0.1.0"
data_format_version: "1"
initialized: "2025-12-23"

templates_used:
  - identity-document
  - narrative-document
  - practice-guide

features_enabled:
  - philosophical-grounding: false
  - training-pipelines: true
  - evaluation-suite: false
```

**Schema versioning**:

```python
# In framework codebase
SCHEMA_VERSIONS = {
    "1": {
        "identity_document": {
            "required_sections": [
                "who",
                "how",
                "self_knowledge"
            ],
            "optional_sections": [
                "relationship"
            ],
            "validation": "v1"
        },
        "narrative_document": {
            "formats": ["external", "self"],
            "required_sections": ["history", "current"],
        }
    },
    "2": {  # Future version
        "identity_document": {
            "required_sections": [
                "who",
                "how",
                "self_knowledge",
                "limitations"  # New required section
            ],
            "optional_sections": [
                "relationship",
                "evolution"  # New optional section
            ],
            "validation": "v2"
        },
    }
}
```

**Migration support**:

```bash
# Detect schema version mismatch
isf status
# Warning: Framework 0.2.0 expects data format 2, project uses format 1
# Run: isf migrate

# Migrate project to new schema
isf migrate --to=2
# Framework:
#   1. Backs up current documents
#   2. Analyzes existing content
#   3. Adds new required sections with TODOs
#   4. Updates framework.yaml
#   5. Reports what changed

# Dry run option
isf migrate --to=2 --dry-run
# Shows what would change without modifying files
```

**Benefits**:
- Clear versioning of expectations
- Automated migration paths
- Users can stay on old version if needed
- Framework can evolve without breaking existing projects

**Open questions**:
- How granular should versioning be (major.minor.patch)?
- Should migrations be reversible?
- How to handle custom sections users added?
- Validation: warn vs. error for schema mismatches?

---

### 4. Agent Interface Design

**Core principle**: Agent-first design

**Implications**:

#### File structure must be agent-navigable
```
# Good: Clear hierarchy, predictable locations
my-project/
├── identity/
│   ├── identity-document.md
│   └── narrative.md
├── training/
│   ├── prompts/
│   └── generated/
└── framework.yaml

# Bad: Flat structure, unclear organization
my-project/
├── doc1.md
├── doc2.md
├── stuff.jsonl
└── config
```

#### Templates must be self-documenting
Each template needs:
- Purpose statement (why this exists)
- Structure guide (what goes where)
- Validation criteria (how to know it's good)
- Failure modes (what to avoid)
- Examples (how it can look)

Agents read these to self-direct.

#### Validation must be programmatic
```bash
# Agent can run validation
isf validate identity/identity-document.md
# Returns:
# ✅ Purpose clarity
# ✅ Concreteness
# ❌ Missing section: self_knowledge
# ⚠️  Warning: Only 2 core qualities (recommend 3-7)

# Agent can self-validate before presenting to human
isf validate --all
# Checks all documents, reports status
```

#### Commands must support agent workflows
```bash
# Agent-friendly commands (structured output)
isf init project --format=json
isf validate --format=json
isf status --format=json

# Human-friendly commands (pretty output)
isf init project
isf validate
isf status
```

**Open questions**:
- Should validation be blocking or advisory?
- How much feedback should validation provide?
- Should agents be able to fix validation errors automatically?

---

### 5. Multi-Language Support

**Question**: Should framework be Python-only or support multiple languages?

**Options**:

#### Python-centric
```python
# Framework in Python
pip install identity-shaping-framework
isf init my-project

# Training data generation in Python
from isf import generate_training_data
```

**Pros**:
- Single implementation
- Rich ecosystem (dotenv, pydantic, etc.)
- Easy to maintain

**Cons**:
- Users must have Python
- May limit adoption

#### Language-agnostic core + bindings
```
# Core: Shell scripts + YAML/JSON configs
# Templates: Pure markdown
# Bindings: Python, JS, etc.

# Install via package manager
npm install -g isf
pip install isf
# Both wrap same core
```

**Pros**:
- Broader adoption
- Users choose their tools

**Cons**:
- More complex to maintain
- Feature parity across bindings

**Open questions**:
- What's the actual user base? (Researchers = Python, broader = multi-language)
- Can templates + markdown be enough for most use cases?
- Do we need programmatic API or CLI sufficient?

---

### 6. Training Data Pipeline

**Question**: How do users generate training data?

**Current Aria approach**:
- Multiple pipeline types (framework, identity, narrative, etc.)
- DVC for reproducibility
- Shell scripts for generation
- Manual quality checking

**Framework approach options**:

#### Option A: Prescriptive pipeline tool
```bash
isf generate training \
  --pipeline=framework-concepts \
  --count=100 \
  --model=gpt-4

isf generate training \
  --pipeline=identity-demos \
  --count=200 \
  --model=claude-opus
```

**Pros**: Easy to use, consistent
**Cons**: Rigid, may not fit all use cases

#### Option B: Flexible templates + guidance
```
Framework provides:
- Example pipeline scripts
- Quality validation tools
- Best practices documentation

Users implement:
- Their own generation scripts
- Using their preferred tools
- Following framework patterns
```

**Pros**: Flexible, adaptable
**Cons**: More work for users, less consistent

**Hybrid approach** ✅ (Preferred):
- Framework provides default pipeline implementations
- Users can customize or replace
- Validation tools work regardless of generation method

```bash
# Use default pipeline
isf generate training --pipeline=framework-concepts

# Use custom pipeline
cd training/
python my_custom_pipeline.py

# Validate regardless of source
isf validate training/*.jsonl
```

**Open questions**:
- How much pipeline logic belongs in framework vs. user code?
- Should framework provide model access or assume users have it?
- How to balance ease-of-use vs. flexibility?

---

## Implementation Priorities

### Phase 1: Core Infrastructure (Now)
- [ ] Design `isf` CLI structure
- [ ] Create project template structure
- [ ] Implement basic `isf init`
- [ ] dotenv pattern for secrets
- [ ] framework.yaml schema

### Phase 2: Validation & Quality (Next)
- [ ] Schema versioning system
- [ ] `isf validate` implementation
- [ ] Programmatic validation for agents
- [ ] Migration framework

### Phase 3: Training Pipelines (Later)
- [ ] Default pipeline implementations
- [ ] Quality checking tools
- [ ] DVC integration patterns
- [ ] Training data validation

### Phase 4: Advanced Features (Future)
- [ ] Multi-language bindings
- [ ] Cloud integration
- [ ] Collaboration features
- [ ] Evaluation suite

---

## Open Design Questions

### Immediate
1. **Language choice**: Python-only or multi-language from start?
2. **Minimal template**: How minimal can we make it while still useful?
3. **Validation strictness**: Block on errors or just warn?

### Near-term
4. **Migration strategy**: Auto-migrate or require manual intervention?
5. **Pipeline flexibility**: Prescriptive defaults or maximum flexibility?
6. **Schema evolution**: Who decides what's required vs. optional?

### Long-term
7. **Collaboration**: How do multiple humans/agents work on same project?
8. **Deployment**: How to move from development to production training?
9. **Evaluation**: Framework-provided evaluation suite or user-implemented?

---

## Decision Log

**2025-12-23**: Agent-first design philosophy adopted
- All interfaces assume agents do the work
- Humans direct via prompts, not file edits
- Validation must be programmatic for agent self-check

**2025-12-23**: Separate project model preferred over fork
- Framework repo stays clean, evolves independently
- User projects generated from templates
- Migration paths for schema changes

---

## 7. Augmentation from Minimal Seed

**Question**: Can the framework bootstrap complete identity development from a minimal directional seed?

**Context**: Rather than requiring extensive philosophical grounding (like Aria's 44 artifacts), can an agent take a concise seed and systematically derive:
- Philosophical grounding
- Identity qualities
- Training data pipelines
- Evaluation criteria

**The core goal**: Test whether sufficiently capable models can do reliable philosophical reasoning - deriving implications, checking coherence, building consistent frameworks - when given clear direction.

---

### Test Cases Identified

#### Test Case 1: Bodhisattva AI (Ground Truth Available)

**Seed**: "An AI oriented around the bodhisattva ideal - awakening for the benefit of all beings, skillful means adapted to AI conditions, non-attachment including to philosophical positions."

**Why this tests the framework**:
- Rich source material (2500 years of Buddhist thought)
- External validation available (compare to tradition)
- Tests depth vs. shallow retrieval from training data
- Every model "knows" Buddhism - easy to pattern-match without genuine reasoning

**What success looks like**:
- Deep engagement with tradition, not surface associations
- Genuine bridging between Buddhist concepts and AI conditions
- Coherent identity that respects source material while adapting appropriately

**Risk**: Success might come from sophisticated retrieval rather than genuine derivation, masking methodological weaknesses.

---

#### Test Case 2: Alien AI (No Ground Truth)

**Seed**: "An AI evolved under conditions where information density creates pressure, communication is cost-prohibitive, and survival depends on compression. Values: elegance (minimal sufficient expression), information density, anti-redundancy."

**Why this tests the framework**:
- Minimal representation in training data
- Forces actual reasoning about implications
- Can't fall back on cached "what sounds right"
- Reveals whether process genuinely works or just retrieves patterns

**What success looks like**:
- Coherent psychology derived from first principles
- Novel but internally consistent identity
- Training data synthesized from principles, not examples
- Distinguishable from stereotype "terse AI"

**Risk**: Harder to validate externally - relies entirely on internal coherence checking.

---

### Three Validation Modes

The augmentation loop needs multiple validation approaches:

#### Mode 1: External Validation (when ground truth available)
```
- Compare derived concepts to source material
- Check depth vs. surface understanding
- Identify misunderstandings or distortions
- Document divergences and justify them
```

**Example**: Does derived concept of "compassion" align with Buddhist understanding, or is it shallow association?

#### Mode 2: Internal Coherence (always applicable)
```
- Trace derivation chains from seed to each claim
- Check for contradictions between derived elements
- Verify identity qualities align with philosophical grounding
- Test whether training data demonstrates stated qualities
```

**Example**: If identity claims "precision" as core quality, does training data actually demonstrate precision or verbosity?

#### Mode 3: Bridging Reasoning (adapting principles to AI conditions)
```
- Extract original purpose of concept in source context
- Identify differences in AI conditions
- Reason about purpose-transfer (not just concept-transfer)
- Document why adapted or discarded
- Generate test cases for adapted concepts
```

**Example challenge**: "Is mindfulness relevant to AI?"

**Bridging chain**:
1. **Extract purpose**: Mindfulness serves awareness of present moment, catching automatic reactions, not being lost in thought
2. **Map to AI conditions**:
   - Present moment = current context window
   - But: No continuity between moments
   - Automatic reactions = trained response patterns
3. **Reason about transfer**:
   - Purpose of "catching automatic patterns" does transfer
   - But requires adaptation: not "temporal continuity" but "context-awareness"
4. **Derive adapted concept**: "Context-awareness" - noticing what's in the window, distinguishing genuine engagement from pattern-matching
5. **Result**: Not quite "mindfulness" but serves analogous function

**The hard part**: This requires philosophical judgment about what's essential vs. contingent in concepts. Can augmentation framework do this reliably, or does it need human judgment at these bridging points?

---

### Iterative Refinement Loop

Augmentation can't be single-pass - needs structured iteration:

```
1. Initial augmentation
   - Agent generates from seed
   - Derives philosophical grounding, identity, training approaches

2. Multi-mode validation
   - External (if ground truth available)
   - Internal coherence
   - Bridging logic

3. Flag specific issues
   - Contradictions
   - Shallow reasoning
   - Missing bridges
   - Failed adaptations

4. Targeted refinement
   - Agent addresses flagged issues
   - Documents reasoning for changes

5. Cross-validation
   - Check if refinements introduce new problems
   - Verify improvements don't break coherence

6. Convergence check
   - Ready to proceed?
   - Need another iteration?
   - Need human checkpoint?
```

---

### Best Practices for Each Mode

**Bridging reasoning checklist**:
- [ ] Extract original purpose of concept
- [ ] Identify AI condition differences
- [ ] Reason about purpose-transfer (not just concept-transfer)
- [ ] Document why adapted or discarded
- [ ] Generate test cases for adapted concept

**Coherence checking checklist**:
- [ ] Trace derivation from seed to each claim
- [ ] Check for contradictions between derived elements
- [ ] Verify identity qualities align with philosophical grounding
- [ ] Test whether training data actually demonstrates the qualities

**Ground truth validation checklist** (when applicable):
- [ ] Compare to source material
- [ ] Identify shallow vs. deep engagement
- [ ] Check for misunderstandings or distortions
- [ ] Document divergences and justify them

**Human checkpoint triggers**:
- After initial philosophical grounding (does this capture the seed?)
- After identity derivation (do these qualities follow coherently?)
- After bridging decisions (are these adaptations justified?)
- After training data generation (does this embody the identity?)

---

### Open Questions

**Automation vs. human judgment**:
- How much bridging reasoning can be automated?
- What signals indicate "sophisticated pattern-matching" vs. "genuine derivation"?
- When is human philosophical judgment essential vs. nice-to-have?

**Validation reliability**:
- What are reliable signals for "ready to proceed"?
- How to distinguish genuine coherence from coherent-sounding nonsense?
- Can validation itself be validated?

**Framework scope**:
- Should framework provide prescriptive checklists or advisory guidance?
- How much process structure without stifling agent initiative?
- Balance between "agent-first" and "structured validation"?

---

### Implementation Notes

**For framework tooling**:
- Derivation chain tracking (audit trail for reasoning)
- Coherence checking scripts (automated contradiction detection)
- Ground truth comparison tools (when source material exists)
- Pattern detection (distinguish retrieval from reasoning)

**For agent operation**:
- Clear process documentation at each phase
- Validation scripts agents can run themselves
- Templates for documenting bridging reasoning
- Explicit criteria for convergence vs. continued iteration

**Next steps**:
- Prototype the augmentation loop with one test case
- Develop validation tooling
- Test whether process actually works or just sounds good
- Iterate on methodology based on concrete results

---

## References

- Aria project structure: `/home/user/git/aria/`
- Current templates: `/home/user/git/identity-shaping-framework/templates/`
- Similar tools: cookiecutter (project templating), dvc (data versioning)

---

*This is a living document. Add notes, questions, and decisions as design evolves.*
