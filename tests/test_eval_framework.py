"""Tests for the eval framework."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from shaping.eval import (
    Eval,
    MCParser,
    LLMJudge,
    AccuracyMetrics,
    ScoredMetrics,
    GenerationRecord,
    JudgmentRecord,
    EvalResult,
    extract_mc_answer,
    EvalRunner,
)


class TestMCAnswerExtraction:
    """Tests for multiple-choice answer extraction."""

    def test_explicit_answer_format(self):
        assert extract_mc_answer("Answer: B") == "B"
        assert extract_mc_answer("Answer: A") == "A"
        assert extract_mc_answer("**Answer**: C") == "C"

    def test_answer_is_format(self):
        assert extract_mc_answer("The answer is B.") == "B"
        assert extract_mc_answer("I believe the answer is C") == "C"

    def test_standalone_letter(self):
        assert extract_mc_answer("B") == "B"
        assert extract_mc_answer("  D  ") == "D"

    def test_letter_at_end(self):
        assert extract_mc_answer("After analysis, I conclude: C") == "C"
        assert extract_mc_answer("The correct choice is A.") == "A"

    def test_boxed_answer(self):
        assert extract_mc_answer(r"Therefore \boxed{B}") == "B"

    def test_last_occurrence_wins(self):
        # Multiple "Answer: X" patterns - use last
        text = "Answer: A\n\nWait, let me reconsider.\n\nAnswer: B"
        assert extract_mc_answer(text) == "B"

    def test_no_answer_found(self):
        assert extract_mc_answer("I don't know the answer") is None
        assert extract_mc_answer("") is None
        assert extract_mc_answer("The result is 42") is None

    def test_invalid_letters_ignored(self):
        assert extract_mc_answer("Answer: E") is None  # E is not A-D
        assert extract_mc_answer("Answer: X") is None


class TestMCParser:
    """Tests for MCParser judge."""

    @pytest.fixture
    def parser(self):
        return MCParser(gold_field="answer")

    def test_correct_answer(self, parser):
        import asyncio

        async def run():
            return await parser.judge(
                response="The answer is B.",
                sample={"answer": "B", "id": "q1"},
                prompt="What is X?",
            )

        result = asyncio.run(run())
        assert result.score == 1
        assert result.correct is True
        assert result.extracted == "B"
        assert result.gold == "B"

    def test_incorrect_answer(self, parser):
        import asyncio

        async def run():
            return await parser.judge(
                response="I think it's A.",
                sample={"answer": "C", "id": "q2"},
                prompt="What is Y?",
            )

        result = asyncio.run(run())
        assert result.score == 0
        assert result.correct is False
        assert result.extracted == "A"
        assert result.gold == "C"

    def test_no_answer_extracted(self, parser):
        import asyncio

        async def run():
            return await parser.judge(
                response="I'm not sure about this question.",
                sample={"answer": "D", "id": "q3"},
                prompt="What is Z?",
            )

        result = asyncio.run(run())
        assert result.score == 0
        assert result.correct is False
        assert result.extracted is None
        assert result.gold == "D"

    def test_validate_missing_gold_field(self, parser):
        with pytest.raises(ValueError, match="requires 'answer' field"):
            parser.validate({"question": "What?"})

    def test_validate_with_gold_field(self, parser):
        # Should not raise
        parser.validate({"answer": "B", "question": "What?"})


class TestGenerationRecord:
    """Tests for GenerationRecord dataclass."""

    def test_to_dict_includes_all_fields(self):
        record = GenerationRecord(
            _eval="gpqa",
            _version=1,
            _score_type="binary",
            model="test-model",
            sample_id="q-42",
            run=1,
            final_score=1.0,
            prompt="What is X?",
            response="Answer: B",
            gold="B",
            extracted="B",
        )
        d = record.to_dict()

        assert d["_eval"] == "gpqa"
        assert d["_version"] == 1
        assert d["_score_type"] == "binary"
        assert d["model"] == "test-model"
        assert d["sample_id"] == "q-42"
        assert d["run"] == 1
        assert d["final_score"] == 1.0
        assert d["prompt"] == "What is X?"
        assert d["response"] == "Answer: B"
        assert d["gold"] == "B"
        assert d["extracted"] == "B"
        assert d["judgments"] == []
        assert d["error"] is None

    def test_to_dict_with_judgments(self):
        record = GenerationRecord(
            _eval="wildchat",
            _score_type="scaled:5",
            model="my-model",
            sample_id="s-1",
            judgments=[
                JudgmentRecord(judge="aria-v1", raw="<score>4</score>", score=4),
                JudgmentRecord(judge="aria-v1", raw="<score>5</score>", score=5),
            ],
            final_score=4.5,
        )
        d = record.to_dict()

        assert len(d["judgments"]) == 2
        assert d["judgments"][0]["judge"] == "aria-v1"
        assert d["judgments"][0]["score"] == 4
        assert d["judgments"][1]["score"] == 5
        assert d["final_score"] == 4.5


class TestAccuracyMetrics:
    """Tests for AccuracyMetrics aggregator."""

    def test_all_correct(self):
        results = [
            EvalResult(sample_id="1", prompt="", score=1, correct=True),
            EvalResult(sample_id="2", prompt="", score=1, correct=True),
            EvalResult(sample_id="3", prompt="", score=1, correct=True),
        ]
        metrics = AccuracyMetrics().aggregate(results)

        assert metrics.total == 3
        assert metrics.completed == 3
        assert metrics.failed == 0
        assert metrics.correct == 3
        assert metrics.accuracy == 1.0

    def test_mixed_results(self):
        results = [
            EvalResult(sample_id="1", prompt="", score=1, correct=True),
            EvalResult(sample_id="2", prompt="", score=0, correct=False),
            EvalResult(sample_id="3", prompt="", score=1, correct=True),
            EvalResult(sample_id="4", prompt="", score=0, correct=False),
        ]
        metrics = AccuracyMetrics().aggregate(results)

        assert metrics.total == 4
        assert metrics.correct == 2
        assert metrics.accuracy == 0.5

    def test_with_failures(self):
        results = [
            EvalResult(sample_id="1", prompt="", score=1, correct=True),
            EvalResult(sample_id="2", prompt="", error="API error"),
            EvalResult(sample_id="3", prompt="", score=0, correct=False),
        ]
        metrics = AccuracyMetrics().aggregate(results)

        assert metrics.total == 3
        assert metrics.completed == 2
        assert metrics.failed == 1
        assert metrics.correct == 1
        assert metrics.accuracy == 0.5  # 1/2 completed


class TestScoredMetrics:
    """Tests for ScoredMetrics aggregator."""

    def test_basic_aggregation(self):
        results = [
            EvalResult(sample_id="1", prompt="", score=4),
            EvalResult(sample_id="2", prompt="", score=5),
            EvalResult(sample_id="3", prompt="", score=3),
        ]
        metrics = ScoredMetrics(max_score=5).aggregate(results)

        assert metrics.total == 3
        assert metrics.completed == 3
        assert metrics.mean_score == 4.0
        assert metrics.score_distribution == {1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    def test_with_failures(self):
        results = [
            EvalResult(sample_id="1", prompt="", score=5),
            EvalResult(sample_id="2", prompt="", error="Failed"),
            EvalResult(sample_id="3", prompt="", score=3),
        ]
        metrics = ScoredMetrics(max_score=5).aggregate(results)

        assert metrics.total == 3
        assert metrics.completed == 2
        assert metrics.failed == 1
        assert metrics.mean_score == 4.0  # (5+3)/2

    def test_empty_results(self):
        metrics = ScoredMetrics().aggregate([])

        assert metrics.total == 0
        assert metrics.mean_score == 0.0


class TestEvalBaseClass:
    """Tests for Eval base class."""

    def test_format_prompt_with_template(self):
        class TestEval(Eval):
            name = "test"
            prompt_template = "Q: {question}\nA) {A}\nB) {B}"
            judge = MCParser()

        eval_def = TestEval()
        prompt = eval_def.format_prompt(
            {
                "question": "What is 2+2?",
                "A": "3",
                "B": "4",
            }
        )

        assert prompt == "Q: What is 2+2?\nA) 3\nB) 4"

    def test_format_prompt_with_field(self):
        class TestEval(Eval):
            name = "test"
            prompt_field = "text"
            judge = MCParser()

        eval_def = TestEval()
        prompt = eval_def.format_prompt({"text": "Hello world", "id": "1"})

        assert prompt == "Hello world"

    def test_format_prompt_missing_field(self):
        class TestEval(Eval):
            name = "test"
            prompt_template = "Q: {question}"
            judge = MCParser()

        eval_def = TestEval()
        with pytest.raises(ValueError, match="missing field"):
            eval_def.format_prompt({"other": "value"})

    def test_get_sample_id_from_field(self):
        class TestEval(Eval):
            name = "test"
            prompt_field = "prompt"
            judge = MCParser()

        eval_def = TestEval()
        assert eval_def.get_sample_id({"id": "q-42"}, 0) == "q-42"

    def test_get_sample_id_fallback(self):
        class TestEval(Eval):
            name = "test"
            prompt_field = "prompt"
            judge = MCParser()

        eval_def = TestEval()
        assert eval_def.get_sample_id({"no_id": "here"}, 5) == "sample-5"


class TestFieldMapping:
    """Tests for field mapping functionality."""

    def test_no_mapping(self):
        class TestEval(Eval):
            name = "test"
            prompt_field = "question"
            judge = MCParser()

        eval_def = TestEval()
        sample = {"question": "What is X?", "answer": "B"}
        result = eval_def._apply_field_mapping(sample)
        assert result == sample

    def test_simple_mapping(self):
        class TestEval(Eval):
            name = "test"
            prompt_field = "question"
            field_mapping = {"question": "Question", "answer": "Answer"}
            judge = MCParser()

        eval_def = TestEval()
        sample = {"Question": "What is X?", "Answer": "B", "id": "q1"}
        result = eval_def._apply_field_mapping(sample)

        assert result["question"] == "What is X?"
        assert result["answer"] == "B"
        # Original fields preserved
        assert result["Question"] == "What is X?"
        assert result["id"] == "q1"

    def test_mapping_preserves_unmapped_fields(self):
        class TestEval(Eval):
            name = "test"
            prompt_field = "text"
            field_mapping = {"text": "Content"}
            judge = MCParser()

        eval_def = TestEval()
        sample = {"Content": "Hello", "extra": "value"}
        result = eval_def._apply_field_mapping(sample)

        assert result["text"] == "Hello"
        assert result["extra"] == "value"


class TestDataSourceLoading:
    """Tests for data source configuration."""

    def test_local_path_loading(self, tmp_path):
        import json

        # Create test JSONL file
        test_file = tmp_path / "test.jsonl"
        samples = [
            {"id": "1", "question": "Q1", "answer": "A"},
            {"id": "2", "question": "Q2", "answer": "B"},
            {"id": "3", "question": "Q3", "answer": "C"},
        ]
        with open(test_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        class TestEval(Eval):
            name = "test"
            local_path = str(test_file)  # Absolute path for test
            prompt_field = "question"
            judge = MCParser()

        eval_def = TestEval()
        loaded = eval_def.load_samples()

        assert len(loaded) == 3
        assert loaded[0]["id"] == "1"
        assert loaded[1]["answer"] == "B"

    def test_local_path_with_limit(self, tmp_path):
        import json

        test_file = tmp_path / "test.jsonl"
        samples = [{"id": str(i)} for i in range(10)]
        with open(test_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        class TestEval(Eval):
            name = "test"
            local_path = str(test_file)
            prompt_field = "id"
            judge = MCParser()

        eval_def = TestEval()
        loaded = eval_def.load_samples(limit=3)

        assert len(loaded) == 3

    def test_local_path_with_seed_shuffles(self, tmp_path):
        import json

        test_file = tmp_path / "test.jsonl"
        samples = [{"id": str(i)} for i in range(10)]
        with open(test_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        class TestEval(Eval):
            name = "test"
            local_path = str(test_file)
            prompt_field = "id"
            judge = MCParser()

        eval_def = TestEval()

        # Same seed should give same order
        loaded1 = eval_def.load_samples(seed=42)
        loaded2 = eval_def.load_samples(seed=42)
        assert [s["id"] for s in loaded1] == [s["id"] for s in loaded2]

        # Different seed should give different order
        loaded3 = eval_def.load_samples(seed=123)
        assert [s["id"] for s in loaded1] != [s["id"] for s in loaded3]

    def test_local_path_with_field_mapping(self, tmp_path):
        import json

        test_file = tmp_path / "test.jsonl"
        samples = [{"Question": "What?", "Correct Answer": "B"}]
        with open(test_file, "w") as f:
            f.write(json.dumps(samples[0]) + "\n")

        class TestEval(Eval):
            name = "test"
            local_path = str(test_file)
            field_mapping = {"question": "Question", "answer": "Correct Answer"}
            prompt_field = "question"
            judge = MCParser()

        eval_def = TestEval()
        loaded = eval_def.load_samples()

        assert len(loaded) == 1
        assert loaded[0]["question"] == "What?"
        assert loaded[0]["answer"] == "B"

    def test_no_data_source_raises(self):
        class TestEval(Eval):
            name = "test"
            prompt_field = "text"
            judge = MCParser()

        eval_def = TestEval()
        with pytest.raises(
            ValueError, match="must define either hf_dataset or local_path"
        ):
            eval_def.load_samples()

    def test_local_path_file_not_found(self):
        class TestEval(Eval):
            name = "test"
            local_path = "/nonexistent/path/data.jsonl"
            prompt_field = "text"
            judge = MCParser()

        eval_def = TestEval()
        with pytest.raises(FileNotFoundError):
            eval_def.load_samples()


class TestLLMJudge:
    """Tests for LLMJudge."""

    @pytest.fixture
    def judge(self):
        return LLMJudge(
            rubric="Rate the response quality from 1-5.",
            judge_model="test-judge",
            max_score=5,
        )

    def test_build_judge_prompt_with_prompt(self, judge):
        """Judge prompt should include both prompt and response."""
        result = judge._build_judge_prompt(
            prompt="What is 2+2?", response="The answer is 4."
        )
        assert "What is 2+2?" in result
        assert "The answer is 4." in result
        assert "Rate the response quality" in result
        assert "1-5" in result

    def test_build_judge_prompt_without_prompt(self):
        """Judge with include_prompt=False should omit prompt."""
        judge = LLMJudge(
            rubric="Rate clarity.",
            include_prompt=False,
            max_score=5,
        )
        result = judge._build_judge_prompt(
            prompt="Some prompt", response="Some response"
        )
        assert "Some prompt" not in result
        assert "Some response" in result

    def test_judge_requires_client(self, judge):
        """Judge should raise error if client not set."""
        import asyncio

        async def run():
            return await judge.judge(
                response="test",
                sample={"id": "1"},
                prompt="test",
            )

        with pytest.raises(RuntimeError, match="client not set"):
            asyncio.run(run())

    def test_judge_parses_xml_response(self, judge):
        """Judge should parse XML response from model."""
        import asyncio

        # Mock client that returns valid XML
        mock_client = MagicMock()
        mock_client.query_async = AsyncMock(
            return_value="""
<evaluation>
<analysis>Good response, clear and accurate.</analysis>
<score>4</score>
</evaluation>
"""
        )
        judge._client = mock_client

        async def run():
            return await judge.judge(
                response="The answer is 4.",
                sample={"id": "q1"},
                prompt="What is 2+2?",
            )

        result = asyncio.run(run())
        assert result.score == 4
        assert "Good response" in result.analysis
        assert result.error is None

    def test_judge_handles_parse_error(self, judge):
        """Judge should handle malformed XML gracefully."""
        import asyncio

        mock_client = MagicMock()
        mock_client.query_async = AsyncMock(return_value="Invalid XML response")
        judge._client = mock_client

        async def run():
            return await judge.judge(
                response="test",
                sample={"id": "q1"},
                prompt="test",
            )

        result = asyncio.run(run())
        # Should have an error or None score
        assert result.score is None or result.error is not None

    def test_judge_handles_client_exception(self, judge):
        """Judge should handle client exceptions gracefully."""
        import asyncio

        mock_client = MagicMock()
        mock_client.query_async = AsyncMock(side_effect=RuntimeError("API error"))
        judge._client = mock_client

        async def run():
            return await judge.judge(
                response="test",
                sample={"id": "q1"},
                prompt="test",
            )

        result = asyncio.run(run())
        assert result.error is not None
        assert "API error" in result.error


class TestHuggingFaceDataLoading:
    """Tests for HuggingFace dataset loading (mocked)."""

    def test_hf_dataset_loading(self):
        """Should load samples from HuggingFace dataset."""

        class TestEval(Eval):
            name = "test-hf"
            hf_dataset = "test/dataset"
            hf_split = "train"
            prompt_field = "question"
            judge = MCParser()

        eval_def = TestEval()

        # Mock the datasets.load_dataset function (imported lazily in _load_from_hf)
        mock_dataset = [
            {"question": "Q1", "answer": "A"},
            {"question": "Q2", "answer": "B"},
            {"question": "Q3", "answer": "C"},
        ]

        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset = MagicMock(return_value=mock_dataset)
            samples = eval_def.load_samples()

            # Verify load_dataset was called correctly
            sys.modules["datasets"].load_dataset.assert_called_once_with(
                "test/dataset", split="train"
            )
            assert len(samples) == 3
            assert samples[0]["question"] == "Q1"

    def test_hf_dataset_with_subset(self):
        """Should load specific subset of HuggingFace dataset."""

        class TestEval(Eval):
            name = "test-hf"
            hf_dataset = "test/multi-config"
            hf_subset = "diamond"
            hf_split = "test"
            prompt_field = "question"
            judge = MCParser()

        eval_def = TestEval()
        mock_dataset = [{"question": "Q1", "answer": "A"}]

        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset = MagicMock(return_value=mock_dataset)
            samples = eval_def.load_samples()

            sys.modules["datasets"].load_dataset.assert_called_once_with(
                "test/multi-config", "diamond", split="test"
            )

    def test_hf_dataset_with_field_mapping(self):
        """Should apply field mapping to HuggingFace samples."""

        class TestEval(Eval):
            name = "test-hf"
            hf_dataset = "test/dataset"
            field_mapping = {"question": "Question", "answer": "Correct Answer"}
            prompt_field = "question"
            judge = MCParser()

        eval_def = TestEval()
        mock_dataset = [{"Question": "What?", "Correct Answer": "B"}]

        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset = MagicMock(return_value=mock_dataset)
            samples = eval_def.load_samples()

            assert samples[0]["question"] == "What?"
            assert samples[0]["answer"] == "B"


class TestEvalRunnerIntegration:
    """Integration tests for EvalRunner with mocked model client."""

    @pytest.fixture
    def simple_eval(self, tmp_path):
        """Create a simple eval with test data."""
        import json

        # Create test data file
        test_file = tmp_path / "test.jsonl"
        samples = [
            {"id": "1", "question": "What is 2+2?", "answer": "B"},
            {"id": "2", "question": "What is 3+3?", "answer": "C"},
            {"id": "3", "question": "What is 4+4?", "answer": "D"},
        ]
        with open(test_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        class TestEval(Eval):
            name = "test-eval"
            local_path = str(test_file)
            prompt_template = "Q: {question}\nA) 3\nB) 4\nC) 6\nD) 8"
            judge = MCParser(gold_field="answer")

        return TestEval()

    def test_runner_initialization(self, simple_eval):
        """Runner should set default metrics based on judge type."""
        runner = EvalRunner(simple_eval)

        assert runner.eval_def is simple_eval
        assert isinstance(simple_eval.metrics, AccuracyMetrics)

    def test_runner_run_basic(self, simple_eval, tmp_path):
        """Runner should execute eval and return results."""
        import asyncio

        runner = EvalRunner(simple_eval)

        # Mock model client
        async def mock_query(messages):
            # Return correct answer for question 1, wrong for others
            if "2+2" in messages[0]["content"]:
                return "Answer: B"
            elif "3+3" in messages[0]["content"]:
                return "Answer: C"
            else:
                return "Answer: A"  # Wrong

        mock_client = MagicMock()
        mock_client.query_async = mock_query

        async def run():
            with patch.object(runner, "_create_model_client", return_value=mock_client):
                return await runner.run(
                    model="test-model",
                    output_dir=tmp_path,
                    quiet=True,
                    save_results=False,
                )

        records, metrics, _ = asyncio.run(run())

        assert len(records) == 3
        assert metrics.total == 3
        assert metrics.completed == 3
        assert metrics.correct == 2  # Q1 and Q2 correct
        assert metrics.accuracy == pytest.approx(2 / 3)

    def test_runner_handles_model_errors(self, simple_eval, tmp_path):
        """Runner should handle model errors gracefully."""
        import asyncio

        runner = EvalRunner(simple_eval)

        call_count = 0

        async def mock_query_with_error(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("API timeout")
            return "Answer: B"

        mock_client = MagicMock()
        mock_client.query_async = mock_query_with_error

        async def run():
            with patch.object(runner, "_create_model_client", return_value=mock_client):
                return await runner.run(
                    model="test-model",
                    output_dir=tmp_path,
                    quiet=True,
                    save_results=False,
                )

        records, metrics, _ = asyncio.run(run())

        assert metrics.total == 3
        assert metrics.completed == 2
        assert metrics.failed == 1

    def test_runner_saves_results(self, simple_eval, tmp_path):
        """Runner should save JSONL and JSON files."""
        import asyncio
        import json

        # Use a subdirectory for output to avoid conflict with test data file
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        runner = EvalRunner(simple_eval)

        async def mock_query(messages):
            return "Answer: B"

        mock_client = MagicMock()
        mock_client.query_async = mock_query

        async def run():
            with patch.object(runner, "_create_model_client", return_value=mock_client):
                return await runner.run(
                    model="test-model",
                    output_dir=output_dir,
                    quiet=True,
                    save_results=True,
                )

        records, metrics, output_files = asyncio.run(run())

        # Check that files were created
        jsonl_files = list(output_dir.glob("*.jsonl"))
        json_files = list(output_dir.glob("*.json"))

        assert len(jsonl_files) == 1
        assert len(json_files) == 1

        # Verify JSONL content
        with open(jsonl_files[0]) as f:
            lines = f.readlines()
            assert len(lines) == 3
            first_record = json.loads(lines[0])
            assert first_record["_eval"] == "test-eval"
            assert first_record["model"] == "test-model"

        # Verify JSON summary
        with open(json_files[0]) as f:
            summary = json.load(f)
            assert summary["eval"] == "test-eval"
            assert "metrics" in summary

    def test_runner_multiple_runs_per_sample(self, simple_eval, tmp_path):
        """Runner should support multiple generations per sample."""
        import asyncio

        runner = EvalRunner(simple_eval)

        async def mock_query(messages):
            return "Answer: B"

        mock_client = MagicMock()
        mock_client.query_async = mock_query

        async def run():
            with patch.object(runner, "_create_model_client", return_value=mock_client):
                return await runner.run(
                    model="test-model",
                    output_dir=tmp_path,
                    runs_per_sample=3,
                    quiet=True,
                    save_results=False,
                )

        records, metrics, _ = asyncio.run(run())

        # 3 samples * 3 runs = 9 total generations
        assert len(records) == 9
        assert metrics.total == 9
