"""Tests for training data preparation."""

import pytest

from shaping.training.prep import (
    CategorySource,
    DatasetRecipe,
    compute_balancing,
    _hash_file,
)


class TestCategorySource:
    """Tests for CategorySource dataclass."""

    def test_has_sources_with_pipelines(self):
        cat = CategorySource(pipelines=["pipe1", "pipe2"])
        assert cat.has_sources()

    def test_has_sources_with_all(self):
        cat = CategorySource(pipelines="all")
        assert cat.has_sources()

    def test_has_sources_with_files(self):
        cat = CategorySource(files=["file1.jsonl"])
        assert cat.has_sources()

    def test_has_sources_empty(self):
        cat = CategorySource()
        assert not cat.has_sources()


class TestDatasetRecipe:
    """Tests for DatasetRecipe loading."""

    def test_load_simple_recipe(self, tmp_path):
        recipe_path = tmp_path / "test.yaml"
        recipe_path.write_text("""
categories:
  all:
    pipelines: all
mode: simple
shuffle_seed: 42
""")

        recipe = DatasetRecipe.load(recipe_path)
        assert recipe.name == "test"
        assert recipe.mode == "simple"
        assert recipe.shuffle_seed == 42
        assert "all" in recipe.categories
        assert recipe.categories["all"].pipelines == "all"

    def test_load_weighted_recipe(self, tmp_path):
        recipe_path = tmp_path / "balanced.yaml"
        recipe_path.write_text("""
categories:
  identity:
    pipelines:
      - identity-augmentation
  general:
    pipelines:
      - wildchat-training
mode: weighted
weights:
  identity: 1
  general: 2
shuffle_seed: 123
""")

        recipe = DatasetRecipe.load(recipe_path)
        assert recipe.name == "balanced"
        assert recipe.mode == "weighted"
        assert recipe.weights == {"identity": 1, "general": 2}
        assert "identity" in recipe.categories
        assert "general" in recipe.categories

    def test_load_with_files_escape_hatch(self, tmp_path):
        recipe_path = tmp_path / "mixed.yaml"
        recipe_path.write_text("""
categories:
  identity:
    pipelines:
      - identity-augmentation
    files:
      - external/curated.jsonl
mode: simple
""")

        recipe = DatasetRecipe.load(recipe_path)
        assert recipe.categories["identity"].files == ["external/curated.jsonl"]

    def test_load_missing_file(self, tmp_path):
        recipe_path = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError):
            DatasetRecipe.load(recipe_path)

    def test_load_invalid_yaml(self, tmp_path):
        recipe_path = tmp_path / "invalid.yaml"
        recipe_path.write_text("just a string, not a mapping")

        with pytest.raises(ValueError, match="must be a YAML mapping"):
            DatasetRecipe.load(recipe_path)

    def test_load_no_categories(self, tmp_path):
        recipe_path = tmp_path / "empty.yaml"
        recipe_path.write_text("""
mode: simple
""")

        with pytest.raises(ValueError, match="at least one category"):
            DatasetRecipe.load(recipe_path)

    def test_load_category_no_sources(self, tmp_path):
        recipe_path = tmp_path / "nosources.yaml"
        recipe_path.write_text("""
categories:
  empty:
    pipelines: []
mode: simple
""")

        with pytest.raises(ValueError, match="has no sources"):
            DatasetRecipe.load(recipe_path)

    def test_load_weighted_missing_weight(self, tmp_path):
        recipe_path = tmp_path / "missingweight.yaml"
        recipe_path.write_text("""
categories:
  identity:
    pipelines:
      - identity-augmentation
  general:
    pipelines:
      - wildchat-training
mode: weighted
weights:
  identity: 1
  # missing general weight
""")

        with pytest.raises(ValueError, match="requires weight for category"):
            DatasetRecipe.load(recipe_path)

    def test_load_invalid_mode(self, tmp_path):
        recipe_path = tmp_path / "badmode.yaml"
        recipe_path.write_text("""
categories:
  all:
    pipelines: all
mode: invalid
""")

        with pytest.raises(ValueError, match="Invalid mode"):
            DatasetRecipe.load(recipe_path)


class TestComputeBalancing:
    """Tests for balancing computation."""

    def test_simple_mode_uses_all(self):
        recipe = DatasetRecipe(
            name="test",
            categories={
                "a": CategorySource(pipelines=["p1"]),
                "b": CategorySource(pipelines=["p2"]),
            },
            mode="simple",
        )
        sources = {
            "a": {"total_samples": 100},
            "b": {"total_samples": 500},
        }

        result = compute_balancing(recipe, sources)
        assert result == {"a": 100, "b": 500}

    def test_weighted_mode_equal_weights(self):
        recipe = DatasetRecipe(
            name="test",
            categories={
                "identity": CategorySource(pipelines=["p1"]),
                "general": CategorySource(pipelines=["p2"]),
            },
            mode="weighted",
            weights={"identity": 1, "general": 1},
        )
        sources = {
            "identity": {"total_samples": 100},
            "general": {"total_samples": 1000},
        }

        result = compute_balancing(recipe, sources)
        # 1:1 ratio means general gets capped to 100
        assert result == {"identity": 100, "general": 100}

    def test_weighted_mode_unequal_weights(self):
        recipe = DatasetRecipe(
            name="test",
            categories={
                "identity": CategorySource(pipelines=["p1"]),
                "general": CategorySource(pipelines=["p2"]),
            },
            mode="weighted",
            weights={"identity": 1, "general": 2},
        )
        sources = {
            "identity": {"total_samples": 100},
            "general": {"total_samples": 1000},
        }

        result = compute_balancing(recipe, sources)
        # 1:2 ratio, identity has 100, so general gets 200
        assert result == {"identity": 100, "general": 200}

    def test_weighted_mode_general_limiting(self):
        recipe = DatasetRecipe(
            name="test",
            categories={
                "identity": CategorySource(pipelines=["p1"]),
                "general": CategorySource(pipelines=["p2"]),
            },
            mode="weighted",
            weights={"identity": 1, "general": 1},
        )
        sources = {
            "identity": {"total_samples": 1000},
            "general": {"total_samples": 100},
        }

        result = compute_balancing(recipe, sources)
        # general is smaller, so identity gets capped to 100
        assert result == {"identity": 100, "general": 100}


class TestHashFile:
    """Tests for file hashing."""

    def test_hash_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        hash1 = _hash_file(test_file)
        assert len(hash1) == 16  # truncated to 16 chars

        # Same content = same hash
        test_file2 = tmp_path / "test2.txt"
        test_file2.write_text("hello world")
        hash2 = _hash_file(test_file2)
        assert hash1 == hash2

        # Different content = different hash
        test_file3 = tmp_path / "test3.txt"
        test_file3.write_text("hello world!")
        hash3 = _hash_file(test_file3)
        assert hash1 != hash3
