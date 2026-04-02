"""Tests for embedding classifier save/load (training) and CLI train command."""

from pathlib import Path

import pytest

sentence_transformers = pytest.importorskip("sentence_transformers")

import yaml
from click.testing import CliRunner

from mmrouter.classifier.embeddings import EmbeddingClassifier
from mmrouter.cli import cli
from mmrouter.models import Category, ClassificationResult, Complexity


EXAMPLES_PATH = Path(__file__).resolve().parent.parent.parent / "eval_data" / "embedding_examples.yaml"


@pytest.fixture(scope="module")
def classifier():
    return EmbeddingClassifier(examples_path=EXAMPLES_PATH, k=5)


@pytest.fixture
def mini_yaml(tmp_path):
    """Create a minimal YAML training file."""
    data = [
        {"prompt": "What is 2+2?", "complexity": "simple", "category": "factual"},
        {"prompt": "Write a poem about rain.", "complexity": "medium", "category": "creative"},
        {"prompt": "Explain quantum entanglement in detail.", "complexity": "complex", "category": "reasoning"},
        {"prompt": "Implement a linked list in Python.", "complexity": "medium", "category": "code"},
        {"prompt": "What color is the sky?", "complexity": "simple", "category": "factual"},
        {"prompt": "Design a distributed cache system.", "complexity": "complex", "category": "code"},
    ]
    yaml_path = tmp_path / "train.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return yaml_path


class TestSaveLoad:
    def test_save_creates_files(self, classifier, tmp_path):
        out_dir = tmp_path / "model"
        classifier.save(out_dir)
        assert (out_dir / "embeddings.npz").exists()
        assert (out_dir / "metadata.json").exists()

    def test_save_returns_path(self, classifier, tmp_path):
        out_dir = tmp_path / "model"
        result = classifier.save(out_dir)
        assert result == out_dir

    def test_save_metadata_content(self, classifier, tmp_path):
        import json

        out_dir = tmp_path / "model"
        classifier.save(out_dir)

        with open(out_dir / "metadata.json") as f:
            meta = json.load(f)

        assert meta["model_name"] == "all-MiniLM-L6-v2"
        assert meta["k"] == 5
        assert meta["num_examples"] == len(classifier._prompts)
        assert len(meta["prompts"]) == meta["num_examples"]
        assert len(meta["complexities"]) == meta["num_examples"]
        assert len(meta["categories"]) == meta["num_examples"]
        assert "timestamp" in meta

    def test_load_roundtrip_classify(self, classifier, tmp_path):
        out_dir = tmp_path / "model"
        classifier.save(out_dir)

        loaded = EmbeddingClassifier.load(out_dir)
        result = loaded.classify("What is the capital of Germany?")
        assert isinstance(result, ClassificationResult)
        assert result.complexity == Complexity.SIMPLE
        assert result.category == Category.FACTUAL

    def test_load_preserves_k(self, tmp_path):
        clf = EmbeddingClassifier(examples_path=EXAMPLES_PATH, k=3)
        clf.save(tmp_path / "model")
        loaded = EmbeddingClassifier.load(tmp_path / "model")
        assert loaded.k == 3

    def test_load_preserves_prompts(self, classifier, tmp_path):
        out_dir = tmp_path / "model"
        classifier.save(out_dir)
        loaded = EmbeddingClassifier.load(out_dir)
        assert loaded._prompts == classifier._prompts

    def test_load_preserves_labels(self, classifier, tmp_path):
        out_dir = tmp_path / "model"
        classifier.save(out_dir)
        loaded = EmbeddingClassifier.load(out_dir)
        assert loaded._complexities == classifier._complexities
        assert loaded._categories == classifier._categories

    def test_load_same_classification(self, classifier, tmp_path):
        """Loaded classifier produces identical results to original."""
        out_dir = tmp_path / "model"
        classifier.save(out_dir)
        loaded = EmbeddingClassifier.load(out_dir)

        prompts = [
            "What is the capital of Germany?",
            "Write a haiku about mountains.",
            "Implement a binary search function in Python.",
        ]
        for prompt in prompts:
            orig = classifier.classify(prompt)
            new = loaded.classify(prompt)
            assert orig.complexity == new.complexity, f"Mismatch for: {prompt}"
            assert orig.category == new.category, f"Mismatch for: {prompt}"

    def test_load_missing_metadata_raises(self, tmp_path):
        import numpy as np

        out_dir = tmp_path / "model"
        out_dir.mkdir()
        np.savez(out_dir / "embeddings.npz", embeddings=np.zeros((1, 10)))
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            EmbeddingClassifier.load(out_dir)

    def test_load_missing_embeddings_raises(self, tmp_path):
        import json

        out_dir = tmp_path / "model"
        out_dir.mkdir()
        with open(out_dir / "metadata.json", "w") as f:
            json.dump({"model_name": "x", "k": 5, "prompts": [], "complexities": [], "categories": []}, f)
        with pytest.raises(FileNotFoundError, match="Embeddings file not found"):
            EmbeddingClassifier.load(out_dir)

    def test_save_creates_parent_dirs(self, classifier, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        classifier.save(deep)
        assert (deep / "embeddings.npz").exists()


class TestCLITrain:
    def test_train_basic(self, mini_yaml, tmp_path):
        out_dir = tmp_path / "trained"
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--data", str(mini_yaml), "--output", str(out_dir)])
        assert result.exit_code == 0, result.output
        assert "Training complete" in result.output
        assert (out_dir / "embeddings.npz").exists()
        assert (out_dir / "metadata.json").exists()

    def test_train_custom_k(self, mini_yaml, tmp_path):
        out_dir = tmp_path / "trained"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "train", "--data", str(mini_yaml), "--output", str(out_dir), "--k", "3",
        ])
        assert result.exit_code == 0, result.output
        assert "k: 3" in result.output

    def test_train_with_eval_split(self, mini_yaml, tmp_path):
        out_dir = tmp_path / "trained"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "train", "--data", str(mini_yaml), "--output", str(out_dir), "--eval-split", "0.3",
        ])
        assert result.exit_code == 0, result.output
        assert "Eval on held-out split:" in result.output
        assert "Accuracy:" in result.output

    def test_train_eval_split_too_high(self, mini_yaml, tmp_path):
        out_dir = tmp_path / "trained"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "train", "--data", str(mini_yaml), "--output", str(out_dir), "--eval-split", "0.8",
        ])
        assert result.exit_code != 0
        assert "must be between" in result.output

    def test_train_shows_config_hint(self, mini_yaml, tmp_path):
        out_dir = tmp_path / "trained"
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--data", str(mini_yaml), "--output", str(out_dir)])
        assert result.exit_code == 0, result.output
        assert "strategy: embeddings" in result.output
        assert "trained_model:" in result.output

    def test_trained_model_usable_via_load(self, mini_yaml, tmp_path):
        """Train via CLI, then load the result and classify."""
        out_dir = tmp_path / "trained"
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--data", str(mini_yaml), "--output", str(out_dir)])
        assert result.exit_code == 0, result.output

        loaded = EmbeddingClassifier.load(out_dir)
        r = loaded.classify("What is 2+2?")
        assert isinstance(r, ClassificationResult)
