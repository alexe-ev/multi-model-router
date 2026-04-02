"""Embedding-based classifier: kNN over labeled example embeddings."""

from __future__ import annotations

import json
import time
from pathlib import Path

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as _import_error:
    raise ImportError(
        "sentence-transformers is required for EmbeddingClassifier. "
        "Install it with: pip install -e \".[embeddings]\""
    ) from _import_error

import yaml

from mmrouter.classifier import ClassifierBase
from mmrouter.models import Category, ClassificationResult, Complexity


_DEFAULT_EXAMPLES_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "eval_data" / "embedding_examples.yaml"
)


class EmbeddingClassifier(ClassifierBase):
    """Classify prompts using cosine similarity against labeled example embeddings (kNN)."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        examples_path: str | None = None,
        k: int = 5,
    ) -> None:
        self.k = k
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

        path = Path(examples_path) if examples_path is not None else _DEFAULT_EXAMPLES_PATH
        with open(path) as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(f"Examples file must be a non-empty YAML list: {path}")

        self._prompts: list[str] = []
        self._complexities: list[Complexity] = []
        self._categories: list[Category] = []

        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {i} in examples file is not a dict")
            try:
                self._prompts.append(entry["prompt"])
                self._complexities.append(Complexity(entry["complexity"]))
                self._categories.append(Category(entry["category"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Entry {i} in examples file is invalid: {exc}") from exc

        # Pre-compute and normalize embeddings for vectorized cosine similarity
        raw_embeddings = self._model.encode(self._prompts, convert_to_numpy=True)
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings: np.ndarray = raw_embeddings / norms  # shape: (N, D)

    def save(self, path: str | Path) -> Path:
        """Save pre-computed embeddings (.npz) and metadata (.json) to directory.

        Returns the directory path where files were saved.
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(out_dir / "embeddings.npz", embeddings=self._embeddings)

        metadata = {
            "model_name": self._model_name,
            "k": self.k,
            "num_examples": len(self._prompts),
            "timestamp": time.time(),
            "prompts": self._prompts,
            "complexities": [c.value for c in self._complexities],
            "categories": [c.value for c in self._categories],
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return out_dir

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingClassifier":
        """Load a pre-trained classifier from saved embeddings directory.

        Skips YAML parsing and re-encoding. Only loads the sentence-transformers
        model for inference.
        """
        load_dir = Path(path)
        meta_path = load_dir / "metadata.json"
        emb_path = load_dir / "embeddings.npz"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

        with open(meta_path) as f:
            metadata = json.load(f)

        model_name = metadata["model_name"]
        k = metadata.get("k", 5)
        prompts = metadata["prompts"]
        complexities = [Complexity(c) for c in metadata["complexities"]]
        categories = [Category(c) for c in metadata["categories"]]

        data = np.load(emb_path)
        embeddings = data["embeddings"]

        # Build instance without calling __init__ (skip YAML + encoding)
        instance = object.__new__(cls)
        instance.k = k
        instance._model_name = model_name
        instance._model = SentenceTransformer(model_name)
        instance._prompts = prompts
        instance._complexities = complexities
        instance._categories = categories
        instance._embeddings = embeddings

        return instance

    def classify(self, prompt: str) -> ClassificationResult:
        prompt = prompt.strip()
        if not prompt:
            return ClassificationResult(
                complexity=Complexity.SIMPLE,
                category=Category.FACTUAL,
                confidence=0.5,
            )

        # Embed and normalize the query
        query_vec = self._model.encode([prompt], convert_to_numpy=True)[0]
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Cosine similarity against all examples (dot product of normalized vectors)
        similarities: np.ndarray = self._embeddings @ query_vec  # shape: (N,)

        k = min(self.k, len(self._prompts))
        top_k_indices = np.argpartition(similarities, -k)[-k:]

        top_complexities = [self._complexities[i] for i in top_k_indices]
        top_categories = [self._categories[i] for i in top_k_indices]

        # Majority vote: pick the most frequent label in each dimension
        complexity = _majority(top_complexities)
        category = _majority(top_categories)

        # Confidence: fraction of k neighbors that agree with majority
        comp_conf = top_complexities.count(complexity) / k
        cat_conf = top_categories.count(category) / k
        confidence = round((comp_conf + cat_conf) / 2, 4)

        return ClassificationResult(
            complexity=complexity,
            category=category,
            confidence=confidence,
        )


def _majority(labels: list) -> object:
    """Return the most common element in a list. Ties broken by first occurrence."""
    counts: dict = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)
