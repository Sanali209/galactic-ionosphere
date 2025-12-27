from typing import Literal, Dict, List, Optional
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import joblib
import os
import warnings
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class VectorReducer:
    """
    Универсальный класс для проекции векторов с учетом обученных PCA или UMAP.
    Поддерживает сохранение/загрузку моделей с учётом метода.
    """

    def __init__(
        self,
        target_dim: int = 512,
        method: Literal["pca", "svd", "umap"] = "svd"
    ):
        self.target_dim = target_dim
        self.method = method
        self.models: Dict[str, object] = {}

    def _model_filename(self, name: str) -> str:
        return f"{name}.{self.method}.model"

    def fit(self, name: str, vectors: np.ndarray):
        d = min(self.target_dim, vectors.shape[1])

        if self.method == "pca":
            model = PCA(n_components=d).fit(vectors)
        elif self.method == "svd":
            model = TruncatedSVD(n_components=d).fit(vectors)
        elif self.method == "umap":
            if not HAS_UMAP:
                raise ImportError("UMAP is not installed. Use: pip install umap-learn")
            model = umap.UMAP(n_components=d, random_state=42).fit(vectors)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.models[name] = model

    def transform(self, name: str, vector: np.ndarray) -> np.ndarray:
        if name not in self.models:
            raise ValueError(f"Model '{name}' not trained. Call fit() first.")

        reduced = self.models[name].transform(vector.reshape(1, -1)).flatten()

        if reduced.shape[0] < self.target_dim:
            padded = np.zeros(self.target_dim, dtype=reduced.dtype)
            padded[:reduced.shape[0]] = reduced
            return padded

        return reduced

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        meta = {
            "target_dim": self.target_dim,
            "method": self.method,
            "model_names": list(self.models.keys())
        }
        joblib.dump(meta, os.path.join(folder, "meta.pkl"))

        for name, model in self.models.items():
            filename = self._model_filename(name)
            joblib.dump(model, os.path.join(folder, filename))

    def load(self, folder: str):
        meta = joblib.load(os.path.join(folder, "meta.pkl"))
        self.target_dim = meta["target_dim"]
        self.method = meta["method"]
        self.models = {}

        for name in meta["model_names"]:
            filename = self._model_filename(name)
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
            else:
                warnings.warn(f"Model file missing: {filename}")


class EmbeddingFusion:
    """
    Класс для взвешенного объединения эмбеддингов разных типов.
    Поддерживает предварительную проекцию через обученный VectorReducer.
    """

    def __init__(self, reducer: VectorReducer):
        self.reducer = reducer
        self.embeddings: Dict[str, np.ndarray] = {}
        self.weights: Dict[str, float] = {}

    def add_embedding(self, name: str, vector: np.ndarray, weight: float = 1.0):
        self.embeddings[name] = vector
        self.weights[name] = weight

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            warnings.warn("Zero-norm vector encountered.")
            return v
        return v / norm

    def fuse(self) -> np.ndarray:
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            raise ValueError("Total weight must be > 0")

        fused = np.zeros(self.reducer.target_dim, dtype=np.float32)

        for name, vec in self.embeddings.items():
            reduced = self.reducer.transform(name, vec)
            normed = self._normalize(reduced)
            fused += self.weights[name] * normed

        fused /= total_weight
        return self._normalize(fused)

    def clear(self):
        self.embeddings.clear()
        self.weights.clear()