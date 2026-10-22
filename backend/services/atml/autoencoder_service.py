"""Autoencoder Outlier & Anomaly Detection Service for Nalanda AI.

Directly addresses ATML Unit 2:
"Outlier detection using Autoencoders, Applications of Autoencoders"

Provides folder-level out-of-distribution (OOD) detection for:
1. Student Chat Queries: Intercepts out-of-syllabus questions (<2ms).
2. Document Ingestion: Identifies mismatched/off-topic PDF uploads.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn

from backend.services.atml.autoencoder_models import UndercompleteAutoencoder, VariationalAutoencoder

logger = logging.getLogger("nalanda.atml.autoencoders")

# Device resolution
_DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# In-memory model cache: key=(user_id, folder_id) -> dict(model, threshold, mean, std)
_AUTOENCODER_CACHE: Dict[str, Dict[str, Any]] = {}


class AutoencoderOutlierDetector:
    """Manages training, caching, and inference of Autoencoder Outlier Detectors per folder."""

    @classmethod
    def get_or_train_folder_model(
        cls,
        embeddings: np.ndarray,
        cache_key: str,
        weights_path: Optional[Path] = None,
        force_retrain: bool = False,
        epochs: int = 35,
        latent_dim: int = 32,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve existing cached autoencoder or train a new one on folder embeddings.
        
        Args:
            embeddings: 2D array of shape (N, 384) representing all passages in folder.
            cache_key: Unique identifier string for folder (e.g. folder_id).
            weights_path: Optional path to save/load weights on disk (e.g. autoencoder.pt).
            force_retrain: If True, forces retraining even if cached.
        """
        if not force_retrain and cache_key in _AUTOENCODER_CACHE:
            return _AUTOENCODER_CACHE[cache_key]

        if embeddings is None or len(embeddings.shape) != 2 or embeddings.shape[0] < 3:
            logger.debug("Insufficient embeddings (<3) to train robust folder autoencoder.")
            return None

        N, D = embeddings.shape
        k = min(latent_dim, N - 1, D)

        # Check if saved model exists on disk
        if weights_path and weights_path.exists() and not force_retrain:
            try:
                checkpoint = torch.load(weights_path, map_location=_DEVICE)
                if checkpoint.get("latent_dim") == k:
                    model = UndercompleteAutoencoder(input_dim=D, hidden_dim=128, latent_dim=k).to(_DEVICE)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.eval()
                    mean_e = checkpoint["mean_error"]
                    std_e = checkpoint["std_error"]
                    # 3-sigma statistical control threshold
                    thresh = float(mean_e + 3.0 * max(std_e, 0.0005))
                    entry = {
                        "model": model,
                        "threshold": thresh,
                        "mean_error": mean_e,
                        "std_error": std_e,
                        "latent_dim": k,
                    }
                    _AUTOENCODER_CACHE[cache_key] = entry
                    return entry
            except Exception as exc:
                logger.warning("Could not load cached autoencoder from %s: %s", weights_path, exc)

        # Train a new lightweight model
        model = UndercompleteAutoencoder(input_dim=D, hidden_dim=128, latent_dim=k).to(_DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
        criterion = nn.MSELoss()

        tensor_x = torch.from_numpy(embeddings.astype(np.float32)).to(_DEVICE)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            recon, _ = model(tensor_x)
            loss = criterion(recon, tensor_x)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            recon_all, _ = model(tensor_x)
            # Per-sample MSE error
            errors = torch.mean((tensor_x - recon_all) ** 2, dim=-1).cpu().numpy()

        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))
        # 3-sigma statistical outlier threshold (Gaussian 99.7% confidence interval)
        threshold = float(mean_err + 3.0 * max(std_err, 0.0005))

        entry = {
            "model": model,
            "threshold": threshold,
            "mean_error": mean_err,
            "std_error": std_err,
            "latent_dim": k,
        }
        _AUTOENCODER_CACHE[cache_key] = entry

        # Save to disk if path provided
        if weights_path:
            try:
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "threshold": threshold,
                    "mean_error": mean_err,
                    "std_error": std_err,
                    "latent_dim": k,
                }, weights_path)
            except Exception as exc:
                logger.warning("Could not persist autoencoder weights to %s: %s", weights_path, exc)

        logger.info(
            "Trained ATML Autoencoder for %s (N=%d, MSE=%.5f, Threshold=%.5f)",
            cache_key, N, mean_err, threshold,
        )
        return entry

    @classmethod
    def evaluate_vector_outlier(
        cls,
        vector: np.ndarray,
        detector_entry: Dict[str, Any],
        sensitivity_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """Check if an individual embedding vector is an outlier.
        
        Args:
            vector: 1D array of shape (D,) or 2D array (1, D).
            detector_entry: Cache entry with model and threshold.
            sensitivity_multiplier: Scales threshold (e.g. 1.2 for looser threshold).
        """
        model = detector_entry["model"]
        base_threshold = detector_entry["threshold"]
        threshold = base_threshold * sensitivity_multiplier

        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)

        model.eval()
        with torch.no_grad():
            tensor_v = torch.from_numpy(vector.astype(np.float32)).to(_DEVICE)
            recon, _ = model(tensor_v)
            error = float(torch.mean((tensor_v - recon) ** 2).item())

        deviation = float((error - detector_entry["mean_error"]) / (detector_entry["std_error"] + 1e-6))
        # Outlier triggered if error exceeds threshold or z-score >= 3.0
        is_outlier = bool(error > threshold or deviation >= 3.0)

        return {
            "is_outlier": is_outlier,
            "reconstruction_error": round(error, 5),
            "threshold": round(threshold, 5),
            "baseline_mean_error": round(detector_entry["mean_error"], 5),
            "standard_deviations_away": round(deviation, 2),
        }

    @classmethod
    def evaluate_document_compatibility(
        cls,
        doc_embeddings: np.ndarray,
        detector_entry: Dict[str, Any],
        outlier_ratio_threshold: float = 0.65,
    ) -> Dict[str, Any]:
        """Evaluate if an entire uploaded document matches the course folder manifold.
        
        If > 65% of chunks trigger outlier reconstruction errors, the document is flagged
        as out-of-distribution (e.g., cooking recipe uploaded into Machine Learning folder).
        """
        if doc_embeddings is None or len(doc_embeddings.shape) != 2 or doc_embeddings.shape[0] == 0:
            return {"is_mismatched_document": False, "outlier_ratio": 0.0}

        model = detector_entry["model"]
        threshold = detector_entry["threshold"]

        model.eval()
        with torch.no_grad():
            tensor_docs = torch.from_numpy(doc_embeddings.astype(np.float32)).to(_DEVICE)
            recon, _ = model(tensor_docs)
            errors = torch.mean((tensor_docs - recon) ** 2, dim=-1).cpu().numpy()

        outlier_flags = errors > threshold
        outlier_count = int(np.sum(outlier_flags))
        total_chunks = len(errors)
        outlier_ratio = float(outlier_count / total_chunks)

        is_mismatched = bool(outlier_ratio >= outlier_ratio_threshold)

        return {
            "is_mismatched_document": is_mismatched,
            "outlier_ratio": round(outlier_ratio, 3),
            "outlier_chunk_count": outlier_count,
            "total_chunk_count": total_chunks,
            "mean_document_error": round(float(np.mean(errors)), 5),
            "folder_threshold": round(threshold, 5),
        }
