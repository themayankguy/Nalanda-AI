"""Denoising Autoencoder (DAE) for OCR Noise Cleanup for ATML Unit 2.

Directly addresses the syllabus specification:
"Types of Autoencoders: Denoising Autoencoders, Applications of Autoencoders"

Provides self-healing projection for OCR-corrupted passages and noisy scanned slide decks.
Takes feature representations corrupted by OCR artifacts (e.g. character substitutions,
typos, and noise) and projects them back onto the pristine academic vocabulary manifold.
"""
import logging
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from backend.services.atml.autoencoder_models import DenoisingAutoencoder

logger = logging.getLogger("nalanda.atml.denoiser")

_DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class OCREmbeddingDenoiser:
    """Manages training and inference for Denoising Autoencoder (DAE) on OCR text embeddings."""

    def __init__(
        self,
        input_dim: int = 384,
        latent_dim: int = 48,
        noise_std: float = 0.15,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = DenoisingAutoencoder(
            input_dim=input_dim,
            hidden_dim=128,
            latent_dim=latent_dim,
            noise_std=noise_std,
            mask_prob=0.08,
        ).to(_DEVICE)
        self.is_trained = False

    def train_on_corpus(
        self,
        clean_embeddings: np.ndarray,
        epochs: int = 40,
        learning_rate: float = 0.005,
    ) -> float:
        """Train the DAE on clean academic text embeddings with synthetic noise injection."""
        if len(clean_embeddings.shape) != 2 or clean_embeddings.shape[0] < 2:
            return 0.0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        tensor_x = torch.from_numpy(clean_embeddings.astype(np.float32)).to(_DEVICE)
        self.model.train()

        final_loss = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            recon, _ = self.model(tensor_x)
            loss = criterion(recon, tensor_x)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        self.is_trained = True
        self.model.eval()
        logger.info("Trained Denoising Autoencoder on %d passages (Final MSE Loss: %.5f)", len(clean_embeddings), final_loss)
        return final_loss

    def repair_noisy_embeddings(self, noisy_embeddings: np.ndarray) -> np.ndarray:
        """Inference projection: Reconstruct clean embeddings from noisy OCR embeddings."""
        if not self.is_trained:
            # If not yet trained, return normalized original
            norms = np.linalg.norm(noisy_embeddings, axis=-1, keepdims=True) + 1e-8
            return noisy_embeddings / norms

        self.model.eval()
        with torch.no_grad():
            tensor_noisy = torch.from_numpy(noisy_embeddings.astype(np.float32)).to(_DEVICE)
            cleaned_tensor = self.model.denoise(tensor_noisy)
            return cleaned_tensor.cpu().numpy()


# Global singleton instance
_GLOBAL_DENOISER = OCREmbeddingDenoiser()


def get_ocr_denoiser() -> OCREmbeddingDenoiser:
    """Retrieve shared global OCR Denoising Autoencoder instance."""
    return _GLOBAL_DENOISER
