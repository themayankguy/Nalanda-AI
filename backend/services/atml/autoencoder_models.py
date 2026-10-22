"""PyTorch Autoencoder Architectures for ATML Unit 2.

Implements all core Autoencoder variants from Unit 2 of the Advanced Topics in
Machine Learning syllabus:
1. Undercomplete Autoencoder (Non-linear dimensionality reduction bottleneck)
2. Variational Autoencoder (VAE, probabilistic latent space, reparameterization)
3. Denoising Autoencoder (DAE, noise manifold projection for OCR cleanup)
4. Sparse Autoencoder (L1 activation sparsity constraints)
"""
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class UndercompleteAutoencoder(nn.Module):
    """Undercomplete Autoencoder for nonlinear bottleneck dimensionality reduction.
    
    Maps high-dimensional sentence embeddings (e.g. 384-d MiniLM) through a constrained
    latent bottleneck (e.g. 32-d) to learn the underlying academic knowledge manifold.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: 384 -> 128 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Decoder: 32 -> 128 -> 384
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project input into compressed latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct original vector from latent representation."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstructed_x, latent_z)."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample Mean Squared Error (MSE) reconstruction loss."""
        with torch.no_grad():
            x_recon, _ = self.forward(x)
            # Row-wise MSE
            error = torch.mean((x - x_recon) ** 2, dim=-1)
            return error


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder (VAE) for probabilistic latent space modeling.
    
    Applies the reparameterization trick to sample z ~ N(mu, Sigma) and optimizes
    the Evidence Lower Bound (ELBO) combining reconstruction fidelity with KL divergence.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Shared encoder backbone
        self.encoder_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Latent Gaussian parameter projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps input x to distribution parameters (mu, logvar)."""
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std, where eps ~ N(0, I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct original vector from latent sample z."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstructed_x, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Computes Negative ELBO = MSE Reconstruction Loss + beta * KL Divergence."""
        mse_loss = F.mse_loss(recon_x, x, reduction="mean")
        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        total_loss = mse_loss + beta * kl_div
        return {
            "loss": total_loss,
            "recon_loss": mse_loss,
            "kl_loss": kl_div,
        }


class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder (DAE) for manifold projection and OCR noise healing.
    
    Trained to reconstruct pristine sentence embeddings from noise-corrupted variants
    (Gaussian jitter or dropout), pulling off-manifold OCR corrupted tokens back
    onto the valid academic manifold.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        latent_dim: int = 48,
        noise_std: float = 0.15,
        mask_prob: float = 0.10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.noise_std = noise_std
        self.mask_prob = mask_prob

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

    def inject_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Injects Gaussian noise and random feature zero-masking."""
        if not self.training:
            return x
        # Additive Gaussian noise
        noise = torch.randn_like(x) * self.noise_std
        noisy_x = x + noise
        # Feature dropout
        if self.mask_prob > 0:
            mask = (torch.rand_like(x) > self.mask_prob).float()
            noisy_x = noisy_x * mask
        return noisy_x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass injecting noise in training, returning (cleaned_x, latent_z)."""
        noisy_x = self.inject_noise(x)
        z = self.encoder(noisy_x)
        x_recon = self.decoder(z)
        return x_recon, z

    def denoise(self, noisy_vector: torch.Tensor) -> torch.Tensor:
        """Inference-mode projection of a noisy embedding onto the clean manifold."""
        self.eval()
        with torch.no_grad():
            z = self.encoder(noisy_vector)
            cleaned = self.decoder(z)
            # Re-normalize to unit length like standard embeddings
            norm = torch.norm(cleaned, p=2, dim=-1, keepdim=True) + 1e-8
            return cleaned / norm


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with L1 penalty on bottleneck activations.
    
    Enforces that only a small subset of latent units fire for any given passage,
    disentangling distinct conceptual topics in the lecture corpus.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        sparsity_weight: float = 1e-4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),  # Enforces non-negative activations for L1 sparsity
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse = F.mse_loss(recon_x, x)
        l1_sparse = torch.mean(torch.abs(z))
        total = mse + self.sparsity_weight * l1_sparse
        return {"loss": total, "mse": mse, "sparsity_loss": l1_sparse}
