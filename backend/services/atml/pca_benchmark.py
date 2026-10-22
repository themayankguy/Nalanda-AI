"""Comparison of PCA with Autoencoders for ATML Unit 2.

Directly addresses the syllabus specification:
"Comparison of PCA with Autoencoders, Applications of Autoencoders"

Provides empirical benchmarks comparing linear orthogonal projection (PCA via SVD)
versus non-linear deep representation learning (Undercomplete Autoencoder).
"""
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from backend.services.atml.autoencoder_models import UndercompleteAutoencoder


def compute_pca_reconstruction(
    X: np.ndarray,
    n_components: int = 32,
) -> Tuple[np.ndarray, float]:
    """Compute PCA reconstruction using Singular Value Decomposition (SVD).
    
    X_centered = X - mean
    U, S, Vt = svd(X_centered)
    Z = X_centered @ V[:, :n_components]
    X_recon = Z @ V[:, :n_components].T + mean
    """
    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean

    # Compute SVD: X = U @ diag(S) @ Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V_k = Vt[:n_components, :].T  # shape (D, k)

    # Encode & Decode
    Z = np.dot(X_centered, V_k)
    X_recon = np.dot(Z, V_k.T) + mean

    # Explained variance ratio of the top k components
    total_var = np.sum(S ** 2)
    explained_var = np.sum(S[:n_components] ** 2) / (total_var + 1e-10)

    return X_recon, float(explained_var)


def compare_pca_vs_autoencoder(
    embeddings: np.ndarray,
    latent_dim: int = 32,
    epochs: int = 35,
    learning_rate: float = 0.005,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Empirically compare linear PCA vs. non-linear Autoencoder on a set of embeddings.
    
    Args:
        embeddings: 2D array of shape (N, D) representing document or query vectors.
        latent_dim: Number of bottleneck dimensions (e.g. 32).
        epochs: Training epochs for the lightweight autoencoder.
        learning_rate: Adam optimizer learning rate.
        device: PyTorch device ('cpu', 'mps', or 'cuda').
        
    Returns:
        Dictionary containing comparative MSE, cosine reconstruction similarity,
        training runtime, and theoretical architectural differences.
    """
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be a 2D array of shape (N, D)")

    N, D = embeddings.shape
    if N < 2:
        raise ValueError("At least 2 embedding samples are required for PCA comparison.")

    k = min(latent_dim, N - 1, D)

    # 1. Evaluate PCA
    pca_start = time.perf_counter()
    pca_recon, pca_var_ratio = compute_pca_reconstruction(embeddings, n_components=k)
    pca_time_ms = (time.perf_counter() - pca_start) * 1000.0

    pca_mse = float(np.mean((embeddings - pca_recon) ** 2))
    # Cosine similarity between original and reconstructed
    norm_orig = np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8
    norm_pca = np.linalg.norm(pca_recon, axis=-1, keepdims=True) + 1e-8
    pca_cosine = float(np.mean(np.sum((embeddings / norm_orig) * (pca_recon / norm_pca), axis=-1)))

    # 2. Evaluate Non-linear Autoencoder
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    ae_model = UndercompleteAutoencoder(input_dim=D, hidden_dim=128, latent_dim=k).to(device)
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    tensor_x = torch.from_numpy(embeddings.astype(np.float32)).to(device)

    ae_start = time.perf_counter()
    ae_model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon, _ = ae_model(tensor_x)
        loss = criterion(recon, tensor_x)
        loss.backward()
        optimizer.step()

    ae_time_ms = (time.perf_counter() - ae_start) * 1000.0

    ae_model.eval()
    with torch.no_grad():
        recon_tensor, _ = ae_model(tensor_x)
        ae_recon = recon_tensor.cpu().numpy()

    ae_mse = float(np.mean((embeddings - ae_recon) ** 2))
    norm_ae = np.linalg.norm(ae_recon, axis=-1, keepdims=True) + 1e-8
    ae_cosine = float(np.mean(np.sum((embeddings / norm_orig) * (ae_recon / norm_ae), axis=-1)))

    return {
        "dataset_metadata": {
            "num_samples": N,
            "original_dimension": D,
            "latent_dimension": k,
            "compression_ratio": round(D / k, 2),
        },
        "pca": {
            "method": "Principal Component Analysis (PCA)",
            "linearity": "Linear Orthogonal Projection",
            "mse": round(pca_mse, 6),
            "reconstruction_cosine_similarity": round(pca_cosine, 4),
            "explained_variance_ratio": round(pca_var_ratio, 4),
            "runtime_ms": round(pca_time_ms, 2),
        },
        "autoencoder": {
            "method": "Undercomplete Neural Autoencoder",
            "linearity": "Non-linear Deep Representation (LeakyReLU + LayerNorm)",
            "mse": round(ae_mse, 6),
            "reconstruction_cosine_similarity": round(ae_cosine, 4),
            "runtime_ms": round(ae_time_ms, 2),
            "epochs": epochs,
        },
        "theoretical_comparison": {
            "key_distinctions": [
                "PCA is strictly restricted to orthogonal linear hyperplanes, whereas Autoencoders model non-linear manifold curvature.",
                "PCA provides an exact analytical closed-form solution via SVD; Autoencoders optimize iteratively via gradient descent.",
                "Autoencoders naturally support generative extensions (VAE), noise regularization (DAE), and sparsity constraints (Sparse AE)."
            ]
        }
    }


def format_comparison_table(metrics: Dict[str, Any]) -> str:
    """Generate clean Markdown comparison table suitable for IEEE reports or viva evaluation."""
    meta = metrics["dataset_metadata"]
    pca = metrics["pca"]
    ae = metrics["autoencoder"]

    table = (
        f"### Comparison: PCA vs. Autoencoder (ATML Unit 2 Benchmark)\n\n"
        f"**Input Dimension:** {meta['original_dimension']} → **Latent Dimension:** {meta['latent_dimension']} "
        f"(Compression: {meta['compression_ratio']}x across {meta['num_samples']} passages)\n\n"
        f"| Evaluation Metric | PCA (Linear SVD) | Undercomplete Autoencoder (Non-Linear) |\n"
        f"| :--- | :---: | :---: |\n"
        f"| **Linearity** | {pca['linearity']} | {ae['linearity']} |\n"
        f"| **Reconstruction MSE** | `{pca['mse']}` | `{ae['mse']}` |\n"
        f"| **Cosine Reconstruction** | `{pca['reconstruction_cosine_similarity']}` | `{ae['reconstruction_cosine_similarity']}` |\n"
        f"| **Computation Time** | `{pca['runtime_ms']} ms` | `{ae['runtime_ms']} ms` |\n"
    )
    return table
