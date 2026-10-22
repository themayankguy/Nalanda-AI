"""Automated Test Suite for ATML Unit 2: Autoencoders & Outlier Detection.

Verifies:
1. Undercomplete Autoencoder (forward pass, compression, reconstruction MSE).
2. Variational Autoencoder (VAE, reparameterization trick, ELBO loss, KL divergence).
3. Denoising Autoencoder (DAE, synthetic noise injection, manifold restoration).
4. Sparse Autoencoder (L1 activation sparsity constraints).
5. PCA vs. Autoencoder empirical benchmark comparison and reporting.
6. Autoencoder Outlier & Anomaly Detection on in-distribution vs. out-of-distribution queries.
7. Document compatibility evaluation for mismatched course uploads.
8. OCR embedding denoising pipeline integration.
"""
import pytest
import numpy as np
import torch

from backend.services.atml.autoencoder_models import (
    UndercompleteAutoencoder,
    VariationalAutoencoder,
    DenoisingAutoencoder,
    SparseAutoencoder,
)
from backend.services.atml.pca_benchmark import (
    compare_pca_vs_autoencoder,
    format_comparison_table,
    compute_pca_reconstruction,
)
from backend.services.atml.autoencoder_service import AutoencoderOutlierDetector
from backend.services.atml.ocr_denoiser import OCREmbeddingDenoiser, get_ocr_denoiser
from backend.services.NLP.text_normalizer import denoise_ocr_embeddings


class TestUndercompleteAutoencoder:
    """Tests for Undercomplete Autoencoder architecture and reconstruction."""

    def test_dimensions_and_forward_pass(self):
        model = UndercompleteAutoencoder(input_dim=384, hidden_dim=128, latent_dim=32)
        x = torch.randn(5, 384)
        x_recon, z = model(x)

        assert x_recon.shape == (5, 384)
        assert z.shape == (5, 32)

    def test_reconstruction_error_computation(self):
        model = UndercompleteAutoencoder(input_dim=384, hidden_dim=64, latent_dim=16)
        x = torch.randn(4, 384)
        errors = model.compute_reconstruction_error(x)

        assert errors.shape == (4,)
        assert (errors >= 0.0).all()

    def test_training_reduces_loss(self):
        torch.manual_seed(42)
        model = UndercompleteAutoencoder(input_dim=64, hidden_dim=32, latent_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(10, 64)

        # Initial loss
        model.eval()
        initial_loss = float(torch.mean((x - model(x)[0]) ** 2))

        # Train for 20 epochs
        model.train()
        for _ in range(20):
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = torch.mean((x - recon) ** 2)
            loss.backward()
            optimizer.step()

        model.eval()
        final_loss = float(torch.mean((x - model(x)[0]) ** 2))
        assert final_loss < initial_loss


class TestVariationalAutoencoder:
    """Tests for Variational Autoencoder (VAE) probabilistic modeling."""

    def test_vae_forward_and_reparameterization(self):
        vae = VariationalAutoencoder(input_dim=384, hidden_dim=128, latent_dim=32)
        x = torch.randn(6, 384)
        x_recon, mu, logvar = vae(x)

        assert x_recon.shape == (6, 384)
        assert mu.shape == (6, 32)
        assert logvar.shape == (6, 32)

    def test_vae_loss_components(self):
        vae = VariationalAutoencoder(input_dim=64, hidden_dim=32, latent_dim=8)
        x = torch.randn(4, 64)
        recon_x, mu, logvar = vae(x)

        loss_dict = vae.loss_function(recon_x, x, mu, logvar, beta=1.0)
        assert "loss" in loss_dict
        assert "recon_loss" in loss_dict
        assert "kl_loss" in loss_dict
        assert loss_dict["loss"] >= 0.0
        assert loss_dict["kl_loss"] >= 0.0


class TestDenoisingAutoencoder:
    """Tests for Denoising Autoencoder (DAE) noise filtering."""

    def test_noise_injection_in_training_only(self):
        dae = DenoisingAutoencoder(input_dim=64, hidden_dim=32, latent_dim=16, noise_std=0.2)
        x = torch.ones(2, 64)

        # In training mode, noise is injected
        dae.train()
        noisy = dae.inject_noise(x)
        assert not torch.equal(noisy, x)

        # In eval mode, clean vector is passed untouched
        dae.eval()
        clean = dae.inject_noise(x)
        assert torch.equal(clean, x)

    def test_denoise_projection_normalizes_to_unit_length(self):
        dae = DenoisingAutoencoder(input_dim=128, hidden_dim=64, latent_dim=16)
        corrupted = torch.randn(3, 128)
        cleaned = dae.denoise(corrupted)

        assert cleaned.shape == (3, 128)
        # Unit norm check
        norms = torch.norm(cleaned, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


class TestSparseAutoencoder:
    """Tests for Sparse Autoencoder with L1 activation penalty."""

    def test_sparsity_loss(self):
        sae = SparseAutoencoder(input_dim=64, hidden_dim=32, latent_dim=16, sparsity_weight=0.01)
        x = torch.randn(4, 64)
        recon, z = sae(x)

        loss_dict = sae.loss_function(recon, x, z)
        assert "sparsity_loss" in loss_dict
        assert loss_dict["sparsity_loss"] >= 0.0
        assert (z >= 0.0).all()  # ReLU non-negativity


class TestPCABenchmark:
    """Tests for PCA vs. Autoencoder comparative benchmarking."""

    def test_pca_vs_autoencoder_comparison(self):
        np.random.seed(42)
        # 25 samples in 64-dimensional space
        embeddings = np.random.randn(25, 64).astype(np.float32)

        results = compare_pca_vs_autoencoder(embeddings, latent_dim=8, epochs=15, device="cpu")

        assert "dataset_metadata" in results
        assert "pca" in results
        assert "autoencoder" in results
        assert results["dataset_metadata"]["latent_dimension"] == 8
        assert "mse" in results["pca"]
        assert "mse" in results["autoencoder"]

        # Verify formatted table output
        table = format_comparison_table(results)
        assert "### Comparison: PCA vs. Autoencoder" in table
        assert "| **Reconstruction MSE** |" in table


class TestAutoencoderOutlierDetector:
    """Tests for Outlier & Anomaly Detection on academic queries and documents."""

    @pytest.fixture
    def corpus_embeddings(self):
        np.random.seed(42)
        # Simulate a cohesive course topic cluster (mean around 0.5 with low variance)
        base = np.full((30, 384), 0.5, dtype=np.float32)
        noise = np.random.normal(0, 0.05, (30, 384)).astype(np.float32)
        return base + noise

    def test_in_distribution_query_accepted(self, corpus_embeddings):
        entry = AutoencoderOutlierDetector.get_or_train_folder_model(
            embeddings=corpus_embeddings,
            cache_key="test_folder_01",
            force_retrain=True,
            epochs=25,
            latent_dim=16,
        )
        assert entry is not None

        # Sample vector drawn from the exact same in-distribution cluster
        in_dist_vec = np.full((384,), 0.5, dtype=np.float32) + np.random.normal(0, 0.04, (384,)).astype(np.float32)
        res = AutoencoderOutlierDetector.evaluate_vector_outlier(in_dist_vec, entry)

        assert res["is_outlier"] is False
        assert res["reconstruction_error"] <= res["threshold"]

    def test_out_of_distribution_query_rejected(self, corpus_embeddings):
        entry = AutoencoderOutlierDetector.get_or_train_folder_model(
            embeddings=corpus_embeddings,
            cache_key="test_folder_02",
            force_retrain=True,
            epochs=25,
            latent_dim=16,
        )
        assert entry is not None

        # Extreme outlier vector (completely negative / orthogonal to cluster)
        outlier_vec = np.full((384,), -3.0, dtype=np.float32)
        res = AutoencoderOutlierDetector.evaluate_vector_outlier(outlier_vec, entry)

        assert res["is_outlier"] is True
        assert res["reconstruction_error"] > res["threshold"]

    def test_document_compatibility_mismatch_detection(self, corpus_embeddings):
        entry = AutoencoderOutlierDetector.get_or_train_folder_model(
            embeddings=corpus_embeddings,
            cache_key="test_folder_03",
            force_retrain=True,
            epochs=20,
        )

        # Mismatched document: 10 chunks from an alien distribution
        alien_doc_embeddings = np.full((10, 384), -2.5, dtype=np.float32)
        compat = AutoencoderOutlierDetector.evaluate_document_compatibility(
            alien_doc_embeddings,
            entry,
            outlier_ratio_threshold=0.60,
        )

        assert compat["is_mismatched_document"] is True
        assert compat["outlier_ratio"] > 0.60


class TestOCREmbeddingDenoiserIntegration:
    """Tests for OCR Denoising Autoencoder pipeline integration."""

    def test_denoiser_training_and_repair(self):
        np.random.seed(42)
        clean_corpus = np.random.randn(20, 384).astype(np.float32)
        denoiser = OCREmbeddingDenoiser(input_dim=384, latent_dim=32)

        loss = denoiser.train_on_corpus(clean_corpus, epochs=15)
        assert denoiser.is_trained is True

        noisy_input = clean_corpus[:3] + np.random.normal(0, 0.2, (3, 384)).astype(np.float32)
        repaired = denoiser.repair_noisy_embeddings(noisy_input)

        assert repaired.shape == (3, 384)
        # Output should be unit-normalized
        norms = np.linalg.norm(repaired, axis=-1)
        assert np.allclose(norms, np.ones(3), atol=1e-3)

    def test_text_normalizer_denoise_helper(self):
        noisy_vecs = np.random.randn(2, 384).astype(np.float32)
        cleaned = denoise_ocr_embeddings(noisy_vecs)
        assert cleaned.shape == (2, 384)
