"""Evaluation utilities for audio diffusion models.

This module provides:
- Validation loss computation
"""

import hashlib
import contextlib
import io
import logging
import os
from pathlib import Path
import warnings

import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F

# Quiet TensorFlow/absl/BirdNET startup chatter before importing BirdNET.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import birdnet

from dataset import AUDIO_PATH, BIRDNET_EMBEDDINGS_PATH


for _logger_name in ("absl", "tensorflow", "birdnet", "transformers"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


@contextlib.contextmanager
def _suppress_birdnet_noise():
    """Suppress warnings and stdout/stderr noise from BirdNET backends."""
    previous_disable_level = logging.root.manager.disable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    yield
                finally:
                    logging.disable(previous_disable_level)


def compute_validation_loss(model, val_dataloader, noise_scheduler, device, prediction_type):
    """Compute average loss on validation set.
    
    Args:
        model: The diffusion model
        val_dataloader: DataLoader for validation set
        noise_scheduler: The noise scheduler
        device: Device to run evaluation on
    
    Returns:
        Average MSE loss on validation set
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            clean_images = batch.to(device)
            noise = torch.randn(clean_images.shape).to(device)
            bs = clean_images.shape[0]
            
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=device
            ).long()
            
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps)

            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
            else:
                raise ValueError(f"Unsupported prediction type: {prediction_type}")

            loss = F.mse_loss(noise_pred, target)
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    
    Params:
    -- mu1   : Numpy array containing the mean of generated samples.
    -- mu2   : Numpy array containing the mean of reference samples.
    -- sigma1: The covariance matrix for generated samples.
    -- sigma2: The covariance matrix for reference samples.
    
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def compute_embedding_statistics(embeddings):
    """Compute mean and covariance of embeddings.
    
    Args:
        embeddings: Tensor of shape (N, ...) where N is number of samples.
                   Will be flattened to (N, D) where D is the embedding dimension.
    
    Returns:
        mu: Mean vector of shape (D,)
        sigma: Covariance matrix of shape (D, D)
    """
    # Flatten embeddings to (N, D)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    N = embeddings.shape[0]
    embeddings_flat = embeddings.reshape(N, -1)  # (N, D)
    
    mu = np.mean(embeddings_flat, axis=0)
    sigma = np.cov(embeddings_flat, rowvar=False)
    
    return mu, sigma


def compute_fad(generated_embeddings, reference_embeddings):
    """Compute Frechet Distance between generated and reference latent embeddings.
    
    This computes FAD directly on the latent space embeddings without needing
    to decode to audio and use an external audio embedding model.
    
    Args:
        generated_embeddings: Tensor of shape (N, C, H, W) - generated latents
        reference_embeddings: Tensor of shape (M, C, H, W) - reference latents from dataset
    
    Returns:
        FAD score (lower is better), or None if computation fails
    """
    try:
        mu_gen, sigma_gen = compute_embedding_statistics(generated_embeddings)
        mu_ref, sigma_ref = compute_embedding_statistics(reference_embeddings)
        
        fad_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
        return float(fad_score)
    except Exception as e:
        print(f"Warning: FAD computation failed: {e}")
        return None


def get_reference_embeddings(dataset, num_samples=None):
    """Extract embeddings from the dataset for FAD computation.
    
    Args:
        dataset: Dataset containing latent embeddings
        num_samples: Number of samples to use (None = use all)
    
    Returns:
        Tensor of shape (N, C, H, W) containing reference embeddings
    """
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    embeddings = []
    for i in range(num_samples):
        emb = dataset[i]
        embeddings.append(emb)
    
    return torch.stack(embeddings)


DEFAULT_BIRDNET_TARGET_SPECIES = "Turdus merula_Eurasian Blackbird"

def get_birdnet_model():
    """Load BirdNET acoustic model (PB backend, FP32)."""
    with _suppress_birdnet_noise():
        return birdnet.load("acoustic", "2.4", "pb", precision="fp32")


def _to_mono_float32(audio: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    audio = np.asarray(audio)
    if audio.ndim == 1:
        mono = audio
    elif audio.ndim == 2:
        # Accept both [C, N] and [N, C], convert to mono by channel average.
        if audio.shape[0] <= 8:
            mono = audio.mean(axis=0)
        elif audio.shape[1] <= 8:
            mono = audio.mean(axis=1)
        else:
            raise ValueError(f"Unexpected 2D audio shape {audio.shape}.")
    else:
        raise ValueError(f"Expected 1D/2D audio, got shape {audio.shape}.")

    return np.ascontiguousarray(mono.astype(np.float32, copy=False))


def _build_audio_tuples(generated_sounds, sample_rate):
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError("sample_rate must be > 0")
    return [(_to_mono_float32(audio), sr) for audio in generated_sounds]


def compute_birdnet_classification_metrics(
    generated_sounds,
    sample_rate,
    birdnet_model,
    target_species=DEFAULT_BIRDNET_TARGET_SPECIES,
    device="CPU",
):
    """Compute BirdNET cls_conf and cls_error from generated audio.

    cls_conf: median over samples of max target confidence across windows.
    cls_error: median over samples of fraction of windows where top-1 species is not target.
    """
    if len(generated_sounds) == 0:
        return {"cls_conf": float("nan"), "cls_error": float("nan")}

    audio_tuples = _build_audio_tuples(generated_sounds, sample_rate)
    n_inputs = len(audio_tuples)

    # 1) Target confidence trajectory (single species only)
    with _suppress_birdnet_noise():
        target_pred = birdnet_model.predict_arrays(
            audio_tuples,
            device=device,
            top_k=1,
            default_confidence_threshold=-np.inf,
            custom_species_list=[target_species],
        )
        target_df = target_pred.to_dataframe()

    per_input_conf = np.zeros(n_inputs, dtype=np.float64)
    if not target_df.empty:
        grouped = target_df.groupby("input")["confidence"].max()
        for i in range(n_inputs):
            if i in grouped.index:
                per_input_conf[i] = float(grouped.loc[i])

    # 2) Frame-level top1 misclassification ratio
    with _suppress_birdnet_noise():
        top1_pred = birdnet_model.predict_arrays(
            audio_tuples,
            device=device,
            top_k=1,
            default_confidence_threshold=-np.inf,
        )
        top1_df = top1_pred.to_dataframe()

    per_input_err = np.ones(n_inputs, dtype=np.float64)
    if not top1_df.empty:
        top1_df = top1_df.copy()
        top1_df["mis"] = top1_df["species_name"] != target_species
        grouped_err = top1_df.groupby("input")["mis"].mean()
        for i in range(n_inputs):
            if i in grouped_err.index:
                per_input_err[i] = float(grouped_err.loc[i])

    return {
        "cls_conf": float(np.median(per_input_conf)),
        "cls_error": float(np.median(per_input_err)),
    }


def _list_reference_audio_files(reference_audio_dir: Path, max_files: int | None = None):
    exts = {".wav", ".flac", ".ogg", ".opus", ".mp3", ".aiff", ".aifc", ".w64"}
    files = [p for p in reference_audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files = sorted(files)
    if max_files is not None:
        files = files[:max_files]
    return files


def _reference_signature(files):
    h = hashlib.sha256()
    for p in files:
        st = p.stat()
        h.update(str(p).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(st.st_mtime_ns).encode("utf-8"))
    return h.hexdigest()


def _extract_window_embeddings(result):
    embeddings = result.embeddings
    mask = result.embeddings_masked
    valid_window_mask = ~(mask.all(axis=2))
    windows = embeddings[valid_window_mask]
    return np.asarray(windows, dtype=np.float64)


def compute_or_load_reference_birdnet_stats(
    birdnet_model,
    reference_audio_dir=AUDIO_PATH,
    reference_cache_dir=BIRDNET_EMBEDDINGS_PATH,
    max_files=None,
    device="CPU",
):
    """Build or load cached BirdNET reference window-embedding statistics."""
    reference_audio_dir = Path(reference_audio_dir)
    reference_cache_dir = Path(reference_cache_dir)
    reference_cache_dir.mkdir(parents=True, exist_ok=True)

    files = _list_reference_audio_files(reference_audio_dir, max_files=max_files)
    if len(files) == 0:
        raise RuntimeError(f"No reference audio files found in {reference_audio_dir}")

    signature = _reference_signature(files)
    stats_path = reference_cache_dir / "reference_stats.npz"
    embeddings_path = reference_cache_dir / "reference_window_embeddings.npy"
    backend_key = f"{type(birdnet_model).__name__}:{device}"

    if stats_path.exists() and embeddings_path.exists():
        cached = np.load(stats_path, allow_pickle=True)
        if (
            str(cached["signature"]) == signature
            and str(cached["model_backend"]) == backend_key
        ):
            return {
                "mu": cached["mu"],
                "sigma": cached["sigma"],
                "n_windows": int(cached["n_windows"]),
                "signature": signature,
            }

    with _suppress_birdnet_noise():
        encoded = birdnet_model.encode(files)

    windows = _extract_window_embeddings(encoded)
    if windows.shape[0] < 2:
        raise RuntimeError("Need at least 2 BirdNET windows for reference statistics.")

    mu = np.mean(windows, axis=0)
    sigma = np.cov(windows, rowvar=False)

    np.save(embeddings_path, windows.astype(np.float32, copy=False))
    np.savez(
        stats_path,
        mu=mu,
        sigma=sigma,
        n_windows=np.array([windows.shape[0]], dtype=np.int64),
        signature=np.array(signature),
        model_backend=np.array(backend_key),
    )

    return {
        "mu": mu,
        "sigma": sigma,
        "n_windows": int(windows.shape[0]),
        "signature": signature,
    }


def compute_birdnet_fad(
    generated_sounds,
    sample_rate,
    birdnet_model,
    reference_stats,
    device="CPU",
):
    """Compute BirdNET-window FAD without per-file pooling."""
    if len(generated_sounds) == 0:
        return float("nan")

    audio_tuples = _build_audio_tuples(generated_sounds, sample_rate)
    with _suppress_birdnet_noise():
        encoded = birdnet_model.encode_arrays(audio_tuples, device=device)
    gen_windows = _extract_window_embeddings(encoded)

    if gen_windows.shape[0] < 2:
        return float("nan")

    mu_gen = np.mean(gen_windows, axis=0)
    sigma_gen = np.cov(gen_windows, rowvar=False)
    mu_ref = reference_stats["mu"]
    sigma_ref = reference_stats["sigma"]

    return float(calculate_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref))
