"""Evaluation utilities for audio diffusion models.

This module provides:
- Validation loss computation
"""

import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F


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
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

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