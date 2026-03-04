"""Dataset utilities for audio diffusion training.

This module provides:
- Audio loading and preprocessing functions
- Embedding generation and storage
- PyTorch Dataset for loading pre-computed embeddings
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Any, Dict

import numpy as np
import scipy.io.wavfile
import torch
import torchaudio
from einops import rearrange
from torch.utils.data import Dataset
from stable_audio_tools.inference.utils import prepare_audio
from stable_audio_tools.interface.gradio import load_model as _load_model


def load_model(
    pretrained_name: str = "stabilityai/stable-audio-open-1.0",
    pretransform_ckpt_path: Optional[str] = None,
    device: Optional[str] = None,
    model_half: Optional[bool] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load a stable audio model and return parameters needed for embedding generation.
    
    Args:
        pretrained_name: HuggingFace model name or path. Defaults to stable-audio-open-1.0.
        pretransform_ckpt_path: Optional path to a pretransform checkpoint.
        device: Device to load model on. Defaults to "cuda" if available, else "cpu".
        model_half: Whether to use half precision. Defaults to True only on CUDA
                    when MODEL_HALF env var is "1".
    
    Returns:
        Tuple of (model, params_dict) where params_dict contains:
            - sample_rate: Model's expected sample rate
            - sample_size: Model's expected sample size
            - device: Device the model is loaded on
            - seconds_total: Total seconds of audio the model can process
    
    Example:
        >>> model, params = load_model()
        >>> generate_embeddings(
        ...     audio_dir="data/audio",
        ...     save_dir="data/embeddings",
        ...     model=model,
        ...     sample_size=params["sample_size"],
        ...     device=params["device"],
        ... )
    """
    # Resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Half precision only on CUDA
    if model_half is None:
        model_half = bool(os.environ.get("MODEL_HALF", "0") == "1") and (device == "cuda")
    
    # Allow overriding pretransform checkpoint via env var
    if pretransform_ckpt_path is None:
        pretransform_ckpt_path = os.environ.get("PRETRANSFORM_CKPT_PATH", None)
    
    # Load using stable_audio_tools loader
    model, model_config = _load_model(
        pretrained_name=pretrained_name,
        pretransform_ckpt_path=pretransform_ckpt_path,
        device=device,
        model_half=model_half,
    )
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    seconds_total = int(sample_size / sample_rate)
    
    params = {
        "sample_rate": sample_rate,
        "sample_size": sample_size,
        "device": device,
        "seconds_total": seconds_total,
    }
    
    print(f"Loaded pretrained model: {pretrained_name}")
    print(f"Device: {device} | Half: {model_half} | Sample rate: {sample_rate} | Sample size: {sample_size}")
    
    return model, params



def encode_audio(audio: torch.Tensor, sr: int, model, params, device: str) -> torch.Tensor:
    """Encode audio into latent embedding using the model's pretransform.
    
    Args:
        audio: Audio tensor of shape [C, N].
        sr: Sample rate of the input audio.
        model: The stable audio model with pretransform encoder.
        params: Dictionary containing model parameters.
        device: Device to run encoding on.
        
    Returns:
        Encoded latent tensor.
    """
    init_audio = prepare_audio(
        audio,
        in_sr=sr,
        target_sr=model.sample_rate,
        target_length=params['sample_size'],
        target_channels=model.pretransform.io_channels,
        device=device,
    )
    encoded = model.pretransform.encode(init_audio)
    return encoded


def decode_audio(encoding: torch.Tensor, model) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode latent embedding back to audio.
    
    Args:
        encoding: Latent tensor from the pretransform encoder.
        model: The stable audio model with pretransform decoder.
        
    Returns:
        Audio
    """
    audio = model.pretransform.decode(encoding).squeeze(0)
    
    maxval = torch.max(torch.abs(audio))
    if maxval > 0:
        audio = audio / maxval

    # I think jupyter can work with audio in range [-1, 1], so we dont need this
    # if jupyter: # convert to the format expected by IPython.display.Audio
    #     audio = (audio.clamp(-1, 1) * 32767).to(torch.int16).cpu()
    
    return audio


def compute_latent_stats(embeddings_dir: str | Path) -> Tuple[float, float]:
    """Compute mean and std of all embeddings in a directory.
    
    Args:
        embeddings_dir: Directory containing .pt embedding files.
        
    Returns:
        Tuple of (mean, std) as floats.
    """
    embeddings_dir = Path(embeddings_dir)
    files = [f for f in os.listdir(embeddings_dir) if f.endswith(".pt") and f != "latent_stats.pt"]
    
    print("Computing latent statistics...")
    all_data = []
    for f in files:
        data = torch.load(embeddings_dir / f)
        all_data.append(data)
    all_data = torch.cat(all_data, dim=0)
    mean = all_data.mean().item()
    std = all_data.std().item()
    print(f"Latent stats: mean={mean:.4f}, std={std:.4f}")
    return mean, std


def normalize_latents(latents: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Normalize latents to zero mean and unit variance."""
    return (latents - mean) / std


def denormalize_latents(latents: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Inverse of normalize_latents."""
    return latents * std + mean


def generate_embeddings(
    audio_dir: str | Path,
    save_dir: str | Path,
    model,
    params: dict,
    device: str,
    expected_sr: int = 44100,
    chunk_size: int = 2097152,
    extensions: Tuple[str, ...] = (".wav", ".flac", ".mp3"),
) -> Tuple[int, int]:
    """Generate and save embeddings for all audio files in a directory.
    
    Also computes and saves normalization statistics (mean/std) to latent_stats.pt.
    
    Args:
        audio_dir: Directory containing audio files.
        save_dir: Directory to save embedding .pt files.
        model: The stable audio model with pretransform encoder.
        params: Dictionary containing model parameters.
        device: Device to run encoding on.
        expected_sr: Expected sample rate of audio files.
        chunk_size: Number of samples per chunk.
        extensions: Tuple of valid audio file extensions.
        
    Returns:
        Tuple of (num_files, num_chunks) processed.
    """
    audio_dir = Path(audio_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_files = 0
    num_chunks = 0
    
    for file in audio_dir.iterdir():
        if file.suffix.lower() in extensions:
            wav, sr = torchaudio.load(str(file))
            assert sr == expected_sr, f"Expected {expected_sr} Hz, got {sr} Hz for file {file}"
            
            for chunk_idx in range(wav.shape[1] // chunk_size):
                chunk_audio = wav[:, chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                embedding = encode_audio(chunk_audio, sr, model, params, device)
                
                save_path = save_dir / f"{file.stem}_chunk_{chunk_idx}.pt"
                torch.save(embedding.detach().cpu(), save_path)
                num_chunks += 1
            
            num_files += 1
            print(f"Saved embedding for {file.stem} ({num_files} files embedded so far)")
    
    print(f"Embedded {num_files} files across {num_chunks} chunks.")
    
    # Compute and save normalization statistics
    mean, std = compute_latent_stats(save_dir)
    stats_path = save_dir / "latent_stats.pt"
    torch.save({"mean": mean, "std": std}, stats_path)
    print(f"Saved latent stats to {stats_path}")
    
    return num_files, num_chunks


class LatentAudioDataset(Dataset):
    """PyTorch Dataset for loading pre-computed audio embeddings.
    
    Args:
        root: Directory containing .pt embedding files.
        normalize: If True, normalize latents using stored mean/std.
                   Stats are loaded from latent_stats.pt or computed if missing.
        transform: Optional transform to apply to each sample.
        dim: If specified, slice the last dimension to this size and expand dataset.
             E.g., dim=64 on (1, 64, 1024) yields 16 samples of (1, 64, 64) per file.
    """
    
    def __init__(self, root: str | Path, normalize: bool = False, transform: Optional[Any] = None, dim: Optional[int] = None):
        self.root = Path(root)
        self.files = [f for f in os.listdir(root) if f.endswith(".pt") and f != "latent_stats.pt"]
        self.normalize = normalize
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.transform = transform
        self.dim = dim
        self.slices_per_file = 1
        
        # Compute number of slices per file based on first file's shape
        if self.dim is not None and len(self.files) > 0:
            sample = torch.load(self.root / self.files[0])
            last_dim = sample.shape[-1]
            self.slices_per_file = last_dim // self.dim
        
        if normalize:
            self._load_or_compute_stats()
    
    def _load_or_compute_stats(self):
        """Load stats from disk or compute and save them."""
        stats_path = self.root / "latent_stats.pt"
        if stats_path.exists():
            stats = torch.load(stats_path)
            self.mean = stats["mean"]
            self.std = stats["std"]
            print(f"Loaded latent stats: mean={self.mean:.4f}, std={self.std:.4f}")
        else:
            self.mean, self.std = compute_latent_stats(self.root)
            torch.save({"mean": self.mean, "std": self.std}, stats_path)
            print(f"Saved latent stats to {stats_path}")
    
    def __len__(self) -> int:
        return len(self.files) * self.slices_per_file
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx = idx // self.slices_per_file
        slice_idx = idx % self.slices_per_file
        
        sample = torch.load(self.root / self.files[file_idx])
        
        # Slice to specified dimension if provided
        if self.dim is not None:
            start = slice_idx * self.dim
            end = start + self.dim
            sample = sample[..., start:end]
        
        if self.normalize and self.mean is not None and self.std is not None:
            sample = normalize_latents(sample, self.mean, self.std)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    # Configuration
    SAMPLING_RATE = 44100
    CHUNK_SIZE = 2097152
    DATA_PATH = Path("C:/Users/dzluk/stable-audio-tools/data/blackbird")
    AUDIO_PATH = DATA_PATH / "audio"
    SAVE_PATH = DATA_PATH / "embeddings"
    
    # Load the model
    model, params = load_model()
    
    # Generate embeddings
    num_files, num_chunks = generate_embeddings(
        audio_dir=AUDIO_PATH,
        save_dir=SAVE_PATH,
        model=model,
        sample_size=params["sample_size"],
        device=params["device"],
        expected_sr=SAMPLING_RATE,
        chunk_size=CHUNK_SIZE,
    )
    
    print("Done!")

