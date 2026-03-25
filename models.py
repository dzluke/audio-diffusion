"""Model wrappers for audio diffusion training.

Each model in this file inherits from ``DiffusionModel`` and exposes a common
interface used by ``train.py``:
- ``data_transform(x)``: shape adaptation before noise is added
- ``forward(x, t)``: predict model output for noisy latents at timestep ``t``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from diffusers import DDIMPipeline, DDIMScheduler, DDPMScheduler, UNet2DModel
from stable_audio_tools.models.dit import DiffusionTransformer


def _save_scheduler_config(folder_path: Path, noise_scheduler: Any) -> None:
	folder_path.mkdir(parents=True, exist_ok=True)
	config_path = folder_path / "scheduler_config.json"
	with open(config_path, "w", encoding="utf-8") as f:
		json.dump(dict(noise_scheduler.config), f, indent=2)


def _load_scheduler_config(folder_path: Path) -> Any:
	config_path = folder_path / "scheduler_config.json"
	with open(config_path, "r", encoding="utf-8") as f:
		config = json.load(f)

	class_name = config.get("_class_name", "DDPMScheduler")
	if class_name == "DDIMScheduler":
		return DDIMScheduler.from_config(config)
	if class_name == "DDPMScheduler":
		return DDPMScheduler.from_config(config)
	raise ValueError(f"Unsupported scheduler type in config: {class_name}")


class DiffusionModel(nn.Module, ABC):
	def __init__(self) -> None:
		super().__init__()

	@abstractmethod
	def data_transform(self, x: torch.Tensor) -> torch.Tensor:
		"""Adapt dataset tensors to the shape expected by this model."""
		raise NotImplementedError()

	@abstractmethod
	def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		"""Predict diffusion target from noisy input and timestep(s)."""
		raise NotImplementedError()

	@abstractmethod
	def to_string(self) -> str:
		"""Human-readable model description."""
		raise NotImplementedError()

	@abstractmethod
	def save(self, folder_path: Path | str, noise_scheduler: Any) -> None:
		"""Save model weights and scheduler config into a folder."""
		raise NotImplementedError()

	@classmethod
	@abstractmethod
	def load(cls, folder_path: Path | str) -> tuple["DiffusionModel", Any]:
		"""Load model and scheduler from a previously saved folder."""
		raise NotImplementedError()


class DiffusersUNet2DModel(DiffusionModel):
	"""Wrapper around diffusers ``UNet2DModel`` for latent audio diffusion."""

	def __init__(self, config: Any) -> None:
		super().__init__()
		self.latent_shape = config.latent_shape
		self.backbone = UNet2DModel(
			sample_size=config.latent_shape,
			in_channels=1,
			out_channels=1,
			layers_per_block=config.layers_per_block,
			block_out_channels=config.block_out_channels,
			attention_head_dim=config.attention_head_dim,
			down_block_types=config.down_block_types,
			up_block_types=config.up_block_types,
			dropout=config.dropout,
			downsample_type=config.sample_type,
			upsample_type=config.sample_type,
		)

	@property
	def config(self) -> Any:
		return self.backbone.config

	def data_transform(self, x: torch.Tensor) -> torch.Tensor:
		return x

	def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		x_in = self.data_transform(x)
		return self.backbone(x_in, t, return_dict=False)[0]

	def to_string(self) -> str:
		return (
			f"Type:                {type(self.backbone).__name__}\n"
			f"Latent shape:        {self.latent_shape}\n"
			f"In channels:         {self.config.in_channels}\n"
			f"Out channels:        {self.config.out_channels}\n"
			f"Layers per block:    {self.config.layers_per_block}\n"
			f"Block out channels:  {self.config.block_out_channels}\n"
			f"Down block types:    {self.config.down_block_types}\n"
			f"Up block types:      {self.config.up_block_types}\n"
			f"Dropout:             {self.config.dropout}\n"
			f"Attention head dim:  {self.config.attention_head_dim}\n"
		)

	def save(self, folder_path: Path | str, noise_scheduler: Any) -> None:
		folder = Path(folder_path)
		pipeline = DDIMPipeline(unet=self.backbone, scheduler=noise_scheduler)
		pipeline.scheduler.config.clip_sample = False
		pipeline.save_pretrained(str(folder))
		_save_scheduler_config(folder, noise_scheduler)

	@classmethod
	def load(cls, folder_path: Path | str) -> tuple["DiffusersUNet2DModel", Any]:
		folder = Path(folder_path)
		pipeline = DDIMPipeline.from_pretrained(str(folder))
		return pipeline.unet, pipeline.scheduler


class SAODiTModel(DiffusionModel):
	"""Wrapper around stable-audio-tools ``DiffusionTransformer``."""

	def __init__(self, config: Any) -> None:
		super().__init__()
		self.latent_shape = config.latent_shape
		self.backbone = DiffusionTransformer(io_channels=config.latent_shape[0])

	def data_transform(self, x: torch.Tensor) -> torch.Tensor:
		# SAO-DiT consumes (B, C, T), while dataset yields (B, 1, C, T).
		if x.dim() == 4 and x.size(1) == 1:
			return x.squeeze(1)
		return x

	def _normalize_timestep(self, t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
		if t.dim() == 0:
			return t.expand(batch_size).to(device)
		return t.to(device)

	def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		had_channel_dim = x.dim() == 4 and x.size(1) == 1
		x_in = self.data_transform(x)
		t_in = self._normalize_timestep(t, x_in.size(0), x_in.device)
		out = self.backbone(x_in, t_in)
		return out.unsqueeze(1) if had_channel_dim else out

	def to_string(self) -> str:
		return (
			f"Type:                {type(self.backbone).__name__}\n"
			f"Latent shape:        {self.latent_shape}\n"
			f"IO channels:         {self.latent_shape[0]}\n"
		)

	def save(self, folder_path: Path | str, noise_scheduler: Any) -> None:
		folder = Path(folder_path)
		folder.mkdir(parents=True, exist_ok=True)

		model_blob = {
			"state_dict": self.backbone.state_dict(),
			"latent_shape": list(self.latent_shape),
		}
		torch.save(model_blob, folder / "model.safetensors.pt")

		with open(folder / "model_config.json", "w", encoding="utf-8") as f:
			json.dump({"model_name": "SAO-DiT", "latent_shape": list(self.latent_shape)}, f, indent=2)

		_save_scheduler_config(folder, noise_scheduler)

	@classmethod
	def load(cls, folder_path: Path | str) -> tuple["SAODiTModel", Any]:
		folder = Path(folder_path)
		with open(folder / "model_config.json", "r", encoding="utf-8") as f:
			model_config = json.load(f)

		latent_shape = tuple(model_config["latent_shape"])
		instance = cls(SimpleNamespace(latent_shape=latent_shape))

		blob = torch.load(folder / "model.safetensors.pt", map_location="cpu")
		instance.backbone.load_state_dict(blob["state_dict"])

		scheduler = _load_scheduler_config(folder)
		return instance, scheduler


def create_model(config: Any) -> DiffusionModel:
	"""Factory for creating a diffusion model from ``TrainingConfig``."""
	if config.model_name == "Diffusers-UNet2DModel":
		return DiffusersUNet2DModel(config)
	if config.model_name == "SAO-DiT":
		return SAODiTModel(config)
	raise ValueError(f"Unsupported model name: {config.model_name}")

	