"""
Docstring for train

based on code from https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb

"""


import torch
import torch.nn.functional as F
import torchaudio
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler, DDPMPipeline, DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np
from dataset import LatentAudioDataset, decode_audio, load_model as load_audio_model, denormalize_latents
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
import os
import time

SAMPLING_RATE = 44100
DATA_PATH = Path("C:/Users/dzluk/stable-audio-tools/data/blackbird")
AUDIO_PATH = DATA_PATH / "audio"
EMBEDDINGS_PATH = DATA_PATH / "embeddings"

BATCH_SIZE = 2
NUM_EPOCHS = 30
EMBEDDING_DIM = (64, 1024)  # this is the size of one audio embedding
CHUNK_SIZE = 2097152 # number of samples per chunk (e.g., 2097152 for ~47 seconds at 44.1kHz)


class TrainingConfig:
    image_size = (64, 1024)  # the generated image resolution
    train_batch_size = 4
    eval_batch_size = 1 # how many images to sample during evaluation
    num_epochs = 500
    gradient_accumulation_steps = 2
    learning_rate = 5e-5
    lr_warmup_steps = 500
    save_audio_epochs = 10  # how often to save generated audio samples during training (in epochs)
    save_model_epochs = 50  # how often to save the model during training (in epochs)
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'blackbird'  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resume_from_checkpoint = None  # None to start fresh, or Path to checkpoint dir, or "latest" to auto-find most recent

def evaluate(config, epoch, pipeline, audio_model, accelerator, global_step, dataset):
    """Sample latents from the diffusion model, decode to audio, and log to TensorBoard.
    
    Args:
        config: Training configuration.
        epoch: Current epoch number.
        pipeline: DDPMPipeline for sampling.
        audio_model: Stable audio model for decoding latents to audio.
        accelerator: Accelerator instance for TensorBoard logging.
        global_step: Current global training step.
        dataset: LatentAudioDataset instance (used to check if normalization was applied).
    """
    # Sample latents from random noise (backward diffusion process)
    # DDPMPipeline returns images, but for us these are latent tensors
    output = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        output_type="np",  # Get numpy arrays instead of PIL images
        num_inference_steps=50,  # DDIM allows fewer steps than DDPM
    )
    
    # The pipeline output is NHWC [B, H, W, C], convert to NCHW [B, C, H, W] -> [B, 1, 64, 1024]
    latents = torch.from_numpy(output.images).to(config.device)
    latents = latents.permute(0, 3, 1, 2)  # NHWC -> NCHW
    
    # Denormalize latents back to original scale before decoding (if training used normalization)
    if dataset.normalize and dataset.mean is not None and dataset.std is not None:
        latents = denormalize_latents(latents, dataset.mean, dataset.std)
    
    # Save directory for audio samples
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    
    # Get TensorBoard writer from accelerator
    tb_tracker = accelerator.get_tracker("tensorboard")
    writer = tb_tracker.writer
    
    # Decode each latent to audio and save
    for i in range(latents.shape[0]):
        # decode_audio expects [B, C, N] latent, we have [B, 1, 64, 1024]
        latent = latents[i:i+1]  # Keep batch dim: [1, 1, 64, 1024]
        latent = latent.squeeze(1)  # [1, 64, 1024]
        
        # Decode latent to audio
        decoded_float, audio_int16 = decode_audio(latent, audio_model)
        
        # Save as wav file
        audio_path = os.path.join(test_dir, f"epoch{epoch:04d}_sample{i:02d}.wav")
        torchaudio.save(audio_path, audio_int16.cpu(), SAMPLING_RATE)
        
        # Save latent tensor
        # latent_path = os.path.join(test_dir, f"epoch{epoch:04d}_sample{i:02d}_latent.pt")
        # torch.save(latent.cpu(), latent_path)
        
        # Log audio to TensorBoard
        # TensorBoard expects audio as 1D float tensor in range [-1, 1]
        audio_float = audio_int16[0].float() / 32767.0  # Take first channel, convert to float [-1, 1]
        writer.add_audio(
            f"samples/sample_{i:02d}",
            audio_float,
            global_step=global_step,
            sample_rate=SAMPLING_RATE,
        )
        
        # Log latent visualization as image (normalize for display)
        # latent_viz = latent[0].cpu().float()  # [64, 1024]
        # latent_viz = (latent_viz - latent_viz.min()) / (latent_viz.max() - latent_viz.min() + 1e-8)
        # writer.add_image(
        #     f"latents/sample_{i:02d}",
        #     latent_viz.unsqueeze(0),  # Add channel dim: [1, 64, 1024]
        #     global_step=global_step,
        # )
    
    # Log epoch summary
    writer.add_scalar("epoch", epoch, global_step)


def main():
    config = TrainingConfig()
    
    # Load the stable audio model for decoding latents back to audio during evaluation
    print("Loading stable audio model for decoding...")
    audio_model, audio_params = load_audio_model(device=config.device)

    
    # create dataloader for training (with normalization enabled)
    dataset = LatentAudioDataset(EMBEDDINGS_PATH, normalize=True)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    # Create a model
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 256, 512),  # the number of output channes for each UNet block
        down_block_types=( 
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        ), 
        up_block_types=(
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",  # a regular ResNet upsampling block
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    model.to(config.device)

    # define noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # Initialize accelerator and tensorboard logging
    logging_dir = os.path.join(config.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            pass
            # repo_id = create_repo(
            #     repo_id=Path(config.output_dir).name, exist_ok=True
            # ).repo_id
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        
        # Determine next run number for TensorBoard logs
        os.makedirs(logging_dir, exist_ok=True)
        existing_runs = [d for d in os.listdir(logging_dir) if d.startswith("run") and d[3:].isdigit()]
        if existing_runs:
            next_run = max(int(d[3:]) for d in existing_runs) + 1
        else:
            next_run = 0
        run_name = f"run{next_run}"
        accelerator.init_trackers(run_name)
        print("Initialized TensorBoard tracking with run name:", run_name)
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Resume from checkpoint if specified
    global_step = 0
    starting_epoch = 0
    checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint == "latest":
            # Find the most recent checkpoint
            if os.path.exists(checkpoint_dir):
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    resume_path = os.path.join(checkpoint_dir, checkpoints[-1])
                    print(f"Resuming from latest checkpoint: {resume_path}")
                else:
                    resume_path = None
                    print("No checkpoints found, starting from scratch")
            else:
                resume_path = None
                print("No checkpoint directory found, starting from scratch")
        else:
            resume_path = config.resume_from_checkpoint
        
        if resume_path and os.path.exists(resume_path):
            accelerator.load_state(resume_path)
            # Extract step from checkpoint name (checkpoint-{global_step})
            global_step = int(os.path.basename(resume_path).split("-")[1])
            starting_epoch = global_step // len(train_dataloader)
            print(f"Resumed training from step {global_step}, epoch {starting_epoch}")

    # Now you train the model
    for epoch in range(starting_epoch, config.num_epochs):
        epoch_start_time = time.time()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        epoch_losses = []  # Track losses for epoch average

        for step, batch in enumerate(train_dataloader):
            clean_images = batch.to(config.device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            epoch_losses.append(loss.detach().item())
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        # Log epoch-level metrics
        if accelerator.is_main_process:
            epoch_duration = time.time() - epoch_start_time
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            tb_tracker = accelerator.get_tracker("tensorboard")
            writer = tb_tracker.writer
            writer.add_scalar("epoch/avg_loss", avg_loss, epoch)
            writer.add_scalar("epoch/learning_rate", lr_scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("epoch/duration_seconds", epoch_duration, epoch)
            print(f"Epoch {epoch} - Average Loss: {avg_loss:.6f} - Duration: {epoch_duration:.1f}s")

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            # Use DDIM for faster sampling (50 steps vs 1000 for DDPM)
            ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
            pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=ddim_scheduler)

            if (epoch + 1) % config.save_audio_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline, audio_model, accelerator, global_step, dataset)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    pass
                    # upload_folder(
                    #     repo_id=repo_id,
                    #     folder_path=config.output_dir,
                    #     commit_message=f"Epoch {epoch}",
                    #     ignore_patterns=["step_*", "epoch_*"],
                    # )
                else:
                    pipeline.save_pretrained(config.output_dir)
                
                # Save full checkpoint for resumption
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                print(f"Saved checkpoint to {save_path}") 

    accelerator.end_training()
    print("Training complete!")


if __name__ == "__main__":
    main()
