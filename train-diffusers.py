"""
Train a diffusion model on the blackbird dataset using the Hugging Face Diffusers library.

Run on CNMATGPU with:

Start-Process powershell -WorkingDirectory "C:\Users\dzluk\Syrinx\audio-diffusion-course" -RedirectStandardOutput "stdout.txt" -RedirectStandardError "stderr.txt" -ArgumentList '-NoProfile', '-Command', 'uv run train-diffusers.py'


"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from matplotlib import pyplot as plt
from PIL import Image
import torchvision
from pathlib import Path
from torchvision import transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler, UNet2DModel, DDIMPipeline
from dataset import LatentAudioDataset, decode_audio, load_model as load_audio_codec
from IPython.display import Audio

EMBEDDINGS_PATH = Path("C:/Users/dzluk/stable-audio-tools/data/blackbird/embeddings")
SAMPLING_RATE = 44100

class TrainingConfig:
    latent_shape = (64, 64)
    train_batch_size = 4
    eval_batch_size = 1 # how many audios to sample during evaluation
    num_epochs = 500
    learning_rate = 4e-4
    save_audio_epochs = 50  # how often to sample during training (in epochs)
    save_model_epochs = 50  # how often to save the model during training (in epochs)
    output_dir = "diffusers-training-runs"

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"


def sample(x, model, scheduler, num_sampling_steps):
        """
        Given a noise sample x, iteratively denoise it num_sampling_steps times using the model and scheduler
        x has shape (b, 1, 64, 64)
        """
        scheduler.set_timesteps(num_sampling_steps)

        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = model(x, t).sample

            x = scheduler.step(
                noise_pred, t, x
            ).prev_sample

        return x


def create_run_dir(parent_dir):
    """Create a timestamped run directory with subdirectories for logs, samples, and pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = parent_dir / f"run_{timestamp}"
    logs_dir = run_dir / "logs"
    samples_dir = run_dir / "samples"
    pipeline_dir = run_dir / "pipeline"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir, logs_dir, samples_dir, pipeline_dir


def generate_and_log_samples(noise_input, model, noise_scheduler,writer, samples_dir, epoch):
    """Generate audio samples and log them to TensorBoard."""
    model.eval()
    num_sampling_steps = 50

    samples = sample(noise_input, model, noise_scheduler, num_sampling_steps)
    
    # Load audio codec only when needed
    audio_codec, _ = load_audio_codec()
    
    for i, s in enumerate(samples):
        audio = decode_audio(s, audio_codec)
        
        # Save audio file (expects shape [channels, samples])
        audio_path = samples_dir / f"epoch_{epoch}_sample_{i}.wav"
        torchaudio.save(str(audio_path), audio.cpu(), SAMPLING_RATE)
        
        # Log to TensorBoard (expects 1D float array in [-1, 1])
        # audio is (2, T) stereo int16, take first channel and convert to float
        audio_mono = (audio[0] + audio[1]) / 2  # Convert to mono by averaging channels
        # audio_np = audio.cpu().numpy().astype(np.float32)
        audio_np = audio_mono.cpu().numpy()
        writer.add_audio(f"samples/sample_{i}", audio_np, epoch, sample_rate=SAMPLING_RATE)
    
    # Unload audio codec to free GPU memory
    del audio_codec
    torch.cuda.empty_cache()
    
    model.train()


def train():
    config = TrainingConfig()

    # Create run directory structure
    run_dir, logs_dir, samples_dir, pipeline_dir = create_run_dir(config.output_dir)
    print(f"Run directory: {run_dir}")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(logs_dir))

    dataset = LatentAudioDataset(EMBEDDINGS_PATH, normalize=False, dim=config.latent_shape[1])

    # Create a dataloader from the dataset to serve up the transformed images in batches
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True
    )

    # Create a model
    model = UNet2DModel(
        sample_size=config.latent_shape[0],  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    )
    model.to(config.device)

    # Set the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=False
    )
    assert noise_scheduler.config.clip_sample is False

    # create some random noise which will be used to generate sample audio during training
    sample_noise = torch.randn((config.eval_batch_size, 1, *config.latent_shape), device=config.device)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses = []
    global_step = 0

    for epoch in range(config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            clean_images = batch.to(config.device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate the loss
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)
            losses.append(loss.item())

            # Log step loss to TensorBoard
            writer.add_scalar("loss/step", loss.item(), global_step)
            global_step += 1

            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()

        # Calculate and log epoch loss
        loss_last_epoch = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
        writer.add_scalar("loss/epoch", loss_last_epoch, epoch + 1)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

        # Generate and log audio samples
        if (epoch + 1) % config.save_audio_epochs == 0:
            print(f"Generating audio samples at epoch {epoch}...")
            generate_and_log_samples(sample_noise, model, noise_scheduler, writer, samples_dir, epoch)

    print("Finished training!")

    generate_and_log_samples(sample_noise, model, noise_scheduler, writer, samples_dir, epoch)

    trained_pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)
    trained_pipeline.scheduler.config.clip_sample = False
    print(f"Saving trained pipeline to {pipeline_dir}")
    trained_pipeline.save_pretrained(str(pipeline_dir))

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    train()
