"""
Train a diffusion model on the blackbird dataset using the Hugging Face Diffusers library.

Run on CNMATGPU with:

"""

import numpy as np
import time
import torch
import torch.nn.functional as F
import torchaudio
from matplotlib import pyplot as plt
from PIL import Image
import torchvision
from pathlib import Path
from torchvision import transforms
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler, UNet2DModel, DDIMPipeline
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from dataset import LatentAudioDataset, decode_audio, load_model as load_audio_codec
from evaluate import compute_validation_loss, compute_fad, get_reference_embeddings

EMBEDDINGS_PATH = Path("C:/Users/dzluk/stable-audio-tools/data/blackbird/embeddings")
SAMPLING_RATE = 44100

class TrainingConfig:
    latent_shape = (64, 64)
    train_batch_size = 16
    eval_batch_size = 1 # how many audios to sample during evaluation
    num_epochs = 250
    learning_rate = 4e-4
    save_audio_epochs = 50  # how often to sample during training (in epochs)
    save_model_epochs = 50  # how often to save the model during training (in epochs)
    eval_every_epochs = 5  # how often to compute evaluation loss
    val_split = 0.1  # fraction of data to use for validation
    output_dir = Path("diffusers-training-runs")
    
    # FAD evaluation settings
    compute_fad = True  # whether to compute FAD during evaluation
    fad_num_samples = 10  # number of samples to generate for FAD computation

    # EMA settings
    use_ema = True  # whether to use exponential moving average
    ema_decay = 0.9999  # EMA decay rate (higher = slower update)
    ema_update_after_step = 0  # start EMA updates after this many steps
    ema_power = 0.75  # power for EMA warmup

    # Learning rate scheduling
    use_lr_scheduler = True  # whether to use learning rate scheduling
    lr_warmup_steps = 500  # number of warmup steps for learning rate scheduler

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
    timestamp = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d_%H%M%S")  # +9h for Paris time
    run_dir = parent_dir / f"run_{timestamp}"
    logs_dir = run_dir / "logs"
    samples_dir = run_dir / "samples"
    pipeline_dir = run_dir / "pipeline"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir, logs_dir, samples_dir, pipeline_dir


def save_run_info(run_dir, config, model, noise_scheduler, train_size, val_size):
    """Save hyperparameters and model architecture to a human-readable text file."""
    info_path = run_dir / "run_info.txt"
    
    with open(info_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("TRAINING RUN CONFIGURATION\n")
        f.write(f"Started: {(datetime.now() + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S')} (Paris)\n")
        f.write("="*60 + "\n\n")
        
        # Hyperparameters
        f.write("HYPERPARAMETERS\n")
        f.write("-"*40 + "\n")
        f.write(f"Latent shape:        {config.latent_shape}\n")
        f.write(f"Train batch size:    {config.train_batch_size}\n")
        f.write(f"Eval batch size:     {config.eval_batch_size}\n")
        f.write(f"Number of epochs:    {config.num_epochs}\n")
        f.write(f"Learning rate:       {config.learning_rate}\n")
        # f.write(f"Save audio epochs:   {config.save_audio_epochs}\n")
        # f.write(f"Save model epochs:   {config.save_model_epochs}\n")
        # f.write(f"Eval every epochs:   {config.eval_every_epochs}\n")
        # f.write(f"Validation split:    {config.val_split}\n")
        # f.write(f"Seed:                {config.seed}\n")
        # f.write(f"Device:              {config.device}\n")
        f.write("\n")
        
        # Dataset info
        f.write("DATASET\n")
        f.write("-"*40 + "\n")
        f.write(f"Embeddings path:     {EMBEDDINGS_PATH}\n")
        f.write(f"Train size:          {train_size}\n")
        f.write(f"Validation size:     {val_size}\n")
        f.write("\n")
        
        # Noise scheduler
        f.write("NOISE SCHEDULER\n")
        f.write("-"*40 + "\n")
        f.write(f"Type:                {type(noise_scheduler).__name__}\n")
        f.write(f"Train timesteps:     {noise_scheduler.config.num_train_timesteps}\n")
        f.write(f"Beta schedule:       {noise_scheduler.config.beta_schedule}\n")
        f.write(f"Clip sample:         {noise_scheduler.config.clip_sample}\n")
        f.write("\n")
        
        # Model architecture
        f.write("MODEL ARCHITECTURE\n")
        f.write("-"*40 + "\n")
        f.write(f"Type:                {type(model).__name__}\n")
        f.write(f"Sample size:         {model.config.sample_size}\n")
        f.write(f"In channels:         {model.config.in_channels}\n")
        f.write(f"Out channels:        {model.config.out_channels}\n")
        f.write(f"Layers per block:    {model.config.layers_per_block}\n")
        f.write(f"Block out channels:  {model.config.block_out_channels}\n")
        f.write(f"Down block types:    {model.config.down_block_types}\n")
        f.write(f"Up block types:      {model.config.up_block_types}\n")
        f.write("\n")
        
        # Model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total parameters:    {total_params:,}\n")
        f.write(f"Trainable params:    {trainable_params:,}\n")
        f.write("\n")
        
        # FAD evaluation settings
        f.write("EVALUATION METRICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Compute FAD:         {config.compute_fad}\n")
        if config.compute_fad:
            f.write(f"FAD num samples:     {config.fad_num_samples}\n")
        f.write("\n")
        
        f.write("="*60 + "\n")
    
    print(f"Run info saved to {info_path}")


def generate_and_log_samples(noise_input, model, noise_scheduler, writer, samples_dir, epoch,
                             config=None, reference_embeddings=None, ema_model=None):
    """Generate audio samples and log them to TensorBoard.
    
    Args:
        noise_input: Random noise tensor to start sampling from
        model: The diffusion model
        noise_scheduler: The noise scheduler
        writer: TensorBoard SummaryWriter
        samples_dir: Directory to save audio samples
        epoch: Current epoch number
        config: TrainingConfig for FAD settings (optional)
        reference_embeddings: Reference latent embeddings for FAD computation (optional)
        ema_model: EMA model wrapper (optional). If provided, uses EMA weights for sampling.
    
    Returns:
        FAD score if computed, else None
    """
    model.eval()
    
    # Use EMA weights for sampling if available
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
    num_sampling_steps = 50

    samples = sample(noise_input, model, noise_scheduler, num_sampling_steps)
    
    # Compute FAD on latent embeddings if enabled
    fad_score = None
    if config is not None and config.compute_fad and reference_embeddings is not None:
        fad_score = compute_fad(samples, reference_embeddings)
        if fad_score is not None:
            writer.add_scalar("metrics/fad", fad_score, epoch)
            print(f"Epoch {epoch} FAD score: {fad_score:.4f}")
    
    # Load audio codec only when needed
    audio_codec, _ = load_audio_codec()
    
    # Create epoch-specific directory for audio samples
    epoch_samples_dir = samples_dir / f"epoch_{epoch}"
    epoch_samples_dir.mkdir(parents=True, exist_ok=True)
    
    for i, s in enumerate(samples):
        audio = decode_audio(s, audio_codec)
        
        # Save audio file (expects shape [channels, samples])
        audio_path = epoch_samples_dir / f"sample_{i}.wav"
        torchaudio.save(str(audio_path), audio.cpu(), SAMPLING_RATE)
        
        if config is not None and i < config.eval_batch_size:
            # Log to TensorBoard (expects 1D float array in [-1, 1])
            audio_mono = (audio[0] + audio[1]) / 2  # Convert to mono by averaging channels
            audio_np = audio_mono.cpu().numpy()
            writer.add_audio(f"samples/sample_{i}", audio_np, epoch, sample_rate=SAMPLING_RATE)
    
    # Unload audio codec to free GPU memory
    del audio_codec
    torch.cuda.empty_cache()
    
    # Restore original model weights if EMA was used
    if ema_model is not None:
        ema_model.restore(model.parameters())
    
    model.train()
    return fad_score


def train():
    config = TrainingConfig()

    # Create run directory structure
    run_dir, logs_dir, samples_dir, pipeline_dir = create_run_dir(config.output_dir)
    print(f"Run directory: {run_dir}")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(logs_dir))

    dataset = LatentAudioDataset(EMBEDDINGS_PATH, normalize=False, dim=config.latent_shape[1])

    # Split dataset into train and validation sets
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed)
    )
    print(f"Train size: {train_size}, Val size: {val_size}")

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.train_batch_size, shuffle=False
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

    # Initialize EMA model
    ema_model = None
    if config.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=config.ema_decay,
            update_after_step=config.ema_update_after_step,
            model_cls=UNet2DModel,
            model_config=model.config,
        )
        ema_model.to(config.device)
        print(f"EMA model initialized with decay={config.ema_decay}")

    # Save run info
    save_run_info(run_dir, config, model, noise_scheduler, train_size, val_size)

    # Prepare reference embeddings for FAD computation
    reference_embeddings = None
    if config.compute_fad:
        reference_embeddings = get_reference_embeddings(val_dataset, num_samples=config.fad_num_samples)

    # create some random noise which will be used to generate sample audio during training
    # Use fad_num_samples when FAD is enabled, otherwise use eval_batch_size
    num_samples = config.fad_num_samples if config.compute_fad else config.eval_batch_size
    sample_noise = torch.randn((num_samples, 1, *config.latent_shape), device=config.device)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Set up learning rate scheduler
    lr_scheduler = None
    if config.use_lr_scheduler:
        num_training_steps = len(train_dataloader) * config.num_epochs
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(f"LR scheduler initialized with {config.lr_warmup_steps} warmup steps, {num_training_steps} total steps")

    losses = []
    global_step = 0
    
    # Timing tracking
    training_start_time = time.time()

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

            # Step the learning rate scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Update EMA weights
            if ema_model is not None:
                ema_model.step(model.parameters())

        # Calculate and log epoch loss
        loss_last_epoch = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
        writer.add_scalar("loss/epoch", loss_last_epoch, epoch + 1)
        
        # Log learning rate
        if lr_scheduler is not None:
            current_lr = lr_scheduler.get_last_lr()[0]
            writer.add_scalar("lr", current_lr, epoch)

        if (epoch) % 5 == 0:
            elapsed_time = time.time() - training_start_time
            iterations_per_sec = global_step / elapsed_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = config.num_epochs - (epoch + 1)
            estimated_remaining = remaining_epochs * avg_time_per_epoch
            
            # Format remaining time
            remaining_mins, remaining_secs = divmod(int(estimated_remaining), 60)
            remaining_hours, remaining_mins = divmod(remaining_mins, 60)
            
            print(f"Epoch:{epoch}, loss: {loss_last_epoch:.6f} | "
                  f"{iterations_per_sec:.2f} it/s | "
                  f"avg: {avg_time_per_epoch:.2f}s/epoch | "
                  f"ETA: {remaining_hours}h {remaining_mins}m {remaining_secs}s")

        # Compute and log evaluation loss
        if (epoch + 1) % config.eval_every_epochs == 0:
            val_loss = compute_validation_loss(model, val_dataloader, noise_scheduler, config.device)
            writer.add_scalar("loss/validation", val_loss, epoch + 1)
            print(f"Epoch:{epoch+1}, val_loss: {val_loss}")

        # Generate and log audio samples
        if (epoch) % config.save_audio_epochs == 0:
            print(f"Generating audio samples at epoch {epoch}...")
            generate_and_log_samples(
                sample_noise, model, noise_scheduler, writer, samples_dir, epoch,
                config=config, reference_embeddings=reference_embeddings, ema_model=ema_model
            )

    print("Finished training!")

    generate_and_log_samples(
        sample_noise, model, noise_scheduler, writer, samples_dir, epoch,
        config=config, reference_embeddings=reference_embeddings, ema_model=ema_model
    )

    # Copy EMA weights to model for saving
    if ema_model is not None:
        ema_model.copy_to(model.parameters())
        print("Copied EMA weights to model for saving")

    trained_pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)
    trained_pipeline.scheduler.config.clip_sample = False
    print(f"Saving trained pipeline to {pipeline_dir}")
    trained_pipeline.save_pretrained(str(pipeline_dir))

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    train()
