"""
Train a diffusion model on the blackbird dataset using the Hugging Face Diffusers library.
"""

import numpy as np
import time
import torch
import torch.nn.functional as F
import soundfile as sf
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from dataset import LatentAudioDataset, decode_audio, load_model as load_audio_codec
from evaluate import compute_validation_loss
from models import create_model, DiffusersUNet2DModel


EMBEDDINGS_PATH = Path("C:/Users/dzluk/stable-audio-tools/data/blackbird/embeddings")
SAMPLING_RATE = 44100

class TrainingConfig:
    # model_name can be "Diffusers-UNet2DModel" or "SAO-DiT"
    model_name = "Diffusers-UNet2DModel" 
    # model_name = "SAO-DiT"

    latent_shape = (64, 256)
    train_batch_size = 16
    eval_batch_size = 3 # how many audios to sample during evaluation
    num_epochs = 300
    learning_rate = 4e-4
    save_audio_epochs = 50  # how often to sample during training (in epochs)
    # save_model_epochs = 50  # how often to save the model during training (in epochs)
    eval_every_epochs = 10  # how often to compute evaluation loss
    
    val_split = 0.1  # fraction of data to use for validation
    output_dir = Path("logs")

    # model architecture settings
    prediction_type = "v_prediction"  # "epsilon" or "v_prediction"
    # block_out_channels = (64, 128, 128, 256)  # used up until 3/5/26
    block_out_channels = (128, 256, 256, 512)
    down_block_types = (
            "DownBlock2D",
            "DownBlock2D", # you can try adding attn here but it will ~5x training time
            "AttnDownBlock2D",
            "AttnDownBlock2D",
    )
    up_block_types = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",  
            "UpBlock2D",
            "UpBlock2D", 
    )
    layers_per_block = 2 # default is 2, can try 3 but will almost double training time
    dropout = 0.1 # default is 0
    attention_head_dim = 8 # default is 8, try 32 (the number of attention heads = num channels / attention_head_dim)
    sample_type = "conv" # can be "conv" or "resnet"
    
    # EMA settings
    use_ema = True  # whether to use exponential moving average
    ema_decay = 0.9999  # EMA decay rate (higher = slower update)
    ema_update_after_step = 0  # start EMA updates after this many steps
    ema_power = 0.75  # power for EMA warmup

    # Learning rate scheduling
    use_lr_scheduler = False  # whether to use learning rate scheduling
    lr_warmup_steps = 500  # number of warmup steps for learning rate scheduler

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"


def sample(x, model, scheduler, num_sampling_steps):
        """
        Given a noise sample x, iteratively denoise it num_sampling_steps times using the model and scheduler
        x has shape (b, 1, 64, time_dim)
        """
        scheduler.set_timesteps(num_sampling_steps)

        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = model(x, t)

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
        f.write(f"Prediction type:     {noise_scheduler.config.prediction_type}\n")
        f.write("\n")
        
        # Model architecture
        f.write("MODEL ARCHITECTURE\n")
        f.write("-"*40 + "\n")
        f.write(model.to_string())
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
        f.write("None!")
        f.write("\n")
        
        f.write("="*60 + "\n")
    
    print(f"Run info saved to {info_path}")


def _format_duration(seconds):
    """Format seconds as Hh Mm Ss string."""
    total = int(seconds)
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m {secs}s"


def append_timing_info(run_dir, total_training_seconds, avg_epoch_seconds):
    """Append timing statistics to run_info.txt at the end of training."""
    info_path = run_dir / "run_info.txt"
    with open(info_path, "a") as f:
        f.write("TIMING SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Total training time:       {_format_duration(total_training_seconds)} ({total_training_seconds:.2f}s)\n")
        f.write(f"Average epoch time:        {_format_duration(avg_epoch_seconds)} ({avg_epoch_seconds:.2f}s)\n")
        f.write("\n")


def generate_and_log_samples(noise_input, model, noise_scheduler, writer, samples_dir, epoch,
                             config, ema_model=None):
    """Generate audio samples and log them to TensorBoard.
    
    Args:
        noise_input: Random noise tensor to start sampling from
        model: The diffusion model
        noise_scheduler: The noise scheduler
        writer: TensorBoard SummaryWriter
        samples_dir: Directory to save audio samples
        epoch: Current epoch number
        config: TrainingConfig (optional)
        ema_model: EMA model wrapper (optional). If provided, uses EMA weights for sampling.
    """
    model.eval()
    
    # Use EMA weights for sampling if available
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
    num_sampling_steps = 50

    samples = sample(noise_input, model, noise_scheduler, num_sampling_steps)
    
    # Load audio codec only when needed
    audio_codec, _ = load_audio_codec()
    
    # Create epoch-specific directory for audio samples
    epoch_samples_dir = samples_dir / f"epoch_{epoch}"
    epoch_samples_dir.mkdir(parents=True, exist_ok=True)
    
    for i, s in enumerate(samples):
        audio = decode_audio(s, audio_codec)
        
        # Save audio file (expects shape [channels, samples])
        audio_path = epoch_samples_dir / f"sample_{i}.wav"
        sf.write(str(audio_path), audio.cpu().numpy().T, SAMPLING_RATE)
        
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


def train(config: TrainingConfig):
    run_start_time = time.time()

    if config is None:
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
    model = create_model(config)
    model.to(config.device)

    # Set the noise scheduler
    noise_scheduler = DDPMScheduler(
        beta_schedule="squaredcos_cap_v2", 
        clip_sample=False, 
        prediction_type=config.prediction_type
    )
    noise_scheduler.config.num_train_timesteps = 1000
    assert noise_scheduler.config.clip_sample is False

    # Initialize EMA model
    ema_model = None
    if config.use_ema and isinstance(model, DiffusersUNet2DModel):  # EMA currently used for UNet2DModel
        ema_model = EMAModel(
            model.parameters(),
            decay=config.ema_decay,
            update_after_step=config.ema_update_after_step,
            model_cls=UNet2DModel,
            model_config=model.config,
        )
        ema_model.to(config.device)

    # Save run info
    save_run_info(run_dir, config, model, noise_scheduler, train_size, val_size)

    # create some random noise which will be used to generate sample audio during training
    sample_noise = torch.randn((config.eval_batch_size, 1, *config.latent_shape), generator=torch.Generator(device=config.device).manual_seed(config.seed), device=config.device)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Set up learning rate scheduler
    # TODO: it's not clear to me if this actually affects the learning rate
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
    epoch_times = []
    
    # Timing tracking
    training_start_time = time.time()

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            total=len(train_dataloader),
            leave=False,
            disable=True
        )
        for step, batch in enumerate(progress_bar):
            clean_images = batch.to(config.device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction
            model_pred = model(noisy_images, timesteps)

            if config.prediction_type == "epsilon":
                target = noise
            elif config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
            else:
                raise ValueError(f"Unsupported prediction type: {config.prediction_type}")

            # Calculate the loss
            loss = F.mse_loss(model_pred, target)
            loss.backward()
            losses.append(loss.item())

            # Log step loss to TensorBoard
            writer.add_scalar("loss/step", loss.item(), global_step)
            global_step += 1
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

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
        epoch_time = time.time() - epoch_start_time
        loss_last_epoch = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
        writer.add_scalar("loss/epoch", loss_last_epoch, epoch + 1)
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. Loss: {loss_last_epoch:.6f}")
        # Log learning rate
        if lr_scheduler is not None:
            current_lr = lr_scheduler.get_last_lr()[0]
            writer.add_scalar("lr", current_lr, epoch)

        if (epoch + 1) % 5 == 0:
            elapsed_time = time.time() - training_start_time
            iterations_per_sec = global_step / elapsed_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = config.num_epochs - (epoch + 1)
            estimated_remaining = remaining_epochs * avg_time_per_epoch
            eta = datetime.now() + timedelta(hours=9, seconds=estimated_remaining)
            
            # Format remaining time
            remaining_mins, remaining_secs = divmod(int(estimated_remaining), 60)
            remaining_hours, remaining_mins = divmod(remaining_mins, 60)
            
            print(f"Epoch:{epoch + 1}, loss: {loss_last_epoch:.6f} | "
                  f"{iterations_per_sec:.2f} it/s | "
                  f"avg: {avg_time_per_epoch:.2f}s/epoch | "
                f"Remaining: {remaining_hours}h {remaining_mins}m {remaining_secs}s "
                f"(ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')})")

        # Compute and log evaluation loss
        if (epoch + 1) % config.eval_every_epochs == 0:
            val_loss = compute_validation_loss(
                model,
                val_dataloader,
                noise_scheduler,
                config.device,
                prediction_type=config.prediction_type,
            )
            writer.add_scalar("loss/validation", val_loss, epoch + 1)
            print(f"Epoch:{epoch+1}, val_loss: {val_loss}")

        # Generate and log audio samples
        if (epoch + 1) % config.save_audio_epochs == 0:
            print(f"Generating audio samples at epoch {epoch + 1}...")
            generate_and_log_samples(
                sample_noise, model, noise_scheduler, writer, samples_dir, epoch + 1,
                config=config, ema_model=ema_model
            )

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

    print("Finished training!")

    if not (epoch + 1) % config.save_audio_epochs == 0:
        print(f"Generating final audio samples...")
        generate_and_log_samples(
            sample_noise, model, noise_scheduler, writer, samples_dir, epoch + 1,
            config=config, ema_model=ema_model
        )

    # Copy EMA weights to model for saving
    if ema_model is not None:
        ema_model.copy_to(model.parameters())

    print(f"Saving trained model to {pipeline_dir}")
    model.save(pipeline_dir, noise_scheduler)

    total_training_time = time.time() - run_start_time
    avg_epoch_time = float(np.mean(epoch_times)) if len(epoch_times) > 0 else 0.0
    print(f"Total training time: {_format_duration(total_training_time)}")

    append_timing_info(
        run_dir=run_dir,
        total_training_seconds=total_training_time,
        avg_epoch_seconds=avg_epoch_time,
    )

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    config = TrainingConfig()
    train(config)
