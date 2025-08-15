from collections import defaultdict
import pufferlib
import pufferlib.vector
import pufferlib.models
from pufferlib.pufferl import PuffeRL
from src.environment.quantum_env import QuantumPrepEnv
import torch
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard logging

## Load the environment and add meta noise
def env_creator(**kwargs):
    env = QuantumPrepEnv(
        meta_noise=False,
    )
    return pufferlib.emulation.GymnasiumPufferEnv(env, **kwargs)

## Define the training configuration
# For testing (small values for quick debug iterations)
args = pufferlib.pufferl.load_config('default')
# Base/Env Setup (unchanged from your suggestion, but confirm env registration)
args['train']['env'] = 'quantum_prep'  # Matches your Gym env ID
args['train']['use_rnn'] = True  # Enable RNN for sequences (quantum prep benefits from memory)
args['train']['seed'] = 43  # For reproducibility
# Batch and Horizon (Increase for better stats/gradients; your bptt=128 is good, but scale batch up)
args['train']['batch_size'] = 512  # Double your 8192; larger batches = longer GPU kernels, better util. Auto-adjusts with bptt.
args['train']['bptt_horizon'] = 'auto'  # Keep—long enough for quantum sequences (e.g., 5-10 gates to Bell state) without OOM.
args['train']['minibatch_size'] = 128  # Double your 256; processes more data per gradient step for stability.
args['train']['max_minibatch_size'] = 256  # Double your 1024; cap to avoid VRAM spikes on RTX 4080.
# Training Hyperparams (Core tweaks here for better learning)
args['train']['total_timesteps'] = 500000  # 5x your 100k; should take ~30-60 min on GPU. Scale to 1e6+ once stable.
args['train']['learning_rate'] = 2e-4  # MUCH lower than your 0.015 (1.5e-2 is aggressive and can cause NaNs or no updates). Standard PPO start; decay via scheduler if needed.
args['train']['update_epochs'] = 8  # Double your 4; more passes over data = extended GPU compute without extra env steps.
args['train']['gamma'] = 0.99  # Discount—high for long-term fidelity rewards.
args['train']['gae_lambda'] = 0.95  # For advantages; helps with variance in noisy quantum envs.
args['train']['clip_coef'] = 0.15  # Standard PPO clip; encourages proximal updates to fix your KL=0.
args['train']['ent_coef'] = 0.01  # Start higher than default (0.0) to boost exploration, then decay to 0.001 over epochs for exploitation.  # <-- This is initial; will decay in loop
args['train']['vf_coef'] = 0.5  # Balances value loss; up if explained_variance stays low.
# Vectorization (Scale up for GPU util; your num_envs=32 is ok, but push higher)
args['vec']['backend'] = 'Multiprocessing'  # Or 'Multiprocessing' if CPU workers help; test for your high CPU issue (more below).
args['vec']['num_envs'] = 16  # 4x your 32; more parallel sims = faster data collection, better GPU fill. QuantumPrepEnv is lightweight (2 qubits), so this fits.
args['vec']['num_workers'] = 8  # Double your 8; spreads CPU load (QuTiP sims are CPU-bound by default).
args['vec']['batch_size'] = 2  # 8x your 4; aligns with num_envs for efficient vec rollouts (masks dones automatically).
# Logging/Other (For monitoring)
#args['wandb'] = True  # Or Neptune; logs fidelity/SPS to dashboard for curves.
#args['wandb_project'] = 'entangleRL'  # Your project name.

## Vectorize the environment
vecenv = pufferlib.vector.make(
    env_creator,
    backend='Multiprocessing',
    num_workers=args['vec']['num_workers'],
    num_envs=args['vec']['num_envs'],
    batch_size=args['vec']['batch_size'],
)

## Define the policy with LSTM wrapper
base_policy = pufferlib.models.Default(
    env=vecenv,
    hidden_size=128,  # Increased from 64 for better encoding before LSTM
).cuda()
policy = pufferlib.models.LSTMWrapper(
    env=vecenv,
    policy=base_policy,
    input_size=base_policy.hidden_size,  # Matches encoder output
    hidden_size=128,  # Larger for LSTM capacity (adjust down if OOM)
).cuda()

## Set up the trainer
trainer = PuffeRL(args['train'], vecenv, policy)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='logs/tensorboard/quantum_prep' + time.strftime("%Y%m%d_%H%M%S"))  # Create a log dir; run tensorboard --logdir=logs/tensorboard to view

# Extract total_timesteps for decay calculation (outside loop for efficiency)
total_timesteps = args['train']['total_timesteps']

# Main training loop: Alternate evaluate (collect data) and train (update policy)
while trainer.global_step < total_timesteps:  # Use variable for brevity
    trainer.evaluate()  # Fill buffers with rollouts; this increments global_step

    # Update ent_coef with linear decay (after evaluate, before train)
    progress_ratio = trainer.global_step / total_timesteps
    trainer.config['ent_coef'] = max(0.001, 0.01 * (1 - progress_ratio))  # Clamp to >=0

    stats_logs = trainer.mean_and_log()  # Compute and log stats means (fidelity, steps) after evaluate
    trainer.train()  # Compute advantages/losses and update; no return, but sets trainer.losses

    # Log key metrics to TensorBoard
    step = trainer.global_step  # Use global_step as the x-axis
    # Losses from trainer.losses (always set after train())
    if 'entropy' in trainer.losses:
        writer.add_scalar('Loss/Entropy', trainer.losses['entropy'], step)
    if 'value_loss' in trainer.losses:
        writer.add_scalar('Loss/Value', trainer.losses['value_loss'], step)
    if 'policy_loss' in trainer.losses:
        writer.add_scalar('Loss/Policy', trainer.losses['policy_loss'], step)
    if 'clipfrac' in trainer.losses:
        writer.add_scalar('Loss/Clipfrac', trainer.losses['clipfrac'], step)
    if 'approx_kl' in trainer.losses:
        writer.add_scalar('Loss/Approx_KL', trainer.losses['approx_kl'], step)
    if 'explained_variance' in trainer.losses:
        writer.add_scalar('Loss/Explained_Variance', trainer.losses['explained_variance'], step)
    writer.add_scalar('Performance/SPS', trainer.sps, step)
    # Custom env stats from mean_and_log() return
    if 'fidelity' in stats_logs:  # Now from stats_logs
        writer.add_scalar('Env/Fidelity', stats_logs['fidelity'], step)
    if 'steps' in stats_logs:  # From env _get_info(); average episode length
        writer.add_scalar('Env/Episode_Length', stats_logs['steps'], step)

    # Optional: Log the current ent_coef for monitoring decay
    writer.add_scalar('Hyperparams/Ent_Coef', trainer.config['ent_coef'], step)

## Save the policy
torch.save(trainer.policy.state_dict(), 'models/ppo_quantum_500k_no_noise_rnn.pth')

## Print the dashboard (keep for console reference)
trainer.print_dashboard()

## Close the environment and TensorBoard writer
trainer.close()
vecenv.close()
writer.close()