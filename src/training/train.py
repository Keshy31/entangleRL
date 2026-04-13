import pufferlib
import pufferlib.vector
import pufferlib.models
from pufferlib.pufferl import PuffeRL
from src.environment.quantum_env import QuantumPrepEnv
import torch
from torch.utils.tensorboard import SummaryWriter

def env_creator(**kwargs):
    env = QuantumPrepEnv(meta_noise=False)
    return pufferlib.emulation.GymnasiumPufferEnv(env, **kwargs)

# --- Configuration ---
args = pufferlib.pufferl.load_config('default')

args['train']['env'] = 'quantum_prep'
args['train']['use_rnn'] = False
args['train']['seed'] = 42
args['train']['batch_size'] = 2048
args['train']['bptt_horizon'] = 'auto'
args['train']['minibatch_size'] = 512
args['train']['max_minibatch_size'] = 2048
args['train']['total_timesteps'] = 100_000
args['train']['learning_rate'] = 3e-4
args['train']['update_epochs'] = 4
args['train']['gamma'] = 0.99
args['train']['gae_lambda'] = 0.95
args['train']['clip_coef'] = 0.2
args['train']['ent_coef'] = 0.01
args['train']['vf_coef'] = 0.5
args['train']['clip_vloss'] = True

args['vec']['backend'] = 'Multiprocessing'
args['vec']['num_envs'] = 32
args['vec']['num_workers'] = 4
args['vec']['batch_size'] = 8

# --- Vectorized Environment ---
vecenv = pufferlib.vector.make(
    env_creator,
    backend=args['vec']['backend'],
    num_workers=args['vec']['num_workers'],
    num_envs=args['vec']['num_envs'],
    batch_size=args['vec']['batch_size'],
)

# --- Policy (MLP only -- no LSTM until environment learnability is proven) ---
policy = pufferlib.models.Default(
    env=vecenv,
    hidden_size=128,
).cuda()

# --- Trainer ---
trainer = PuffeRL(args['train'], vecenv, policy)

writer = SummaryWriter(log_dir='logs/tensorboard/mlp_100k_baseline')
total_timesteps = args['train']['total_timesteps']

while trainer.global_step < total_timesteps:
    trainer.evaluate()

    progress = trainer.global_step / total_timesteps
    trainer.config['ent_coef'] = max(0.005, 0.01 * (1 - progress))

    # train() calls mean_and_log() internally and returns the combined logs
    # (env stats under 'environment/*', losses under 'losses/*', etc.)
    # It returns None on non-logging steps.
    logs = trainer.train()

    if logs is None:
        continue

    step = trainer.global_step

    # Loss metrics (populated by train -> mean_and_log -> self.losses)
    for key in ('policy_loss', 'value_loss', 'entropy', 'approx_kl',
                'clipfrac', 'explained_variance'):
        full_key = f'losses/{key}'
        if full_key in logs:
            writer.add_scalar(f'Loss/{key}', logs[full_key], step)

    if 'SPS' in logs:
        writer.add_scalar('Performance/SPS', logs['SPS'], step)

    # Environment info metrics
    if 'environment/fidelity' in logs:
        writer.add_scalar('Env/Fidelity', logs['environment/fidelity'], step)
    if 'environment/steps' in logs:
        writer.add_scalar('Env/Episode_Length', logs['environment/steps'], step)
    if 'environment/entanglement' in logs:
        writer.add_scalar('Env/Entanglement', logs['environment/entanglement'], step)

    for i in range(9):
        action_key = f'environment/action_{i}_taken'
        if action_key in logs:
            writer.add_scalar(f'Actions/Action_{i}', logs[action_key], step)

    writer.add_scalar('Hyperparams/Ent_Coef', trainer.config['ent_coef'], step)

torch.save(trainer.policy.state_dict(), 'models/mlp_100k_baseline.pth')
trainer.close()
vecenv.close()
writer.close()
