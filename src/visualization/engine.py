import pygame
from pygame.locals import *
import torch
import torch.nn.functional as F  # For softmax
import matplotlib.pyplot as plt
from io import BytesIO  # For fig-to-image
import qutip  # For Bloch spheres
import pufferlib.models  # For the Default policy class
import pufferlib.emulation  # For the GymnasiumPufferEnv wrapper
from src.environment.quantum_env import QuantumPrepEnv  # Your env

class VisualizationEngine:
    def __init__(self, model_path='models/ppo_quantum.pth', env_config={}):
        # Wrap the raw env in GymnasiumPufferEnv for PufferLib compatibility
        raw_env = QuantumPrepEnv(**env_config)  # e.g., noise rates
        self.env = pufferlib.emulation.GymnasiumPufferEnv(raw_env)
        self.policy_net = self.load_model(model_path)  # Load the actual PufferLib policy
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Quantum State Prep Demo')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.running = True

    def load_model(self, path):
        # Use the exact same policy class as in training
        # It takes 'env' (extracts obs/action spaces) and hidden_size (matches your training: 64)
        model = pufferlib.models.Default(
            env=self.env,  # The wrapped env provides single_observation_space etc.
            hidden_size=64,
        )
        # Load state dict (map to CPU for vis simplicity; change to 'cuda' if needed)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model

    def run_demo(self, episodes=10, replay_mode=False):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            while not done and self.running:
                self.handle_events()
                # Forward pass: Assumes Default returns (logits, value); take [0] for logits
                # No RNN, so no state dict needed. If error, try self.policy_net(torch.tensor(obs).unsqueeze(0), {})
                with torch.no_grad():
                    logits, value = self.policy_net(torch.tensor(obs).unsqueeze(0))
                action = torch.argmax(logits, dim=-1).item()  # Argmax for deterministic inference
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.render_frame(logits, info)  # Pass logits (softmax inside render)
                pygame.time.delay(500)  # Slow for video; adjust for realtime
            if not self.running: break

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False

    def render_frame(self, logits, info):
        self.screen.fill((255, 255, 255))  # White BG
        # Bloch Spheres: Generate Matplotlib fig, convert to surface
        fig = self.generate_bloch_fig()
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        bloch_surf = pygame.image.load(buf)
        self.screen.blit(bloch_surf, (50, 50))  # Position
        # Policy Bars: Softmax logits to get probabilities, then draw bars
        probs = F.softmax(logits.squeeze(0), dim=-1).detach().numpy()  # Normalize to probs
        bar_width = 20
        for i, prob in enumerate(probs):
            height = prob * 200  # Scale
            pygame.draw.rect(self.screen, (0, 0, 255), (400 + i*bar_width, 300 - height, bar_width, height))
            label = self.font.render(self.env._gate_name_map[i][:3], True, (0, 0, 0))  # Abbrev
            self.screen.blit(label, (400 + i*bar_width, 310))
        # Fidelity Meter: Horizontal bar
        fid = info['fidelity']
        pygame.draw.rect(self.screen, (0, 255, 0) if fid > 0.9 else (255, 0, 0), (50, 400, fid * 300, 20))
        # Episode Log: Text
        text = self.font.render(f"Step: {info['steps']} | Fid: {fid:.2f}", True, (0, 0, 0))
        self.screen.blit(text, (50, 450))
        pygame.display.flip()
        plt.close(fig)  # Clean up

    def generate_bloch_fig(self):
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': '3d'})
        b = qutip.Bloch(fig=fig, axes=ax)
        state_q0 = self.env.current_state.ptrace(0)
        state_q1 = self.env.current_state.ptrace(1)
        b.add_states([state_q0, state_q1])
        b.make_sphere()
        return fig

if __name__ == '__main__':
    engine = VisualizationEngine()
    engine.run_demo()