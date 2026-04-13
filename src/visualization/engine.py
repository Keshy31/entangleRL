import pygame
from pygame.locals import *
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import qutip
import pufferlib.models
import pufferlib.emulation
from src.environment.quantum_env import QuantumPrepEnv


class VisualizationEngine:
    def __init__(self, model_path='models/mlp_100k_baseline.pth', env_config=None):
        if env_config is None:
            env_config = {}
        raw_env = QuantumPrepEnv(**env_config)
        self.env = pufferlib.emulation.GymnasiumPufferEnv(raw_env)
        self.policy_net = self.load_model(model_path)
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Quantum State Prep Demo')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.running = True

    def load_model(self, path):
        model = pufferlib.models.Default(
            env=self.env,
            hidden_size=128,
        )
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model

    def run_demo(self, episodes=10):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            print(f"\n--- Episode {ep + 1} ---")
            step = 0
            while not done and self.running:
                self.handle_events()
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    logits, value = self.policy_net(obs_t)
                action = torch.argmax(logits, dim=-1).item()
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                step += 1
                gate_name = self.env.env._gate_name_map.get(action, str(action))
                print(f"  Step {step}: {gate_name:15s}  fid={info['fidelity']:.4f}  rew={reward:+.4f}")
                self.render_frame(logits, info)
                pygame.time.delay(500)
            if not self.running:
                break

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False

    def render_frame(self, logits, info):
        self.screen.fill((255, 255, 255))

        fig = self.generate_bloch_fig()
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        buf.seek(0)
        bloch_surf = pygame.image.load(buf)
        self.screen.blit(bloch_surf, (50, 30))

        probs = F.softmax(logits.squeeze(0), dim=-1).detach().numpy()
        bar_width = 25
        bar_x_start = 420
        gate_names = self.env.env._gate_name_map
        for i, prob in enumerate(probs):
            height = int(prob * 200)
            x = bar_x_start + i * (bar_width + 5)
            pygame.draw.rect(self.screen, (60, 60, 200), (x, 350 - height, bar_width, height))
            label = self.font.render(gate_names[i][:3], True, (0, 0, 0))
            self.screen.blit(label, (x, 355))

        fid = info['fidelity']
        bar_y = 420
        pygame.draw.rect(self.screen, (200, 200, 200), (50, bar_y, 300, 25))
        color = (30, 180, 30) if fid > 0.9 else (200, 60, 60)
        pygame.draw.rect(self.screen, color, (50, bar_y, int(fid * 300), 25))
        fid_label = self.font.render(f"Fidelity: {fid:.3f}", True, (0, 0, 0))
        self.screen.blit(fid_label, (360, bar_y))

        step_text = self.font.render(f"Step: {info['steps']}", True, (0, 0, 0))
        self.screen.blit(step_text, (50, 460))
        ent_text = self.font.render(f"Entanglement: {info['entanglement']:.3f}", True, (0, 0, 0))
        self.screen.blit(ent_text, (200, 460))

        pygame.display.flip()
        plt.close(fig)

    def generate_bloch_fig(self):
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': '3d'})
        b = qutip.Bloch(fig=fig, axes=ax)
        b.vector_color = ['#e74c3c', '#2ecc71']
        state_q0 = self.env.env.current_state.ptrace(0)
        state_q1 = self.env.env.current_state.ptrace(1)
        b.add_states([state_q0, state_q1])
        b.make_sphere()
        return fig

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    engine = VisualizationEngine()
    try:
        engine.run_demo()
    finally:
        engine.close()
