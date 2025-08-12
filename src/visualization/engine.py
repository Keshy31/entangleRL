import pygame
from pygame.locals import *
import torch
import matplotlib.pyplot as plt
from io import BytesIO  # For fig-to-image
from quantum_prep_env import QuantumPrepEnv  # Your env

class VisualizationEngine:
    def __init__(self, model_path='models/ppo_quantum.pth', env_config={}):
        self.env = QuantumPrepEnv(**env_config)  # e.g., noise rates
        self.policy_net = self.load_model(model_path)  # Assume MLP class defined in train.py
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Quantum State Prep Demo')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.running = True

    def load_model(self, path):
        # Load your PPO actor (policy net)
        model = YourPolicyMLP(input_dim=6, output_dim=9)  # Adjust dims
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def run_demo(self, episodes=10, replay_mode=False):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            while not done and self.running:
                self.handle_events()
                action_probs = self.policy_net(torch.tensor(obs).unsqueeze(0))[0]  # Get logits
                action = torch.argmax(action_probs).item()  # Argmax for deterministic
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.render_frame(action_probs, info)
                pygame.time.delay(500)  # Slow for video; adjust for realtime
            if not self.running: break

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False

    def render_frame(self, action_probs, info):
        self.screen.fill((255, 255, 255))  # White BG

        # Bloch Spheres: Generate Matplotlib fig, convert to surface
        fig = self.generate_bloch_fig()
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        bloch_surf = pygame.image.load(buf)
        self.screen.blit(bloch_surf, (50, 50))  # Position

        # Policy Bars: Draw bars for gate probs
        bar_width = 20
        for i, prob in enumerate(action_probs.detach().numpy()):
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