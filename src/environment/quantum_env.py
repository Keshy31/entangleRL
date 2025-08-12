import gymnasium as gym
from gymnasium import spaces
import numpy as np
import qutip

class QuantumPrepEnv(gym.Env):
    """
    Custom Gymnasium Environment for Quantum State Preparation.

    The agent's goal is to apply a sequence of quantum gates to transform
    an initial quantum state (e.g., |00>) to a target state (e.g., a Bell state)
    with the highest possible fidelity.

    This environment uses QuTiP to simulate the quantum system.

    ### Observation Space
    The observation is the quantum state vector of the system. Since neural
    networks work with flat arrays of floats, the complex state vector is
    flattened and split into its real and imaginary parts.
    - For a 2-qubit system, the state is a 4x1 vector. The observation is a
      flat array of 8 floats (4 real, 4 imaginary).
    - Type: spaces.Box(low=-1, high=1, shape=(2 * 2**num_qubits,), dtype=np.float32)

    ### Action Space
    The agent can choose from a discrete set of quantum gates to apply.
    - For a 2-qubit system, the actions are:
      - 0: Hadamard gate on Qubit 0
      - 1: Hadamard gate on Qubit 1
      - 2: Pauli-X gate on Qubit 0
      - 3: Pauli-X gate on Qubit 1
      - 4: CNOT gate (Control: 0, Target: 1)
    - Type: spaces.Discrete(5)

    ### Reward
    The reward is designed to guide the agent towards the target state. It's
    calculated as the change in fidelity from the previous step.
    - Reward = (current_fidelity^2) - (previous_fidelity^2)
    - A bonus reward of +1 is given for achieving a fidelity > 0.99.
    - A small penalty is applied at each step to encourage shorter solutions.

    ### Episode End
    An episode ends when:
    1. The fidelity to the target state is > 0.99.
    2. The maximum number of steps (e.g., 50) is reached.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, num_qubits=2, target_state=None, max_steps=50):
        """
        Initializes the quantum environment.
        """
        super().__init__()

        self.num_qubits = num_qubits
        self.max_steps = max_steps

        # --- Define Quantum States ---
        # The initial state is |0...0> for the given number of qubits.
        self.initial_state = qutip.tensor([qutip.basis(2, 0)] * self.num_qubits)
        
        # The target state defaults to the Bell state |Φ+⟩ for 2 qubits.
        if target_state is None:
            if self.num_qubits != 2:
                raise ValueError("Default target state is only defined for 2 qubits.")
            self.target_state = qutip.bell_state('00')
        else:
            self.target_state = target_state
            
        # --- Action Space ---
        # We define a discrete set of gates. For a 2-qubit system:
        # H_0, H_1, X_0, X_1, CNOT_01
        self.action_space = spaces.Discrete(5)
        self._gate_map = self._create_gate_map()

        # --- Observation Space ---
        # The state vector's real and imaginary parts.
        obs_shape = 2 * (2**self.num_qubits)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_shape,), dtype=np.float32
        )

        # Initialize state variables
        self.current_state = None
        self.current_step = None
        self.last_fidelity = None

    def _create_gate_map(self):
        """Creates the mapping from action index to a QuTiP gate operator."""
        if self.num_qubits != 2:
            raise NotImplementedError("Gate map is only implemented for 2 qubits.")
        
        # Single qubit gates need to be tensored with Identity for the other qubit
        h_0 = qutip.tensor(qutip.hadamard_gate(), qutip.qeye(2))
        h_1 = qutip.tensor(qutip.qeye(2), qutip.hadamard_gate())
        x_0 = qutip.tensor(qutip.sigmax(), qutip.qeye(2))
        x_1 = qutip.tensor(qutip.qeye(2), qutip.sigmax())
        
        # Two-qubit gate
        cnot_01 = qutip.cnot()
        
        return {0: h_0, 1: h_1, 2: x_0, 3: x_1, 4: cnot_01}

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        
        self.current_state = self.initial_state.copy()
        self.current_step = 0
        
        # Initial fidelity is calculated against the starting state.
        self.last_fidelity = qutip.fidelity(self.current_state, self.target_state)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """Executes one time step by applying a quantum gate."""
        
        # --- Apply Action ---
        gate = self._gate_map[action]
        self.current_state = gate * self.current_state
        self.current_step += 1

        # --- Calculate Reward ---
        current_fidelity = qutip.fidelity(self.current_state, self.target_state)
        
        # Reward is the improvement in fidelity squared, plus a step penalty.
        reward = (current_fidelity**2) - (self.last_fidelity**2)
        reward -= 0.01 # Small penalty to encourage efficiency
        
        self.last_fidelity = current_fidelity
        
        # --- Check for Termination ---
        terminated = False
        if current_fidelity > 0.99:
            reward += 1.0  # Bonus for winning
            terminated = True
        
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """Flattens the complex state vector into a real-valued numpy array."""
        state_vector = self.current_state.full().flatten()
        return np.concatenate((state_vector.real, state_vector.imag)).astype(np.float32)

    def _get_info(self):
        """Returns auxiliary diagnostic information."""
        return {"fidelity": self.last_fidelity, "steps": self.current_step}

    def render(self):
        """Renders the environment for human viewing."""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Fidelity: {self.last_fidelity:.4f}")

    def close(self):
        """Cleans up any resources."""
        pass
