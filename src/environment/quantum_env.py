import gymnasium
from gymnasium import spaces
import numpy as np
import qutip
import matplotlib.pyplot as plt
from qutip.visualization import matrix_histogram
from qutip import gates
from qutip import concurrence
import os
import warnings
from scipy.linalg import LinAlgWarning

class QuantumPrepEnv(gymnasium.Env):
    """
    Custom Gymnasium Environment for Quantum State Preparation.
    The agent's goal is to apply a sequence of quantum gates to transform
    an initial quantum state (e.g., |00>) to a target state (e.g., a Bell state)
    with the highest possible fidelity.
    This environment uses QuTiP to simulate the quantum system.
    ### Observation Space
    The observation consists of expectation values for Pauli operators on each qubit,
    simulating partial observability: ⟨σ_x0⟩, ⟨σ_y0⟩, ⟨σ_z0⟩, ⟨σ_x1⟩, ⟨σ_y1⟩, ⟨σ_z1⟩.
    - For a 2-qubit system, this is a 6-float array (values between -1 and 1).
    - Type: spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
    ### Action Space
    The agent can choose from a discrete set of quantum gates to apply.
    This expanded set provides more control, especially over phase, and is
    better suited for finding efficient solutions and handling noise.
    - Type: spaces.Discrete(9)
      - 0: Hadamard gate on Qubit 0
      - 1: Hadamard gate on Qubit 1
      - 2: Pauli-X gate on Qubit 0
      - 3: Pauli-X gate on Qubit 1
      - 4: Pauli-Z gate on Qubit 0
      - 5: Pauli-Z gate on Qubit 1
      - 6: CNOT gate (Control: 0, Target: 1)
      - 7: CNOT gate (Control: 1, Target: 0)
      - 8: Identity (No-op)
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
    def __init__(
        self,
        num_qubits=2,
        target_state=None,
        max_steps=30,
        render_mode=None,
        gate_time=0.1,
        amplitude_damping_rate=0.0,
        dephasing_rate=0.0,
        depolarizing_rate=0.0,
        bit_flip_rate=0.0,
        thermal_occupation=0.0,
        meta_noise=False,
        seed=None  # Added to handle seed passed by PufferLib
    ):
        """
        Initializes the quantum environment.
        """
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)
       
        # Store seed if needed (e.g., for RNG in env)
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
       
        # Core environment parameters
        self.num_qubits = num_qubits
        self.max_steps = max_steps
       
        # --- Noise Parameters ---
        self.gate_time = gate_time
        self.amplitude_damping_rate = amplitude_damping_rate
        self.dephasing_rate = dephasing_rate
        self.depolarizing_rate = depolarizing_rate
        self.bit_flip_rate = bit_flip_rate
        self.thermal_occupation = thermal_occupation  # n_th >=0; if >0, adds excitation to amplitude damping
        self.meta_noise = meta_noise
       
        # --- Define Quantum States (as Density Matrices) ---
        # The initial state is |0...0> for the given number of qubits.
        initial_ket = qutip.tensor([qutip.basis(2, 0)] * self.num_qubits)
        self.initial_state = qutip.ket2dm(initial_ket)
       
        # The target state defaults to the Bell state |Φ+⟩ for 2 qubits.
        if target_state is None:
            if self.num_qubits != 2:
                raise ValueError("Default target state is only defined for 2 qubits.")
            target_ket = qutip.bell_state('00')
            self.target_state = qutip.ket2dm(target_ket)
        else:
            # Ensure target is a density matrix
            self.target_state = target_state if target_state.isoper else qutip.ket2dm(target_state)
           
        # --- Gate Maps ---
        self._gate_map, self._gate_name_map = self._create_gate_maps()
        # Initialize state variables
        self.current_state = None
        self.current_step = None
        self.last_fidelity = None
        
        warnings.filterwarnings('ignore', category=LinAlgWarning)

        # --- Visualization ---
        self.render_mode = render_mode
        self.fig = None
        self.axes = None
        self.bloch = None
        if self.render_mode == "human":
            self.render_path = "renders"
            os.makedirs(self.render_path, exist_ok=True)
            self._initialize_plot()

    def _initialize_plot(self):
        """Initializes the plot for human rendering."""
        self.fig = plt.figure(figsize=(12, 5))
       
        # GridSpec for advanced layout
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1])
        # Axes for the matrix histogram (must be 3D for view_init)
        ax1 = self.fig.add_subplot(gs[0], projection='3d')
       
        # Axes for the Bloch sphere (must be 3D)
        ax2 = self.fig.add_subplot(gs[1], projection='3d')
       
        self.axes = [ax1, ax2]
       
        # Create a single Bloch sphere instance to be reused
        self.bloch = qutip.Bloch(axes=self.axes[1])
        self.bloch.vector_color = ['r', 'g']
       
    def _create_gate_maps(self):
        """Creates the mapping from action index to a QuTiP gate operator and its name."""
        if self.num_qubits != 2:
            raise NotImplementedError("Gate map is only implemented for 2 qubits.")
       
        # Single qubit gates need to be tensored with Identity for the other qubit
        h_0 = qutip.tensor(gates.hadamard_transform(), qutip.qeye(2))
        h_1 = qutip.tensor(qutip.qeye(2), gates.hadamard_transform())
        x_0 = qutip.tensor(qutip.sigmax(), qutip.qeye(2))
        x_1 = qutip.tensor(qutip.qeye(2), qutip.sigmax())
        z_0 = qutip.tensor(qutip.sigmaz(), qutip.qeye(2))
        z_1 = qutip.tensor(qutip.qeye(2), qutip.sigmaz())
       
        # Two-qubit gates
        cnot_01 = gates.cnot()
        # To get CNOT with control=1 and target=0, we sandwich a standard
        # CNOT with SWAP gates.
        swap_gate = gates.swap()
        cnot_10 = swap_gate * cnot_01 * swap_gate
        # Identity gate
        identity = qutip.tensor(qutip.qeye(2), qutip.qeye(2))
       
        gate_map = {
            0: h_0, 1: h_1, 2: x_0, 3: x_1, 4: z_0, 5: z_1,
            6: cnot_01, 7: cnot_10, 8: identity
        }
        gate_name_map = {
            0: "Hadamard Q0",
            1: "Hadamard Q1",
            2: "Pauli-X Q0",
            3: "Pauli-X Q1",
            4: "Pauli-Z Q0",
            5: "Pauli-Z Q1",
            6: "CNOT (0->1)",
            7: "CNOT (1->0)",
            8: "Identity",
        }
        return gate_map, gate_name_map
    
    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        if seed is not None:
            np.random.seed(seed)  # Use reset seed for RNG if provided
        if self.meta_noise:
            self.amplitude_damping_rate = np.random.uniform(0.0, 0.2)
            self.dephasing_rate = np.random.uniform(0.0, 0.1)
            self.depolarizing_rate = np.random.uniform(0.0, 0.05)
            self.bit_flip_rate = np.random.uniform(0.0, 0.05)
            self.thermal_occupation = np.random.uniform(0.0, 0.1)
       
        self.current_state = self.initial_state.copy()
        self.current_step = 0

        self.last_ent = 0
        self.last_superpos = 0

        # Initial fidelity is calculated against the starting state.
        # Suppress LinAlgWarning which is expected for singular density matrices (pure states)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            self.last_fidelity = qutip.fidelity(self.current_state, self.target_state)
       
        observation = self._get_obs()
        info = self._get_info()
       
        return observation, info
    
    def step(self, action):
        """Executes one time step by applying a quantum gate."""
       
        # --- Apply Ideal Gate ---
        gate = self._gate_map[action]
        # Unitary evolution on a density matrix: ρ' = UρU†
        self.current_state = gate * self.current_state * gate.dag()
        # --- Apply Noise using mesolve ---
        c_ops = []
       
        # Amplitude damping (and generalized if thermal_occupation >0)
        if self.amplitude_damping_rate > 0:
            gamma = self.amplitude_damping_rate
            n_th = self.thermal_occupation
            for i in range(self.num_qubits):
                op_list_m = [qutip.qeye(2)] * self.num_qubits
                op_list_m[i] = qutip.sigmam()  # Decay (σ⁻)
                c_ops.append(np.sqrt(gamma * (1 + n_th)) * qutip.tensor(op_list_m))
               
                if n_th > 0:
                    op_list_p = [qutip.qeye(2)] * self.num_qubits
                    op_list_p[i] = qutip.sigmap()  # Excitation (σ⁺)
                    c_ops.append(np.sqrt(gamma * n_th) * qutip.tensor(op_list_p))
       
        # Dephasing
        if self.dephasing_rate > 0:
            gamma = self.dephasing_rate
            for i in range(self.num_qubits):
                op_list = [qutip.qeye(2)] * self.num_qubits
                op_list[i] = qutip.sigmaz()
                c_ops.append(np.sqrt(gamma / 2) * qutip.tensor(op_list))
       
        # Depolarizing (local per qubit)
        if self.depolarizing_rate > 0:
            gamma = self.depolarizing_rate
            for i in range(self.num_qubits):
                for pauli in [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]:
                    op_list = [qutip.qeye(2)] * self.num_qubits
                    op_list[i] = pauli
                    c_ops.append(np.sqrt(gamma / 4) * qutip.tensor(op_list))
       
        # Bit flip (approximate)
        if self.bit_flip_rate > 0:
            gamma = self.bit_flip_rate
            for i in range(self.num_qubits):
                op_list = [qutip.qeye(2)] * self.num_qubits
                op_list[i] = qutip.sigmax()
                c_ops.append(np.sqrt(gamma / 2) * qutip.tensor(op_list))
       
        if len(c_ops) > 0:
            # Evolve under noise for a short time with no Hamiltonian
            h_null = qutip.qzero(self.current_state.dims[0])
            tlist = [0, self.gate_time]
            result = qutip.mesolve(h_null, self.current_state, tlist, c_ops, e_ops=[])
            self.current_state = result.states[-1]
        self.current_step += 1
       
        # --- Calculate Reward ---
        # Fidelity bonus
        fidelity = qutip.fidelity(self.current_state, self.target_state)
        reward = 1.5 * (fidelity - self.last_fidelity)
        if (fidelity - self.last_fidelity) > 0:
            reward += 0.5 * fidelity ** 2
        
        self.last_fidelity = fidelity
        
        if fidelity > 0.75:
            reward += 0.3

        # Superposition bonus
        superpos = np.sum(np.abs(self.current_state.full() - np.diag(np.diag(self.current_state.full()))))  # Your superpos
        reward += 0.4 * (superpos - self.last_superpos)  # Delta, coefficient for balance (see proportions below)
        self.last_superpos = superpos  # Store for next step

        # Entanglement bonus
        ent = concurrence(self.current_state)
        reward += 0.35 * ent - self.last_ent
        self.last_ent = ent

        # Step penalty
        reward -= 0.05 * self.current_step
       
        # --- Check for Termination ---
        terminated = False
        if fidelity > 0.95:
            reward += 1.5  # Bonus for winning
            terminated = True
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
        observation = self._get_obs()

        info = self._get_info() if terminated or truncated else {}
       
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Returns partial observations: Pauli expectation values."""
        sx0 = qutip.tensor(qutip.sigmax(), qutip.qeye(2))
        sy0 = qutip.tensor(qutip.sigmay(), qutip.qeye(2))
        sz0 = qutip.tensor(qutip.sigmaz(), qutip.qeye(2))
        sx1 = qutip.tensor(qutip.qeye(2), qutip.sigmax())
        sy1 = qutip.tensor(qutip.qeye(2), qutip.sigmay())
        sz1 = qutip.tensor(qutip.qeye(2), qutip.sigmaz())
        return np.array([
            (sx0 * self.current_state).tr().real,
            (sy0 * self.current_state).tr().real,
            (sz0 * self.current_state).tr().real,
            (sx1 * self.current_state).tr().real,
            (sy1 * self.current_state).tr().real,
            (sz1 * self.current_state).tr().real
        ], dtype=np.float32)
    
    def _get_info(self):
        """Returns auxiliary diagnostic information."""
        current_fidelity = self.last_fidelity  # Already have
        # Add quantum-specific: Expectation values (partial obs) for analysis
        obs = self._get_obs()  # Your Pauli <σ> array
        # Action probs if during eval (but log in trainer; here add state details)
        return {
            "fidelity": current_fidelity,
            "steps": self.current_step,
            "expectation_sx0": obs[0],  # <σ_x0>
            "expectation_sy0": obs[1],  # <σ_y0>
            "expectation_sz0": obs[2],  # <σ_z0>
            "expectation_sx1": obs[3],  # <σ_x1>
            "expectation_sy1": obs[4],  # <σ_y1>
            "expectation_sz1": obs[5],  # <σ_z1>
            "entanglement": concurrence(self.current_state),
            "superpos": np.sum(np.abs(self.current_state.full() - np.diag(np.diag(self.current_state.full()))))
        }

    def render(self):
        """Renders the environment for human viewing."""
        if self.render_mode == 'human':
            if self.fig is None:
                self._initialize_plot()
            # --- 1. Plot Matrix Histogram ---
            self.axes[0].clear()
            matrix_histogram(self.current_state, ax=self.axes[0])
            self.axes[0].set_title("Current State Density Matrix")
            # --- 2. Plot Bloch Spheres ---
            self.bloch.clear()  # Clear previous states
           
            # Partial trace to get individual qubit states
            state_qbit0 = self.current_state.ptrace(0)
            state_qbit1 = self.current_state.ptrace(1)
           
            # Add new states to the sphere
            self.bloch.add_states([state_qbit0, state_qbit1])
            self.bloch.make_sphere() # Redraw the sphere
            self.axes[1].set_title("Bloch Spheres")
            # --- Update Titles and Draw ---
            self.fig.suptitle(
                f"Step: {self.current_step} | Fidelity: {self.last_fidelity:.4f}",
                fontsize=16
            )
           
            # Save the figure instead of showing it interactively
            save_path = os.path.join(self.render_path, f"step_{self.current_step}.png")
            self.fig.savefig(save_path)
    def close(self):
        """Cleans up any resources."""
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.ioff()
            plt.close(self.fig)