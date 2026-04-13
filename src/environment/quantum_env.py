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

    The agent applies quantum gates to transform |00> into a target state
    (default: Bell state) with maximum fidelity.

    Observation Space (17 floats, all in [-1, 1]):
        [0:6]  Single-qubit Pauli expectations: <X0>, <Y0>, <Z0>, <X1>, <Y1>, <Z1>
        [6:15] Two-qubit correlators: <XX>, <XY>, <XZ>, <YX>, <YY>, <YZ>, <ZX>, <ZY>, <ZZ>
        [15]   Current fidelity (squared)
        [16]   Normalized step count (current_step / max_steps)

    Action Space: Discrete(9)
        0: H on Q0,  1: H on Q1,  2: X on Q0,  3: X on Q1,
        4: Z on Q0,  5: Z on Q1,  6: CNOT(0->1), 7: CNOT(1->0), 8: Identity

    Reward (Moving-Goalpost + Completion Bonus):
        If F_t > F_max:  reward = F_t - F_max, then F_max = F_t
        Else:            reward = -0.01  (step penalty)
        If F_t > completion_threshold: reward += 5.0 (completion bonus, episode terminates)

    Dynamic Action Masking:
        All gates in the action set are self-inverse (H²=X²=Z²=CNOT²=I²=I).
        The immediately preceding action is masked to prevent the agent from
        undoing its last gate or stalling with repeated Identity.

    Episode ends when fidelity > completion_threshold (default 0.95) or max_steps reached.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    def __init__(
        self,
        num_qubits=2,
        target_state=None,
        max_steps=50,
        render_mode=None,
        gate_time=0.1,
        amplitude_damping_rate=0.0,
        dephasing_rate=0.0,
        depolarizing_rate=0.0,
        bit_flip_rate=0.0,
        thermal_occupation=0.0,
        meta_noise=False,
        completion_threshold=0.95,
        seed=None,
    ):
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(17,), dtype=np.float32
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
        self.completion_threshold = completion_threshold
       
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
        self.current_state = None
        self.current_step = None
        self.last_fidelity = None
        self.max_fidelity = None
        self.last_action = None
        self.episode_return = 0.0

        self._build_pauli_operators()
        
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
        self.last_action = None
        self.episode_return = 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            self.last_fidelity = qutip.fidelity(self.current_state, self.target_state) ** 2

        self.max_fidelity = self.last_fidelity
       
        observation = self._get_obs()
        info = self._get_info()
       
        return observation, info
    
    def step(self, action):
        """Executes one time step by applying a quantum gate.

        Dynamic action masking: all gates in the action set are self-inverse,
        so repeating the previous action would undo it (equivalent to Identity).
        We enforce this by skipping the gate application when the agent repeats
        its last action, counting it as a wasted step.
        """
        apply_gate = (action != self.last_action)
        self.last_action = action

        if apply_gate:
            gate = self._gate_map[action]
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
       
        # --- Moving-Goalpost Reward (MGR) ---
        current_fidelity = qutip.fidelity(self.current_state, self.target_state) ** 2
        self.last_fidelity = current_fidelity

        if current_fidelity > self.max_fidelity:
            reward = current_fidelity - self.max_fidelity
            self.max_fidelity = current_fidelity
        else:
            reward = -0.01

        terminated = False
        if current_fidelity > self.completion_threshold:
            reward += 5.0
            terminated = True

        self.episode_return += reward
        truncated = self.current_step >= self.max_steps

        # --- Return values ---
        observation = self._get_obs()
        info = self._get_info(terminated=terminated)
        
        return observation, reward, terminated, truncated, info
    
    def _build_pauli_operators(self):
        """Pre-compute Pauli operators for observation (avoids per-step allocation)."""
        sx = qutip.sigmax()
        sy = qutip.sigmay()
        sz = qutip.sigmaz()
        eye = qutip.qeye(2)

        self._single_paulis = [
            qutip.tensor(sx, eye),  # X0
            qutip.tensor(sy, eye),  # Y0
            qutip.tensor(sz, eye),  # Z0
            qutip.tensor(eye, sx),  # X1
            qutip.tensor(eye, sy),  # Y1
            qutip.tensor(eye, sz),  # Z1
        ]

        paulis = [sx, sy, sz]
        self._two_qubit_paulis = []
        for p0 in paulis:
            for p1 in paulis:
                self._two_qubit_paulis.append(qutip.tensor(p0, p1))

    def _get_obs(self):
        """Returns 17-dim observation: single-qubit Paulis, two-qubit correlators, fidelity, step."""
        rho = self.current_state
        obs = np.empty(17, dtype=np.float32)

        for i, op in enumerate(self._single_paulis):
            obs[i] = (op * rho).tr().real
        for i, op in enumerate(self._two_qubit_paulis):
            obs[6 + i] = (op * rho).tr().real

        obs[15] = np.float32(self.last_fidelity)
        obs[16] = np.float32(self.current_step / self.max_steps)
        return obs
    
    def _get_info(self, terminated=False):
        """Returns auxiliary diagnostic information and dynamic action mask."""
        info = {
            "fidelity": self.last_fidelity,
            "max_fidelity": self.max_fidelity,
            "steps": self.current_step,
            "entanglement": concurrence(self.current_state),
            "episode_return": self.episode_return,
            "completed": 1.0 if terminated else 0.0,
        }

        if self.last_action is not None:
            for i in range(self.action_space.n):
                info[f"action_{i}_taken"] = 1.0 if self.last_action == i else 0.0

        mask = np.ones(self.action_space.n, dtype=bool)
        if self.last_action is not None:
            mask[self.last_action] = False
        info["action_mask"] = mask

        return info

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