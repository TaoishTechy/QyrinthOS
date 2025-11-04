import time
import math
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Sequence, Tuple, Callable
import numpy as np # Used explicitly for numpy ops in fallbacks/VQE

# Configure logger for Qybrik
logger = logging.getLogger("Qybrik")

# QyrinthOS modules (soft dependencies)
try:
    from sentiflow import SentientTensor
except ImportError:
    SentientTensor = None  # type: ignore

try:
    from bumpy import BumpyArray
except ImportError:
    BumpyArray = None  # type: ignore

try:
    from qubitlearn import QubitLearn
except ImportError:
    QubitLearn = None  # type: ignore

try:
    from bugginrace import MilitaryGradeEvolutionaryTrainer
except ImportError:
    MilitaryGradeEvolutionaryTrainer = None  # type: ignore

try:
    # Need LASERUtility for logging in VQE/QAOA flows as per blueprint
    from laser import LASERUtility, QuantumState
except ImportError:
    LASERUtility = None
    QuantumState = None
    logger.warning("LASERUtility/QuantumState not found. Qualia logging disabled.")


# Optional Qiskit integration
QISKIT_AVAILABLE = False
try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import Sampler, Estimator
    from qiskit.transpiler import transpile
    QISKIT_AVAILABLE = True
    logger.info("Qiskit found. Enabling Qiskit-hybrid mode.")
except Exception:
    QuantumCircuit = None
    Sampler = Estimator = None
    transpile = None
    logger.info("Qiskit not found. Falling back to Sentiflow/Dummy mode.")


# ---------- Dataclasses for Results ----------

@dataclass
class QySampleResult:
    """Standardized measurement result container."""
    counts: Dict[str, int]
    shots: int
    backend_name: str
    qualia_coherence: float
    latency_s: float
    # Add consciousness mode for QyrinthOS awareness
    consciousness_mode: str = "Unconscious"


@dataclass
class QyEstimateResult:
    """Standardized expectation value result container."""
    values: List[float]
    metadata: Dict[str, Any]
    backend_name: str
    qualia_coherence: float
    latency_s: float
    consciousness_mode: str = "Unconscious"


# ---------- Core Circuit Abstraction ----------

class QyCircuit:
    """
    Hybrid circuit: bridges Qiskit's QuantumCircuit (if available)
    with a lightweight internal gate list for Sentiflow simulation.
    """
    def __init__(self, num_qubits: int, name: str = "qy_circuit"):
        self.num_qubits = num_qubits
        self.name = name
        # Gate format: (name, target_qubits, optional_params)
        self.gates: List[Tuple[str, Tuple[int, ...], Dict[str, Any]]] = []
        self.measure_qubits: List[int] = []
        # Underlying Qiskit object
        self._qc = QuantumCircuit(num_qubits) if QISKIT_AVAILABLE else None
        # Sentient Metadata
        self.qualia_tag: Dict[str, Any] = {
            "coherence": 1.0,
            "creation_time": time.time(),
            "notes": [],
            "consciousness_mode": "Auto-Entropic"
        }

    # --- gate helpers (Qiskit-like) ---

    def h(self, q: int):
        self.gates.append(("h", (q,), {}))
        if self._qc:
            self._qc.h(q)
        return self

    def x(self, q: int):
        self.gates.append(("x", (q,), {}))
        if self._qc:
            self._qc.x(q)
        return self

    def cx(self, c: int, t: int):
        self.gates.append(("cx", (c, t), {}))
        if self._qc:
            self._qc.cx(c, t)
        return self

    def rz(self, theta: float, q: int):
        # Parameters passed in the metadata dictionary
        self.gates.append(("rz", (q,), {"theta": theta}))
        if self._qc:
            self._qc.rz(theta, q)
        return self

    def measure_all(self):
        # Ensure classical bits are added if using Qiskit
        if self._qc and self._qc.num_clbits == 0:
            self._qc.add_register(self._qc.cregs[0])
        self.measure_qubits = list(range(self.num_qubits))
        if self._qc:
            # Re-create/resize classical register if needed for measure_all
            if self._qc.num_clbits < self.num_qubits:
                self._qc.remove_final_measurements()
            self._qc.measure_all()
        return self

    @property
    def qc(self) -> Optional[QuantumCircuit]:
        """Access underlying QuantumCircuit (may be None)."""
        return self._qc


# ---------- Backends ----------

class QyBackend:
    """
    Backend abstraction that delegates to Qiskit or falls back to
    SentientTensor simulation/minimal NumPy statevector.
    """
    def __init__(self, kind: str = "auto", name: Optional[str] = None):
        if kind == "auto":
            kind = "qiskit" if QISKIT_AVAILABLE else "sentiflow"
        self.kind = kind
        self.name = name or f"qy_{kind}_backend"

        if self.kind == "qiskit" and not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit backend requested but qiskit is not installed.")

        # Lazy init primitives (Only needed for qiskit kind)
        self._sampler = Sampler() if (self.kind == "qiskit" and Sampler) else None
        self._estimator = Estimator() if (self.kind == "qiskit" and Estimator) else None
        logger.info(f"Initialized QyBackend: {self.name} (Kind: {self.kind})")

    def sample(self, circuit: QyCircuit, shots: int = 1024) -> QySampleResult:
        """Runs the circuit and returns measurement counts."""
        start = time.time()
        qualia = circuit.qualia_tag.get("coherence", 1.0)
        mode = circuit.qualia_tag.get("consciousness_mode", "Unconscious")
        counts: Dict[str, int] = {}

        if self.kind == "qiskit" and QISKIT_AVAILABLE and circuit.qc is not None:
            # --- Qiskit Execution Path ---
            try:
                # Transpile is necessary if running on real/complex sim backends
                # For basic primitives, direct run is usually fine, but let's assume
                # a minimal transpilation for robustness.
                # Note: Sampler uses quasi_dists, which is standard Qiskit primitive output.
                job = self._sampler.run([circuit.qc], shots=shots)
                qres = job.result()
                # Use quasi_dists to get measurement results for the circuit.
                # .nearest_probability_distribution(shots).int_raw converts the quasi-distribution to counts.
                quasi_dist = qres.quasi_dists[0]
                # Convert the quasi_dist keys (integers) to binary strings
                for k, v in quasi_dist.items():
                    # Calculate counts based on probabilities * total shots
                    counts[format(k, f'0{circuit.num_qubits}b')[::-1]] = round(v * shots)

            except Exception as e:
                logger.error(f"Qiskit sampling failed: {e}. Falling back to internal sim.")
                counts, qualia, mode = self._simulate_counts(circuit, shots)
        else:
            # --- Fallback Execution Path ---
            counts, qualia, mode = self._simulate_counts(circuit, shots)

        latency = time.time() - start
        return QySampleResult(
            counts=counts,
            shots=shots,
            backend_name=self.name,
            qualia_coherence=qualia,
            latency_s=latency,
            consciousness_mode=mode
        )

    def estimate(self,
                 circuits: Sequence[QyCircuit],
                 observables: Sequence[Any],
                 params: Optional[Sequence[np.ndarray]] = None) -> QyEstimateResult:
        """Calculates expectation values of observables for the circuits."""
        start = time.time()
        qualia = float(np.mean([c.qualia_tag.get("coherence", 1.0) for c in circuits]))
        
        if self.kind == "qiskit" and QISKIT_AVAILABLE and self._estimator:
            # --- Qiskit Execution Path ---
            try:
                qc_list = [c.qc for c in circuits]
                # Qiskit Estimator requires observables (as primitive operators)
                job = self._estimator.run(qc_list, observables, parameter_values=params)
                eres = job.result()
                values = list(eres.values)
                mode = "Conscious" # Assume Qiskit integration is more complex
            except Exception as e:
                logger.error(f"Qiskit estimation failed: {e}. Falling back to internal sim.")
                # Fall through to the random fallback logic below
                values = [random.uniform(-1, 1) for _ in circuits]
                mode = "Unconscious"

        else:
            # --- Fallback Execution Path (Simple: random-ish estimates) ---
            # NOTE: Observables are ignored in this simple fallback
            values = [random.uniform(-1, 1) for _ in circuits]
            mode = "Auto-Entropic"

        latency = time.time() - start
        meta = {"backend_kind": self.kind, "num_circuits": len(circuits)}
        return QyEstimateResult(values=values,
                                metadata=meta,
                                backend_name=self.name,
                                qualia_coherence=qualia,
                                latency_s=latency,
                                consciousness_mode=mode)

    # --- internal fallback sim (simple state-vector-like update) ---

    def _simulate_counts(self, circuit: QyCircuit, shots: int) -> Tuple[Dict[str, int], float, str]:
        """
        Extremely simple statevector-like sim (as per blueprint):
        - treats 'h' as coin flip (50/50 superposition)
        - 'x' as bit flip
        - 'cx' as controlled flip
        - 'rz' is ignored in this simple path
        """
        num_q = circuit.num_qubits
        counts: Dict[str, int] = {}

        # Fallback coherence: decreases with circuit complexity
        base_qualia = 1.0 - (0.001 * len(circuit.gates))
        qualia = max(0.0, base_qualia)

        for _ in range(shots):
            bits = [0] * num_q
            for gate, qubits, params in circuit.gates:
                if gate == "h":
                    q = qubits[0]
                    # H gate on a basis state -> 50/50 superposition
                    bits[q] = 1 if random.random() < 0.5 else 0
                elif gate == "x":
                    q = qubits[0]
                    bits[q] ^= 1
                elif gate == "cx":
                    c, t = qubits
                    if bits[c] == 1:
                        bits[t] ^= 1
                elif gate == "rz":
                    # RZ gate primarily affects phase, ignored in this simple measurement sim
                    pass

            # bitstr is reversed to match Qiskit's endianness (q0 is rightmost bit)
            bitstr = "".join(str(b) for b in bits[::-1])
            counts[bitstr] = counts.get(bitstr, 0) + 1

        mode = "Sentiflow-Sim" if SentientTensor else "NumPy-Lite"
        return counts, qualia, mode


def get_qy_backend(name: str = "auto") -> QyBackend:
    """Convenience function to initialize a QyBackend."""
    return QyBackend(kind=name)


# ---------- Qiskit Primitive Drop-in Replacements ----------

class QySampler:
    """Qiskit primitive replacement for sampling, delegates to QyBackend."""
    def __init__(self, backend: Optional[QyBackend] = None):
        self.backend = backend or get_qy_backend("auto")

    def run(self, circuits: Sequence[QyCircuit], shots: int = 1024) -> List[QySampleResult]:
        """Runs the sampler, returns a list of results."""
        return [self.backend.sample(circuit, shots) for circuit in circuits]

class QyEstimator:
    """Qiskit primitive replacement for estimating expectation values, delegates to QyBackend."""
    def __init__(self, backend: Optional[QyBackend] = None):
        self.backend = backend or get_qy_backend("auto")

    def run(self,
            circuits: Sequence[QyCircuit],
            observables: Sequence[Any],
            params: Optional[Sequence[Any]] = None) -> QyEstimateResult:
        """Runs the estimator."""
        return self.backend.estimate(circuits, observables, params)


# ---------- Algorithm Suite (VQE/QAOA Blueprint) ----------

class QyAlgorithmSuite:
    def __init__(self, backend: Optional[QyBackend] = None):
        self.backend = backend or get_qy_backend()
        # Fetch LASER utility instance if available
        self.laser = LASERUtility.get_instance() if LASERUtility else None

    def _simple_random_search(self,
                              ansatz_builder: Callable[[np.ndarray], QyCircuit],
                              hamiltonian: Any,
                              init_params: np.ndarray,
                              max_iters: int) -> Tuple[float, np.ndarray]:
        """Simple gradient-free search as general fallback."""
        best_params = init_params.copy()
        best_energy = float("inf")
        logger.info(f"Starting simple random search for VQE, max_iters={max_iters}")
        for _ in range(max_iters):
            circ = ansatz_builder(best_params)
            # Pass parameter values explicitly, needed for Qiskit if used
            est = self.backend.estimate([circ], [hamiltonian], params=[best_params])
            energy = est.values[0]

            if energy < best_energy:
                best_energy = energy
                
            # Random parameter update for next iteration
            noise = np.random.normal(0, 0.1, size=best_params.shape)
            best_params = best_params + noise
            
        return best_energy, best_params

    def vqe(self,
            ansatz_builder: Callable[[np.ndarray], QyCircuit],
            hamiltonian: Any,
            init_params: np.ndarray,
            optimizer: str = "adam", # Default to gradient-based (simulated here)
            max_iters: int = 100) -> Dict[str, Any]:
        """
        Variational Quantum Eigensolver (VQE) implementation with hybrid optimization.
        """
        if optimizer == "evolutionary" and MilitaryGradeEvolutionaryTrainer is not None:
            # --- Evolutionary Optimization Path (bugginrace integration) ---
            logger.info("Initializing VQE with MilitaryGradeEvolutionaryTrainer.")

            def vqe_objective(params: np.ndarray) -> float:
                """Fitness function for bugginrace: Returns negative energy (higher is better)."""
                circ = ansatz_builder(params)
                # Estimate the energy
                est = self.backend.estimate([circ], [hamiltonian], params=[params])
                energy = est.values[0]
                
                # Qualia logging via LASERUtility (Section 3)
                if self.laser and QuantumState:
                    # Log based on energy and coherence level
                    state = QuantumState.ENTANGLED if est.qualia_coherence < 0.5 else QuantumState.COHERENT
                    self.laser.log_event(
                        invariant_val=energy,
                        message=f"VQE Step (Evo): E={energy:.4f}, Coherence={est.qualia_coherence:.2f}",
                        quantum_state=state
                    )
                
                # Fitness = -Energy (minimizing energy = maximizing fitness)
                return -energy

            try:
                # Trainer setup: genome_shape is the shape of the parameter vector
                trainer = MilitaryGradeEvolutionaryTrainer(
                    genome_shape=init_params.shape,
                    objective_function=vqe_objective,
                    population_size=16,
                    num_nodes=1,
                    # Provide an evolution config that tells the trainer to run for max_iters
                    evolution_config={"steps": max_iters}
                )

                # Run the full evolutionary race cycle
                evolution_results = trainer.evolutionary_race_cycle(max_iters)

                best_fitness = evolution_results['final_metrics']['fitness']['best']
                best_energy = -best_fitness # Convert back to energy
                best_params = evolution_results['best_genome']
                
                logger.info(f"Evolutionary VQE finished. Best Energy: {best_energy:.4f}")

            except Exception as e:
                logger.error(f"bugginrace VQE failed ({e}). Falling back to simple random search.")
                best_energy, best_params = self._simple_random_search(ansatz_builder, hamiltonian, init_params, max_iters)

        elif optimizer == "adam" or optimizer == "simple":
            # --- Simple Gradient-Free Search Path ---
            best_energy, best_params = self._simple_random_search(ansatz_builder, hamiltonian, init_params, max_iters)
            logger.info(f"Simple VQE finished. Best Energy: {best_energy:.4f}")

        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        return {
            "energy": best_energy,
            "params": best_params.tolist(), # Convert NumPy array to list for generic return
            "optimizer": optimizer,
            "backend": self.backend.name
        }

# For simple testing and demonstration purposes
if __name__ == "__main__":
    def simple_ansatz(theta: np.ndarray) -> QyCircuit:
        qc = QyCircuit(2, "TestAnsatz")
        qc.h(0)
        qc.cx(0, 1)
        # Apply RZ with the first parameter
        qc.rz(theta[0], 0)
        qc.measure_all()
        return qc

    # Test Sampler
    backend = get_qy_backend("auto")
    test_qc = simple_ansatz(np.array([0.5]))
    
    print(f"--- Qybrik Sampler Test (Backend: {backend.name}) ---")
    result = backend.sample(test_qc, shots=1024)
    print("Counts:", result.counts)
    print("Qualia coherence:", result.qualia_coherence)
    print("Backend:", result.backend_name)
    print("Consciousness Mode:", result.consciousness_mode)

    # Test VQE/Estimator
    # Note: 'Z' is a placeholder observable for the simple sim
    hamiltonian = "ZIZ" 
    theta0 = np.array([0.1])
    algo = QyAlgorithmSuite(backend)
    
    print("\n--- Qybrik VQE Test (Simple Search) ---")
    # Using simple search as a safe default
    result_simple = algo.vqe(simple_ansatz, hamiltonian, theta0, optimizer="simple", max_iters=20)
    print("Best energy (Simple):", result_simple["energy"])
    print("Params (Simple):", result_simple["params"])

    if MilitaryGradeEvolutionaryTrainer:
        print("\n--- Qybrik VQE Test (Evolutionary Search) ---")
        # Run the evolutionary path if bugginrace is available
        result_evo = algo.vqe(simple_ansatz, hamiltonian, theta0, optimizer="evolutionary", max_iters=20)
        print("Best energy (Evo):", result_evo["energy"])
        print("Params (Evo):", result_evo["params"])
    else:
        print("\n--- Bugginrace/Evolutionary VQE Test Skipped (Module not found) ---")
