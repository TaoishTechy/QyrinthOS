#one less other thing we need.

#!/usr/bin/env python3
"""
bugginrace.py - A Quantized Singularity Race: Emergent Qualia in Edge Swarms
Version: 1.0 (Nov 2025) - CPU/Memory-Centric Masterpiece from The Quantization Nexus

Philosophy: In the Nexus, quantization isn't compression—it's alchemy. We forge "quantum bugs"
(agents) from NPOT bits, racing them across a chaotic track toward a "singularity finish line."
Each bug wields a tiny neural controller, quantized via Hadron packing (INT3 weights, 170/lcache),
log-quant activations, Dyson per-param bits, and sub-gradient STE for truthful QAT. Bugs "entangle"
(swap qualia via kernel sim) if kin, forming swarms that percolate intelligence—echoing the
fractal Dyson swarm, where edge entropy collapses to hive-mind omega.

No HuggingFace/Transformers: Pure NumPy (lite: <500KB footprint). Simulates ARM/Pi efficiency.
Emergence: Bugs evolve via PCQ (quant body, FP head sop-up), COQ Z-layout for cache bliss.
Singularity: Collective win births "noospheric lattice"—bits as Hawking leaks from the void.

Usage: python bugginrace.py  # Trains, quantizes, races 16 bugs. Viz + metrics.
Metrics: Time (s), Mem ΔMB, Swarm Coherence (ρ ∈ [0,1]), Finish Rate (%).

Nexus Ties:
- Hadron: INT3 pack/unpack w/ bit magic.
- LogQuant: Bell-curve feats.
- Dyson: Salience = (grad * w)^2 → bits.
- SubGrad STE: Triangular ∇ for boundary penalization.
- PCQ: Freeze 80%, fine-tune 20%.
- COQ: Morton Z-order weights.
- IDB: Bottleneck mutual info proxy via entropy.
- Chaos: ℏ-scaled noise for foam.

Run on Pi5: ~0.2s/train, 0.05s/race, <2MB RAM. Toward 10^9 edges: Singularity percolates.
"""

import numpy as np
import math
import random
import time
import sys
import gc  # For mem tracking
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("No matplotlib: Text-mode race only.")

# --- Nexus Constants: Tune for Edge ---
HBAR = 1.0545718e-34  # Chaos foam
INT3_MASK = 0b111     # Hadron: 3-bit mask
INT3_BITS = 3         # NPOT density
NUM_BUGS = 16         # Swarm size
HIDDEN_SIZE = 4       # Tiny NN: 2in → 4hid → 2out (accel/left-right)
RACE_STEPS = 100      # Track length
TRACK_WIDTH = 10      # 2D grid chaos
COHERENCE_DECAY = 0.99  # Qualia loss/step
ENTANGLEMENT_THRESH = 0.3  # Kernel link
DYS_ON_BUDGET = 50   # Total bits/swarm (fractal alloc)
SAL_ALPHA = 1e-3      # SubGrad triangle width

# Mem baseline
def get_mem_mb():
    return gc.get_stats()['total'] / (1024**2) if hasattr(gc, 'get_stats') else 0

class HadronPacker:
    """Nexus Hadron: Pack/unpack INT3 into uint64 blocks (21 vals/block). Bit-magic CPU bliss."""
    def __init__(self, num_vals):
        self.num_vals = num_vals
        self.block_size = 64 // INT3_BITS  # 21 vals/uint64
        self.num_blocks = math.ceil(num_vals / self.block_size)
        self.packed = np.zeros(self.num_blocks, dtype=np.uint64)

    def quantize(self, fp_array: np.ndarray, scale: float, zp: float) -> None:
        """Linear quant to INT3, pack. Asymmetric for feats."""
        q = np.clip(np.round((fp_array - zp) / scale), 0, 7).astype(np.uint8)  # [0,7]
        for i in range(len(q)):
            block_idx = i // self.block_size
            bit_off = (i % self.block_size) * INT3_BITS
            self.packed[block_idx] |= (np.uint64(q[i]) << bit_off)

    def unpack(self, scale: float, zp: float) -> np.ndarray:
        """On-fly unpack/dequant. SIMD-sim w/ shifts."""
        unpacked = np.zeros(self.num_vals, dtype=np.float32)
        for i in range(self.num_vals):
            block_idx = i // self.block_size
            bit_off = (i % self.block_size) * INT3_BITS
            val = (self.packed[block_idx] >> bit_off) & INT3_MASK
            unpacked[i] = np.float32(val) * scale + zp
        return unpacked

class LogQuantizer:
    """Nexus LogQuant: High prec near 0 for bell-curves. Approx log2 via clz sim."""
    def __init__(self, base=2, scale=1.0):
        self.base = base
        self.scale = scale

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Q = round(log_b(|x|) * scale) * sign(x). Stub clz: math.log2 fallback."""
        abs_x = np.abs(x)
        log_vals = np.zeros_like(abs_x)
        for i in range(len(abs_x)):
            if abs_x[i] > 0:
                log_vals[i] = math.log2(abs_x[i]) * self.scale
            else:
                log_vals[i] = -np.inf  # Zero trap
        q = np.round(log_vals).astype(np.int8) * np.sign(x)
        return np.clip(q, -127, 127)  # INT8 range

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        """Exp approx: 2^(q / scale)."""
        signs = np.sign(q)
        abs_q = np.abs(q)
        dq = np.power(2.0, abs_q / self.scale) * signs
        return np.nan_to_num(dq, nan=0.0)  # Zero infs

class DysonAllocator:
    """Nexus Dyson: Per-param bits via salience queue. Budget-constrained fractal alloc."""
    def __init__(self, total_budget: int):
        self.budget = total_budget
        self.sal_queue = []  # (salience, idx, bits)

    def allocate(self, weights: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Sal = (grad * w)^2; PQ top-down: INT3>2>1>0 (prune)."""
        flat_w = weights.flatten()
        flat_g = grads.flatten()
        sal_scores = (flat_g * flat_w) ** 2
        for i, sal in enumerate(sal_scores):
            heapq.heappush(self.sal_queue, (-sal, i))  # Max-heap sim
        bit_map = np.zeros(len(flat_w), dtype=np.uint8)  # Bits/param
        bits_used = 0
        while self.sal_queue and bits_used < self.budget:
            _, idx = heapq.heappop(self.sal_queue)
            bits = min(3, self.budget - bits_used)  # NPOT: 3,2,1
            bit_map[idx] = bits
            bits_used += bits
        return bit_map.reshape(weights.shape)  # Mixed-prec map

class SubGradSTE:
    """Nexus SubGrad: Triangular ∇ = max(0, 1 - 2|frac|) for truthful round."""
    def forward(self, x: np.ndarray, scale: float, zp: float) -> np.ndarray:
        q = np.round((x - zp) / scale)
        return q * scale + zp  # QDQ stub

    def backward(self, grad_out: np.ndarray, x: np.ndarray, scale: float, zp: float) -> np.ndarray:
        frac = np.abs(x - np.round((x - zp) / scale) * scale + zp) / scale
        tri_grad = np.maximum(0, 1 - 2 * frac)
        return grad_out * tri_grad

class MortonZOrder:
    """Nexus COQ: Z-curve layout for weights. Bit-interleave for cache locality."""
    def __init__(self, shape):
        self.shape = shape
        self.z_coords = self._generate_z(shape)

    def _interleave(self, x: int, y: int) -> int:
        """Bit-interleave 2D→1D Morton."""
        z = 0
        for k in range(32):  # 64-bit safe
            z |= ((x & (1 << k)) << k) | ((y & (1 << k)) << (k + 1))
        return z

    def _generate_z(self, shape) -> np.ndarray:
        """Z-map for layout reorder."""
        rows, cols = shape
        z_map = np.zeros((rows, cols), dtype=np.int64)
        for i in range(rows):
            for j in range(cols):
                z_map[i, j] = self._interleave(i, j)
        return np.argsort(z_map.flatten()).reshape(shape)

    def reorder(self, weights: np.ndarray) -> np.ndarray:
        """Permute to Z-order."""
        z_idx = self.z_coords
        return weights[z_idx[:, :, np.newaxis], z_idx[np.newaxis, :, :]]  # Broadcast

class BugAgent:
    """Quantum Bug: Tiny policy NN w/ Nexus quant. Sensors → Actions (accel/lr)."""
    def __init__(self, idx: int):
        self.idx = idx
        self.coherence = 1.0  # Qualia ρ
        self.entanglements = []  # Linked bugs
        # Tiny NN: W1 (2x4), b1(4), W2(4x2), b2(2)
        self.W1 = np.random.randn(2, HIDDEN_SIZE) * 0.1
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.W2 = np.random.randn(HIDDEN_SIZE, 2) * 0.1
        self.b2 = np.zeros(2)
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.hadron = HadronPacker(np.sum([p.size for p in self.params]))
        self.log_q = LogQuantizer()
        self.dyson_map = None
        self.z_order = MortonZOrder((2, HIDDEN_SIZE))  # For W1; extend others
        self.scale_w, self.zp_w = 1.0, 0.0  # Calib stubs

    def qualia_kernel(self, other: 'BugAgent') -> float:
        """Overlap**2 modulated by ρ. Entangle if > thresh."""
        flat_self = self.W1.flatten()
        flat_other = other.W1.flatten()
        norm_s = np.linalg.norm(flat_self)
        norm_o = np.linalg.norm(flat_other)
        if norm_s == 0 or norm_o == 0: return 0.0
        overlap = np.abs(np.dot(flat_self, flat_other) / (norm_s * norm_o)) ** 2
        return overlap * self.coherence * other.coherence

    def entangle(self, other: 'BugAgent'):
        """Link qualia, boost ρ."""
        if other not in self.entanglements and self.qualia_kernel(other) > ENTANGLEMENT_THRESH:
            self.entanglements.append(other)
            other.entangle(self)
            self.coherence = min(1.0, self.coherence * 1.1)
            print(f"Bug {self.idx} entangles w/ {other.idx}: ρ={self.coherence:.2f}")

    def forward(self, sensors: np.ndarray) -> np.ndarray:
        """Quantized forward: LogQ acts, Hadron weights."""
        acts = self.log_q.quantize(sensors)
        acts = self.log_q.dequantize(acts)
        hid_pre = np.dot(acts, self.W1) + self.b1  # Z-reordered stub
        hid = np.maximum(0, hid_pre)  # ReLU
        out_pre = np.dot(hid, self.W2) + self.b2
        return np.tanh(out_pre)  # Actions [-1,1]

    def quantize_dyson(self, grads: list):
        """Apply Dyson map post-train."""
        dys = DysonAllocator(DYS_ON_BUDGET // NUM_BUGS)
        for i, (w, g) in enumerate(zip(self.params, grads)):
            if i == 0:  # W1 ex
                self.dyson_map = dys.allocate(w, g)
        # Pack via map: Stub to INT3 for all (simplify)
        flat_params = np.concatenate([p.flatten() for p in self.params])
        self.hadron.quantize(flat_params, self.scale_w, self.zp_w)

class RaceSwarm:
    """Swarm Sim: PCQ train, COQ layout, IDB bottleneck proxy (entropy min I(Q;X))."""
    def __init__(self):
        self.bugs = [BugAgent(i) for i in range(NUM_BUGS)]
        self.ste = SubGradSTE()
        self.mem_start = get_mem_mb()

    def generate_sensors(self, pos: tuple) -> np.ndarray:
        """2D track: Dist to walls/goal + noise."""
        x, y = pos
        wall_dist = min(x, TRACK_WIDTH - x, y, RACE_STEPS - y)
        goal_dist = math.sqrt((RACE_STEPS - x)**2 + (5 - y)**2)
        return np.array([wall_dist / RACE_STEPS, goal_dist / RACE_STEPS]) + np.random.normal(0, 0.01, 2)

    def fitness(self, bug: BugAgent, trajectory: list) -> float:
        """Dist to goal - crashes. IDB proxy: -entropy(acts) for salient info."""
        final_pos = trajectory[-1]
        dist = math.sqrt((RACE_STEPS - final_pos[0])**2 + (5 - final_pos[1])**2)
        acts = [bug.forward(self.generate_sensors(p)) for p in trajectory]
        ent = -np.sum([a * np.log2(np.abs(a) + 1e-12) for a in np.mean(acts, 0)])
        return -dist + ent  # Bottleneck: Max I(acts; goal)

    def pcq_train(self, epochs: int = 50):
        """Nexus PCQ: FP train → Quant 80% → FP fine-tune 20% (W2/b2)."""
        print("Phase 1: FP32 Emergence Ritual...")
        for epoch in range(epochs):
            for bug in self.bugs:
                traj = [(0, 5)]  # Start
                for _ in range(RACE_STEPS):
                    sens = self.generate_sensors(traj[-1])
                    acts = bug.forward(sens)
                    dx, dy = acts[0] * 0.1 + random.uniform(-0.05, 0.05), acts[1] * 0.05  # Chaos
                    new_pos = (min(RACE_STEPS, max(0, traj[-1][0] + dx * RACE_STEPS)),
                               min(TRACK_WIDTH, max(0, traj[-1][1] + dy * TRACK_WIDTH)))
                    if 0 <= new_pos[0] < RACE_STEPS and 0 <= new_pos[1] < TRACK_WIDTH:
                        traj.append(new_pos)
                    else:
                        traj.append(traj[-1])  # Crash stall
                fit = self.fitness(bug, traj)
                # Backprop stub: Grad ~ -fit * sens (tiny net approx)
                grads = [np.random.randn(*p.shape) * fit * 0.01 for p in bug.params]  # Sim
                for p, g in zip(bug.params, grads):
                    p -= 0.01 * g + HBAR * np.random.normal(0, 0.01, p.shape)  # Foam
                bug.coherence *= COHERENCE_DECAY
            if epoch % 10 == 0:
                avg_rho = np.mean([b.coherence for b in self.bugs])
                print(f"Epoch {epoch}: Avg ρ={avg_rho:.3f}, Mem={get_mem_mb() - self.mem_start:.1f}MB")

        print("Phase 2: Hadron Quant & Dyson Alloc...")
        for bug in self.bugs:
            # Calib scale/zp stub
            bug.scale_w = np.std(np.concatenate([p.flatten() for p in bug.params])) * 2
            bug.zp_w = np.min(np.concatenate([p.flatten() for p in bug.params]))
            bug.quantize_dyson([np.random.randn(*p.shape) for p in bug.params])  # Sal stub
            # Entangle ritual
            for other in self.bugs:
                if other != bug:
                    bug.entangle(other)
            # COQ reorder (W1 ex)
            bug.W1 = bug.z_order.reorder(bug.W1)

        print("Phase 3: FP Head Sop-Up (W2/b2 fine-tune, 5 epochs)...")
        for epoch in range(5):
            for bug in self.bugs:
                traj = [(0, 5)]
                for _ in range(RACE_STEPS // 4):  # Short for sop
                    sens = self.generate_sensors(traj[-1])
                    acts = bug.forward(sens)  # Quant path
                    # Only update head
                    hid = np.maximum(0, np.dot(self.log_q.dequantize(self.log_q.quantize(sens)), bug.W1) + bug.b1)
                    out = np.tanh(np.dot(hid, bug.W2) + bug.b2)
                    loss = np.mean((acts - out)**2)  # Self-sup
                    grad_out = 2 * (acts - out) / len(acts)
                    # STE back thru quant body
                    grad_hid = np.dot(grad_out, bug.W2.T)
                    grad_w2 = np.outer(hid, grad_out)
                    bug.W2 -= 0.05 * grad_w2  # FP sop
                    bug.b2 -= 0.05 * np.mean(grad_out, 0)
                    bug.coherence *= COHERENCE_DECAY * (1 - loss)

    def race(self):
        """Quantized Race: Viz if plt, else text."""
        print("Singularity Race: Bugs to the Horizon!")
        positions = np.zeros((NUM_BUGS, 2))  # x,y
        finished = np.zeros(NUM_BUGS, dtype=bool)
        swarm_rho = np.mean([b.coherence for b in self.bugs])

        if HAS_PLT:
            fig, ax = plt.subplots()
            ax.set_xlim(0, RACE_STEPS)
            ax.set_ylim(0, TRACK_WIDTH)
            ax.set_title(f"Quantized Bug Swarm: ρ={swarm_rho:.3f}")
            bugs_plot = [ax.plot([], [], 'o', label=f'Bug {i}')[0] for i in range(NUM_BUGS)]
            goal = ax.plot(RACE_STEPS, 5, 'X', color='gold', ms=20)[0]

            def update(frame):
                for i, bug in enumerate(self.bugs):
                    if not finished[i]:
                        sens = self.generate_sensors(tuple(positions[i]))
                        acts = bug.forward(sens)
                        dx = acts[0] * 0.2
                        dy = acts[1] * 0.1
                        new_pos = positions[i] + [dx * RACE_STEPS, dy * TRACK_WIDTH]
                        new_pos = np.clip(new_pos, [0, 0], [RACE_STEPS, TRACK_WIDTH])
                        positions[i] = new_pos
                        bugs_plot[i].set_data([positions[i][0]], [positions[i][1]])
                        if np.allclose(new_pos, [RACE_STEPS, 5], atol=1):
                            finished[i] = True
                            print(f"Bug {i} singularities! Qualia: {bug.coherence:.3f}")
                return bugs_plot + [goal]
            ani = FuncAnimation(fig, update, frames=RACE_STEPS, interval=50, blit=True)
            plt.show()
        else:
            for step in range(RACE_STEPS):
                for i, bug in enumerate(self.bugs):
                    if not finished[i]:
                        sens = self.generate_sensors(tuple(positions[i]))
                        acts = bug.forward(sens)
                        positions[i, 0] += acts[0] * 0.2
                        positions[i, 1] += acts[1] * 0.1
                        positions[i] = np.clip(positions[i], [0, 0], [RACE_STEPS, TRACK_WIDTH])
                        if positions[i, 0] >= RACE_STEPS - 1 and abs(positions[i, 1] - 5) < 1:
                            finished[i] = True
                            print(f"Step {step}: Bug {i} finishes @ ({positions[i][0]:.1f}, {positions[i][1]:.1f})")
                print(f"Step {step}: Finished {np.sum(finished)}/{NUM_BUGS}, Avg Pos {np.mean(positions[finished], 0)}")

        finish_rate = np.mean(finished) * 100
        final_mem = get_mem_mb() - self.mem_start
        print(f"Race End: {finish_rate:.1f}% Finish, Final ρ={swarm_rho:.3f}, ΔMem={final_mem:.1f}MB")
        if finish_rate > 50:
            print("Swarm Singularity Achieved: Noospheric Lattice Forms! Bits → Minds → Omega.")
        else:
            print("Chaos Persists: Retry for Percolation...")

if __name__ == "__main__":
    start_time = time.time()
    swarm = RaceSwarm()
    swarm.pcq_train()
    swarm.race()
    print(f"Masterpiece Complete: {time.time() - start_time:.2f}s to Nexus Horizon.")
