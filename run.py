import subprocess
import sys
import os
import time

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

THEORY_DIR    = os.path.join(BASE, "scripts", "theory")
BENCH_DIR     = os.path.join(BASE, "scripts", "benchmarks")
COMPARE_DIR   = os.path.join(BASE, "scripts", "comparisons")

def run(script_path, extra_args=None, cwd=None):
    cmd = [sys.executable, script_path, "--out", RESULTS]
    if extra_args:
        cmd += extra_args
    label = os.path.basename(script_path)
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=cwd or BASE, timeout=600,
                                capture_output=False, text=True)
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"  ✅  {label} done ({elapsed:.1f}s)")
            return True
        else:
            print(f"  ❌  {label} failed (exit {result.returncode}) after {elapsed:.1f}s")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  {label} timed out after 600s — skipping")
        return False
    except Exception as e:
        print(f"  ❌  {label} exception: {e}")
        return False


# ============================================================
# 1. THEORY SCRIPTS (run from project root so simulate_qfm is found)
# ============================================================
THEORY_SCRIPTS = [
    # Existing analysis scripts  (args matched to each script's argparse)
    ("analyze_convergence_geometry.py",  ["--qubits", "2", "--steps", "10", "--ensemble", "8"]),
    ("analyze_dynamics_thermo.py",       []),   # only --out
    # analyze_expressibility.py skipped: very slow PennyLane sampling (1500 samples × many layers)
    ("analyze_information_dynamics.py",  ["--qubits", "2", "--steps", "8", "--ensemble", "8"]),
    ("analyze_kraus_ptm.py",             []),   # only --out
    ("analyze_lindblad.py",              []),   # only --out
    ("analyze_loss_landscape.py",        ["--qubits", "2", "--steps", "8"]),
    ("analyze_noise_resilience.py",      ["--qubits", "2", "--steps", "8", "--ensemble", "8"]),
    ("analyze_novel_algorithms.py",      []),   # only --out
    ("analyze_phase_transition.py",      ["--qubits", "2", "--steps", "10", "--ensemble", "8"]),
    ("analyze_qfim.py",                  []),   # only --out
    ("analyze_scalability_ablation.py",  []),   # only --out
    ("analyze_transport_comparison.py",  ["--qubits", "2", "--steps", "8", "--ensemble", "8"]),
    ("analyze_visual_comparison.py",     ["--qubits", "2", "--steps", "8", "--ensemble", "8"]),
    # NEW advanced modules
    ("analyze_convergence_bounds.py",    ["--qubits", "2", "--steps", "10", "--ensemble", "8"]),
    ("analyze_quantum_ot.py",            ["--qubits", "2", "--steps", "10", "--ensemble", "8"]),
    ("analyze_error_mitigation.py",      ["--qubits", "1", "--steps", "8",  "--ensemble", "6"]),
    ("analyze_entanglement_scaling.py",  []),   # only --out
    ("analyze_classical_vs_quantum.py",  ["--qubits", "2", "--steps", "8",  "--ensemble", "6"]),
    ("analyze_flow_geometry.py",         ["--qubits", "2", "--steps", "10", "--ensemble", "8"]),
]

# ============================================================
# 2. BENCHMARK SCRIPTS
# ============================================================
BENCH_SCRIPTS = [
    ("benchmark_entanglement.py", []),
    ("benchmark_ring_state.py",   []),
    ("benchmark_tfim.py",         []),
]

# ============================================================
# 3. COMPARISON SCRIPTS
# ============================================================
COMPARE_SCRIPTS = [
    ("compare_ot_cfm_comparison.py", []),
    ("compare_advanced_viz.py",      []),
    # compare_circuit_viz may require display; skip if needed
    ("compare_circuit_viz.py",       []),
]


def main():
    passed, failed = [], []

    print("\n" + "="*60)
    print("  QFM MASTER RUNNER — ALL SCRIPTS")
    print(f"  Output directory: {RESULTS}")
    print("="*60)

    # Theory scripts — run from BASE (so simulate_qfm is importable)
    print("\n### THEORY SCRIPTS ###")
    for script_name, extra in THEORY_SCRIPTS:
        script_path = os.path.join(THEORY_DIR, script_name)
        if not os.path.isfile(script_path):
            print(f"  ⚠️  Skipping missing: {script_name}")
            continue
        ok = run(script_path, extra_args=extra, cwd=BASE)
        (passed if ok else failed).append(script_name)

    # Benchmark scripts
    print("\n### BENCHMARK SCRIPTS ###")
    for script_name, extra in BENCH_SCRIPTS:
        script_path = os.path.join(BENCH_DIR, script_name)
        if not os.path.isfile(script_path):
            print(f"  ⚠️  Skipping missing: {script_name}")
            continue
        ok = run(script_path, extra_args=extra, cwd=BASE)
        (passed if ok else failed).append(script_name)

    # Comparison scripts
    print("\n### COMPARISON SCRIPTS ###")
    for script_name, extra in COMPARE_SCRIPTS:
        script_path = os.path.join(COMPARE_DIR, script_name)
        if not os.path.isfile(script_path):
            print(f"  ⚠️  Skipping missing: {script_name}")
            continue
        ok = run(script_path, extra_args=extra, cwd=BASE)
        (passed if ok else failed).append(script_name)

    # Final summary
    print("\n" + "="*60)
    print(f"  DONE — {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("  FAILED:")
        for f in failed:
            print(f"    - {f}")
    print(f"\n  All results saved to: {RESULTS}/")
    print("="*60)


if __name__ == "__main__":
    main()
