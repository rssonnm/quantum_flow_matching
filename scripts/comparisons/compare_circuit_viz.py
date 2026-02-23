import os
import argparse
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.qfm.visualization.circuit_viz import draw_eha_circuit

def generate_circuits(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating Normal EHA Circuit (n_data=3, n_ancilla=0, L=3)")
    draw_eha_circuit(n_data=3, n_ancilla=0, n_layers=3, out_path=os.path.join(out_dir, "eha_circuit_Un.png"))
    
    print("Generating Over-Parameterized EHA Circuit (n_data=3, n_ancilla=1, L=3)")
    draw_eha_circuit(n_data=3, n_ancilla=1, n_layers=3, out_path=os.path.join(out_dir, "eha_circuit_Una.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    
    generate_circuits(out_dir=args.out)
