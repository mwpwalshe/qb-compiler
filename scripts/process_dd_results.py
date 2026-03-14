#!/usr/bin/env python3
"""Process results from DD hardware validation job d6qknvfr88ds73dch6r0."""
import json
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum_platform")
job = service.job("d6qknvfr88ds73dch6r0")
result = job.result()

N_SHOTS = 4096
pub_labels = [
    "GHZ-8_base", "GHZ-8_dd",
    "QAOA-6_base", "QAOA-6_dd",
    "QFT-6_base", "QFT-6_dd",
]

prepared = {
    "GHZ-8": {"seed": 0, "2q": 7, "dd_gates": 12,
              "physical_qubits": [124, 123, 136, 143, 144, 145, 146, 147]},
    "QAOA-6": {"seed": 8, "2q": 19, "dd_gates": 14,
               "physical_qubits": [142, 143, 136, 123, 124, 144]},
    "QFT-6": {"seed": 1, "2q": 46, "dd_gates": 36,
              "physical_qubits": [143, 136, 142, 144, 123, 124]},
}


def ghz_heavy_output_prob(counts, n):
    total = sum(counts.values())
    all_zeros = "0" * n
    all_ones = "1" * n
    heavy = counts.get(all_zeros, 0) + counts.get(all_ones, 0)
    return heavy / total


results_out = []
for i, label in enumerate(pub_labels):
    counts = result[i].data.c.get_counts()
    name = label.rsplit("_", 1)[0]
    variant = label.rsplit("_", 1)[1]

    if "GHZ" in name:
        n = int(name.split("-")[1])
        quality = ghz_heavy_output_prob(counts, n)
        metric_name = "heavy_output_prob"
    else:
        total = sum(counts.values())
        top_prob = max(counts.values()) / total
        quality = top_prob
        metric_name = "top_state_prob"

    top5 = dict(sorted(counts.items(), key=lambda x: -x[1])[:5])
    results_out.append({
        "circuit": name,
        "variant": variant,
        "shots": N_SHOTS,
        "quality": quality,
        "metric": metric_name,
        "top_counts": top5,
    })
    print(f"  {label}: {metric_name}={quality:.4f}  top={top5}")

# Summary table
print()
hdr = f"{'Circuit':>10} | {'Base':>10} | {'DD':>10} | {'Delta':>8} | {'%':>6} | {'Metric':>16}"
sep = f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*16}"
print("=" * len(hdr))
print(hdr)
print(sep)

for name in prepared:
    base_r = next(r for r in results_out if r["circuit"] == name and r["variant"] == "base")
    dd_r = next(r for r in results_out if r["circuit"] == name and r["variant"] == "dd")
    delta = dd_r["quality"] - base_r["quality"]
    sign = "+" if delta >= 0 else ""
    pct = delta / base_r["quality"] * 100 if base_r["quality"] > 0 else 0
    pct_sign = "+" if pct >= 0 else ""
    print(f"{name:>10} | {base_r['quality']:>10.4f} | {dd_r['quality']:>10.4f} | "
          f"{sign}{delta:.4f} | {pct_sign}{pct:.1f}% | {base_r['metric']:>16}")

print("=" * len(hdr))

# Save
out = Path("results/hardware_validation_dd.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump({
        "backend": "ibm_fez",
        "job_id": "d6qknvfr88ds73dch6r0",
        "n_shots": N_SHOTS,
        "n_seeds": 20,
        "results": results_out,
        "prepared": prepared,
    }, f, indent=2, default=str)
print(f"\nResults saved to {out}")
