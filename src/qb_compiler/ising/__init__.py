"""qb-compiler → NVIDIA Ising-Decoder integration.

qb-compiler is the first Qiskit-side onramp to NVIDIA's
``Ising-Decoder-SurfaceCode-1`` model family (released 2026-04-14).
This package turns a rotated surface-code memory experiment — defined
either via a Qiskit circuit or a :class:`SurfaceCodePatchSpec` — into
the 4-channel input tensor the decoder consumes, and provides a
PyMatching baseline so users can directly compare decoder variants.

Quick start
-----------

::

    from qb_compiler.ising import (
        SurfaceCodePatchSpec,
        PyMatchingDecoder,
        evaluate_logical_error_rate,
    )

    spec = SurfaceCodePatchSpec(distance=5, rounds=5, basis="X", p_error=0.003)
    decoder = PyMatchingDecoder(spec)
    result = evaluate_logical_error_rate(spec, decoder, shots=10_000, seed=42)
    print(f"LER = {result.rate:.4e} ± {result.standard_error:.1e}")

Run the NVIDIA pre-decoder (optional — users supply their own gated-HF
weights and a ``build_model`` callable from the NVIDIA repo)::

    from qb_compiler.ising import IsingDecoderConfig, IsingDecoderWrapper
    config = IsingDecoderConfig(
        weights_path="Ising-Decoder-SurfaceCode-1-Fast.pt",
        device="cpu",
        build_model=my_build_model,
    )
    nvidia = IsingDecoderWrapper(spec, config)
    result = evaluate_logical_error_rate(spec, nvidia, shots=10_000)

Licensing
---------
qb-compiler's ising integration is released under Apache 2.0.
The NVIDIA Ising model weights are distributed by NVIDIA under the
NVIDIA Open Model License; qb-compiler does not redistribute them.
"""

from __future__ import annotations

from qb_compiler.ising.decoder import (
    IsingDecoderConfig,
    IsingDecoderWrapper,
    LogicalErrorRate,
    PyMatchingDecoder,
    SurfaceCodeDecoder,
    evaluate_logical_error_rate,
)
from qb_compiler.ising.patch_spec import SurfaceCodePatchSpec
from qb_compiler.ising.qiskit_bridge import qiskit_circuit_for, stim_circuit_for
from qb_compiler.ising.stim_adapter import (
    SurfaceCodeTensorLayout,
    build_ising_tensor,
    resolve_layout,
    sample_and_build_tensor,
)

__all__ = [
    "IsingDecoderConfig",
    "IsingDecoderWrapper",
    "LogicalErrorRate",
    "PyMatchingDecoder",
    "SurfaceCodeDecoder",
    "SurfaceCodePatchSpec",
    "SurfaceCodeTensorLayout",
    "build_ising_tensor",
    "evaluate_logical_error_rate",
    "qiskit_circuit_for",
    "resolve_layout",
    "sample_and_build_tensor",
    "stim_circuit_for",
]
