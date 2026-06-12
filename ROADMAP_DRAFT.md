# qb-compiler roadmap draft (working file, not committed: review, edit voice, then commit)

## v0.5.3 (patch, asap)
- scrub stale marketing claims from integrations/qubitboost.py (done in working tree, review diff)
- add py.typed marker (done in working tree, pyproject already claims Typing :: Typed)
- fix ibm_fez basis gates cx -> ecr in BACKEND_CONFIGS
- merge dependabot #7 (qiskit 2.3.1) + extend ci matrix to qiskit 1.4/2.3
- timestamp VENDOR_PRICING, warn when stale

## v0.6 "trust the number"
- error-budget breakdown in ViabilityResult + cli (per-source: 2q / readout / idle / routing)
- uncertainty bands on fidelity estimates + published predicted-vs-measured validation table
- calibration freshness warnings + drift signal surfaced in qbc preflight
- implement issue #22 telemetry surface (IsingDecodeResult) per docs/design/v050_telemetry_surface.md

## v0.7
- backend auto-discovery via QiskitRuntimeService (rank the user's actual accessible backends)
- PUB-aware preflight (EstimatorV2/SamplerV2)
- shot-budget estimator (shots to +-epsilon)
- cross-provider cost quotes (ibm session model + braket per-task), staleness-stamped

## v0.8
- qec experiment preflight (memory-experiment viability: projected detector fraction, ler band, shots-for-confidence)
- ising bridge turnkey: build_model shim guidance, orientation auto-alignment, relaybp adapter

## standing
- same-week qiskit-major compatibility
- weekly dependabot merges
- consolidate dual ir / dual fidelity estimator
