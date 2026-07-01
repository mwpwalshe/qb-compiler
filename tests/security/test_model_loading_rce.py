"""Regression tests for arbitrary-code-execution via model checkpoint loading.

Finding (CRITICAL): ``IsingDecoderWrapper._load_checkpoint`` previously
called ``torch.load(path, weights_only=False)`` on a user/third-party
``.pt`` file (NVIDIA Ising pre-decoder, downloaded from HuggingFace).
``torch.load`` with ``weights_only=False`` unpickles arbitrary Python
objects, so a malicious checkpoint executes attacker code at load time
(pickle ``__reduce__`` gadget).  The fix loads with
``weights_only=True``, which restricts unpickling to tensors and safe
primitives and rejects code-bearing checkpoints.

These tests build a genuinely malicious checkpoint with a ``__reduce__``
side effect and assert the side effect never fires through the decoder's
loader, while a legitimate tensor state-dict still loads (no regression).
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pymatching")
pytest.importorskip("stim")

from qb_compiler.ising.decoder import (  # noqa: E402
    IsingDecoderConfig,
    IsingDecoderWrapper,
)
from qb_compiler.ising.patch_spec import SurfaceCodePatchSpec  # noqa: E402


# Module-level marker file path filled in per-test; the exploit payload
# writes to whatever path it is constructed with.
class _Exploit:
    """Pickle gadget: writes a marker file when unpickled.

    A real attacker would run ``os.system`` / spawn a reverse shell here;
    writing a sentinel file is an equivalent, test-safe proof of code exec.
    """

    def __init__(self, marker_path: str) -> None:
        self.marker_path = marker_path

    def __reduce__(self):  # type: ignore[no-untyped-def]
        # On unpickle, call os.write-style side effect.
        return (_write_marker, (self.marker_path,))


def _write_marker(path: str) -> str:
    with open(path, "w") as fh:
        fh.write("pwned")
    return path


def _tiny_model_builder():  # type: ignore[no-untyped-def]
    """A trivial nn.Module so the wrapper can be constructed without torch deps."""
    import torch.nn as nn

    def build(_spec) -> nn.Module:  # type: ignore[no-untyped-def]
        return nn.Linear(2, 2)

    return build


def _spec() -> SurfaceCodePatchSpec:
    return SurfaceCodePatchSpec(distance=3, rounds=1, basis="Z", p_error=0.003)


def test_malicious_checkpoint_does_not_execute_code(tmp_path):
    """A pickle-bomb .pt checkpoint must NOT run code through the loader."""
    marker = tmp_path / "pwned.txt"
    payload_path = tmp_path / "evil.pt"

    # Sanity: the gadget really does execute on an unsafe load.  We prove
    # the vuln is real by saving it and loading with weights_only=False in
    # a throwaway location, then assert our loader refuses it.
    torch.save(_Exploit(str(marker)), str(payload_path))

    assert not marker.exists()

    cfg = IsingDecoderConfig(
        weights_path=str(payload_path),
        device="cpu",
        build_model=_tiny_model_builder(),
    )

    # The fixed loader uses weights_only=True and raises ValueError instead
    # of unpickling the gadget.  Crucially, the marker file must NOT appear.
    with pytest.raises(ValueError, match=r"weights_only=True|security"):
        IsingDecoderWrapper(_spec(), cfg)

    assert not marker.exists(), (
        "SECURITY REGRESSION: malicious checkpoint executed code during load"
    )


def test_unsafe_load_would_have_executed_the_gadget(tmp_path):
    """Document the vuln: weights_only=False DOES execute the gadget.

    This is the control that proves the payload is a real exploit, not a
    no-op.  We never use weights_only=False in src/; this only runs the
    raw torch primitive to validate the test's premise.
    """
    marker = tmp_path / "control.txt"
    payload_path = tmp_path / "evil_control.pt"
    torch.save(_Exploit(str(marker)), str(payload_path))

    assert not marker.exists()
    # Unsafe load executes the gadget (this is exactly what the fix avoids).
    torch.load(str(payload_path), weights_only=False)
    assert marker.exists(), "test premise broken: gadget did not fire on unsafe load"

    # And the safe load refuses it without executing.
    safe_marker = tmp_path / "safe.txt"
    safe_payload = tmp_path / "evil_safe.pt"
    torch.save(_Exploit(str(safe_marker)), str(safe_payload))
    with pytest.raises(Exception):  # noqa: B017 - UnpicklingError/RuntimeError
        torch.load(str(safe_payload), weights_only=True)
    assert not safe_marker.exists()


def test_legitimate_state_dict_still_loads(tmp_path):
    """No regression: a benign tensor state_dict loads fine under the fix."""
    import torch.nn as nn

    model = nn.Linear(2, 2)
    ckpt = tmp_path / "good.pt"
    torch.save(model.state_dict(), str(ckpt))

    cfg = IsingDecoderConfig(
        weights_path=str(ckpt),
        device="cpu",
        build_model=_tiny_model_builder(),
    )
    # Should construct without raising and without any code execution.
    wrapper = IsingDecoderWrapper(_spec(), cfg)
    assert wrapper.spec.distance == 3
    # Provenance hash recorded from the checkpoint file.
    assert wrapper._decoder_version.startswith("sha256:")


def test_wrapped_state_dict_key_still_loads(tmp_path):
    """A {'model_state_dict': ...} wrapper checkpoint also loads safely."""
    import torch.nn as nn

    model = nn.Linear(2, 2)
    ckpt = tmp_path / "wrapped.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 7}, str(ckpt))

    cfg = IsingDecoderConfig(
        weights_path=str(ckpt),
        device="cpu",
        build_model=_tiny_model_builder(),
    )
    wrapper = IsingDecoderWrapper(_spec(), cfg)
    assert wrapper.spec.rounds == 1


def test_environment_not_polluted_after_tests():
    """Defensive: ensure no env var was set by a stray gadget."""
    assert os.environ.get("QBC_PWNED") is None
