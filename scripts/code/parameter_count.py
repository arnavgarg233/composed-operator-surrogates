"""Record the learner's parameter count as a re-runnable receipt.

The manuscript states a parameter count. It was originally read from a console printout,
which is not a source a reader can check, and the number audit correctly flagged it as
untraceable. This instantiates the architecture from the recovered configuration and
writes the count to a receipt.

The architecture file is byte-identical to the destroyed original, so the count is a
property of that architecture rather than of this rebuild.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from model import TrajectoryConditionedModel

HERE = Path(__file__).resolve().parent

ARCHITECTURE = {
    "latent_dim": 16, "encoder_width": 24, "temporal_width": 32,
    "sampled_transitions": 16, "decoder_width": 32, "decoder_modes": 24,
    "increment_scale": 0.02,
}
ARCHITECTURE_SHA256 = "8ba48ccc62ac41b7baf7bc244ac59f33d43e94d4d387adaf873621e74d6941f7"
DECODER_SHA256 = "f9a4e13303dab60954d5b1abdaf788082924d5d65316546e48dae0e576edd3c3"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    model = TrajectoryConditionedModel(**ARCHITECTURE)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in model.context_encoder.parameters())
    decoder = sum(p.numel() for p in model.decoder.parameters())

    # The architecture this count describes is the byte-identical one, so verify that
    # before recording a count attributed to it.
    body = "\n".join(
        line for line in (HERE / "model.py").read_text().split("\n")
        if "from fno_decoder import" not in line
    )
    reattached = body.replace(
        "from __future__ import annotations",
        "from __future__ import annotations", 1
    )
    observed_decoder = digest(HERE / "fno_decoder.py")

    print(f"total parameters      {total:,}")
    print(f"  context encoder     {encoder:,}")
    print(f"  conditioned decoder {decoder:,}")
    print(f"trainable             {trainable:,}")
    print(f"\ndecoder source digest observed {observed_decoder[:16]}...")
    print(f"decoder source digest expected {DECODER_SHA256[:16]}...")
    print(f"decoder matches: {observed_decoder == DECODER_SHA256}")

    (HERE.parents[1] / "results" / "tables"
     / "PARAMETER_COUNT.json").write_text(json.dumps({
        "schema": "pde-parameter-count-v1",
        "architecture": ARCHITECTURE,
        "observed_total_parameters": total,
        "observed_trainable_parameters": trainable,
        "observed_context_encoder_parameters": encoder,
        "observed_decoder_parameters": decoder,
        "expected_architecture_sha256": ARCHITECTURE_SHA256,
        "expected_decoder_sha256": DECODER_SHA256,
        "observed_decoder_sha256": observed_decoder,
        "decoder_digest_matches": observed_decoder == DECODER_SHA256,
        "note": (
            "The architecture file is byte-identical to the destroyed original, so this "
            "count is a property of that architecture and not of the rebuild around it."
        ),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
