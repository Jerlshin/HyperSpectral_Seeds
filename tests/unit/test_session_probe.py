"""D26 — the training-rows session κ (S12 F69), as the final report computes it.

The probe is class-disjoint: session is predicted for classes the classifier
never saw. A representation that encodes session *in a way shared across
classes* scores high; one that encodes only class identity (which, under
``grouped``, determines the session inside the training set) scores ≈ 0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.models.registry import build_model
from spectralquadnet.reporting import session_probe

N_CLASSES, PER_CLASS, N_SESSIONS = 30, 20, 3


def _layout() -> tuple[np.ndarray, np.ndarray]:
    classes = np.repeat(np.arange(N_CLASSES), PER_CLASS)
    sessions = classes % N_SESSIONS  # one session per class, as under `grouped`
    return classes, sessions


def test_a_shared_session_direction_is_decoded_across_classes() -> None:
    rng = np.random.default_rng(0)
    classes, sessions = _layout()
    x = rng.normal(size=(classes.size, 6))
    x[:, 0] += 4.0 * sessions  # the same offset for every class of a session
    out = session_probe.session_kappa(x, sessions, classes)
    assert out["kappa"] > 0.9 and out["n_sessions"] == N_SESSIONS and out["dim"] == 6


def test_class_identity_alone_is_not_session_information() -> None:
    rng = np.random.default_rng(1)
    classes, sessions = _layout()
    centres = rng.normal(size=(N_CLASSES, 6)) * 3.0  # class clusters, no session structure
    x = centres[classes] + rng.normal(size=(classes.size, 6))
    out = session_probe.session_kappa(x, sessions, classes)
    assert abs(out["kappa"]) < 0.2


def test_an_undefined_probe_says_why() -> None:
    classes, _ = _layout()
    out = session_probe.session_kappa(np.ones((classes.size, 2)), np.zeros(classes.size), classes)
    assert out["kappa"] is None and "two sessions" in out["reason"]
    few = np.repeat(np.arange(3), 10)
    out = session_probe.session_kappa(np.ones((30, 2)), few % 2, few)
    assert out["kappa"] is None and "classes" in out["reason"]


class _Rows(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """``RiceSeedDataset``'s interface, as the extractor reads it."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y
        self.indices = np.arange(len(x))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):  # type: ignore[no-untyped-def]
        return self.x[i], self.y[i]


def test_the_extractor_returns_the_embedding_and_every_live_pathway() -> None:
    for pathways, expected in (
        ("[spatial,spectral]", {"embedding", "spatial", "spectral"}),
        ("[spectral]", {"embedding", "spectral"}),
    ):
        cfg = load_experiment_config(
            overrides=["data=ablation/u430k32_grouped", f"model.pathways={pathways}",
                       "model.stem_channels=16", "model.spatial_width_mult=0.25"]
        )
        torch.manual_seed(0)
        model = build_model(cfg, torch.linspace(0.0, 1.0, 32))
        model.train()  # the extractor must switch to eval and back
        x = torch.rand(6, 32, 64, 64) + 0.1
        loader = DataLoader(_Rows(x, torch.arange(6)), batch_size=4)
        reps = session_probe.extract_representations(model, loader, torch.device("cpu"))
        assert set(reps) == expected and model.training
        assert reps["embedding"].shape == (6, 256)
        assert np.allclose(np.linalg.norm(reps["embedding"], axis=1), 1.0, atol=1e-5)
        model.eval()
        with torch.no_grad():
            _, emb = model(x, return_embed=True)
        assert np.allclose(reps["embedding"], emb.numpy(), atol=1e-6)
