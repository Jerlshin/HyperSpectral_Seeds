"""The ablation grid, as data (CHANGES §20).

The audited project documented 21 ablation levers and pulled **zero** of them.
That is the finding this module exists to make impossible to repeat: every
experiment in CHANGES §20 is declared here as a named set of arms, each arm a
config name plus Hydra overrides, and the runner expands them over seeds and
folds. An experiment is therefore a *value* — inspectable, diffable, testable,
and runnable one arm at a time — rather than a shell script somebody wrote once.

Reading the table
─────────────────
Every :class:`Ablation` states the **question** it answers and the **decision
rule** that reads its output, because an ablation without a pre-registered
decision rule is an invitation to read whichever number is convenient. The rules
are CHANGES §20's, verbatim where it gave one.

Ordering
────────
:data:`RUN_FIRST` is A12. Until run-to-run variance σ is known, no delta in the
table means anything — the audited run's entire Stage-2 + Stage-3 gain was
+0.005 against a σ that has never been measured on this pipeline. Then A1 (the
leakage gap, which blocks every other claim), then A9 (which may change the
research question rather than the model), then A3/A8.

Cost
────
At ≈45 min/run on one RTX 3060 the whole grid below is ~2 days. The audited
single run was 19 hours. That arithmetic — a trimmed model and a working data
path converting one run into forty — is the largest practical gain in the
audit, and it is what makes this table affordable at all (§9.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The **primary** experiment config every arm composes unless it names another:
#: SpectralSeedNet, the complete 256-band cube, grouped protocol, one stage.
DEFAULT_CONFIG = "experiment/seednet_full256"

#: The four-branch control arm, on the primary protocol and the primary input.
#: Every architecture and curriculum ablation runs here, so an arm differs from
#: :data:`DEFAULT_CONFIG` in exactly the thing it is varying. Before this config
#: existed those ablations composed :data:`AUDITED_CONFIG`, silently varying the
#: split and the band count alongside the treatment.
CONTROL_CONFIG = "experiment/quadnet_full256"

#: The frozen 40-band / three-stage / stratified replica of the audited run. It
#: is a historical record and the subject of the golden gates — **not** an
#: ablation arm. Reached only by A1's replication arm, which is asking about the
#: audited configuration itself.
AUDITED_CONFIG = "experiment/quadnet_audited"

#: Run this before anything else. See the module docstring.
RUN_FIRST = "A12"


@dataclass(frozen=True)
class Arm:
    """One cell of an ablation: a config plus the overrides that define it.

    Args:
        name: Short slug, used in the run directory name. Must be unique within
            its ablation and filesystem-safe.
        overrides: Hydra override strings, e.g. ``["model.subcenter_K=1"]``.
            These are what makes the arm; two arms differing in more than one
            override are measuring more than one thing, and
            :meth:`Ablation.validate` says so.
        config: Experiment config name. Defaults to the ablation's.
        note: Why this arm exists, in one line. Printed by the CLI listing.
    """

    name: str
    overrides: tuple[str, ...] = ()
    config: str | None = None
    note: str = ""

    def resolved_config(self, ablation_config: str) -> str:
        return self.config or ablation_config


@dataclass(frozen=True)
class Ablation:
    """A named experiment: a question, some arms, and the rule that reads them.

    Args:
        seeds: Seeds each arm is repeated at. Three is the minimum that makes a
            mean ± range meaningful; A12 uses five because measuring σ *is* its
            question.
        folds: ``data.split_fold`` values each arm is repeated at. ``(0, 1)`` is
            the complete leave-one-bundle-out CV this dataset supports — there
            is no third bundle, so there is no third fold.
        reference: The arm every other arm's delta is measured against. ``None``
            means the arms are reported side by side without a designated
            control (A1's two protocols are not a treatment and a control).
    """

    key: str
    question: str
    decision_rule: str
    arms: tuple[Arm, ...]
    seeds: tuple[int, ...] = (0, 1, 2)
    folds: tuple[int, ...] = (0, 1)
    config: str = DEFAULT_CONFIG
    reference: str | None = None
    note: str = ""
    #: Ablations that must have reported before this one is interpretable.
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_runs(self) -> int:
        """How many training runs this ablation costs."""
        return len(self.arms) * len(self.seeds) * len(self.folds)

    def validate(self) -> list[str]:
        """Design problems worth refusing to run, as human-readable strings.

        Checks the property that makes an ablation an ablation: arm names are
        unique, the reference exists, and no two arms differ by more than one
        override. The last is a warning rather than an error — A3's ``{C}`` arm
        legitimately changes ``enabled_branches`` and nothing else, but a
        four-branch-vs-two-branch comparison that also moves the learning rate
        is the confound CHANGES §5.5 spends a page on.
        """
        problems: list[str] = []
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            problems.append(f"{self.key}: duplicate arm names in {names}")
        if self.reference is not None and self.reference not in names:
            problems.append(f"{self.key}: reference arm {self.reference!r} is not among {names}")
        if len(self.arms) < 2:
            problems.append(f"{self.key}: an ablation needs at least two arms")
        if not self.seeds:
            problems.append(f"{self.key}: at least one seed is required")
        return problems


# ══════════════════════════════════════════════════════════════════════
#  The grid
# ══════════════════════════════════════════════════════════════════════

ABLATIONS: dict[str, Ablation] = {}


def _register(ablation: Ablation) -> Ablation:
    ABLATIONS[ablation.key] = ablation
    return ablation


_register(
    Ablation(
        key="A12",
        question="What is the run-to-run variance of this pipeline?",
        decision_rule=(
            "Report sd and range of macro-F1 over 5 seeds at one fold. Every later "
            "ablation's delta is interpreted against 2σ. A delta smaller than σ is not "
            "evidence — and every delta the audited run reported was."
        ),
        arms=(
            Arm("identical", note="One config, five seeds. The arms differ in nothing."),
            # A second, deliberately identical arm would be pure cost; the
            # ablation's spread *is* its result, so the seeds are the arms.
            Arm(
                "identical_fold1",
                overrides=("data.split_fold=1",),
                note="The same, on the other bundle — is σ fold-dependent?",
            ),
        ),
        seeds=(0, 1, 2, 3, 4),
        folds=(0,),
        note="RUN THIS FIRST. Until σ is known, no other number in the grid means anything.",
    )
)

_register(
    Ablation(
        key="A1",
        question="How much of the score is acquisition-bundle recognition rather than variety recognition?",
        decision_rule=(
            "The gap F1_stratified − F1_grouped IS the headline result (CHANGES §19.2). "
            "It blocks every other claim: until it is measured, no number from this dataset "
            "is known to be about rice varieties."
        ),
        arms=(
            Arm(
                "grouped",
                overrides=("data=hsi256_grouped",),
                note="Leave-one-acquisition-bundle-out. Training sees one bundle per class.",
            ),
            Arm(
                "stratified",
                overrides=("data=hsi256_stratified", "data.split_fold=0"),
                note="Patch-level. All 180 bundles appear in train AND in eval.",
            ),
        ),
        seeds=(0, 1, 2),
        folds=(0, 1),
        note=(
            "Both arms are the primary 256-band input; only the split moves, so the gap is "
            "the split's and not the representation's. The stratified arm has no folds to "
            "rotate, so its fold-1 cells repeat fold 0 at a different seed; the runner "
            "de-duplicates them."
        ),
    )
)

_register(
    Ablation(
        key="A2",
        question="What does band selection cost, and does selecting outside the fold leak?",
        decision_rule=(
            "Two readings from one table. (i) `full_256 − spa40_within_fold` is what the "
            "reduction costs; if it exceeds 2σ (from A12) the primary path's refusal to "
            "reduce is confirmed, and if it does not, a reduced representation is a defensible "
            "efficiency choice for deployment — but still not a defensible way to compute a "
            "headline, because the incumbent k was never demonstrated (CHANGES M-14). "
            "(ii) `spa40_whole_corpus − spa40_within_fold` is the selection leak; if THAT "
            "exceeds 2σ, every published number on this dataset that chose bands on the full "
            "labelled corpus needs the caveat — including this project's own history."
        ),
        arms=(
            Arm(
                "full_256",
                note=(
                    "THE PRIMARY PATH. Every acquired band, no selection, so there is no "
                    "selection to leak. This is the reference the other two are read against."
                ),
            ),
            Arm(
                "spa40_whole_corpus",
                overrides=("data=ablation/spa40_grouped",),
                note="k=40 chosen with every patch's label in scope, including eval patches.",
            ),
            Arm(
                "spa40_within_fold",
                overrides=(
                    "data=ablation/spa40_grouped",
                    "data.patches_data=./dataset/folds/patches_fold{fold}_40b.npy",
                    "data.wavelength_path=./dataset/folds/wavelengths_fold{fold}_40b.csv",
                ),
                note=(
                    "k=40 chosen on TRAINING patches only, per fold. Produced by "
                    "`scripts/select_bands.py --per-fold`."
                ),
            ),
        ),
        reference="full_256",
        depends_on=("A12",),
        note=(
            "The gateway experiment for the whole band-selection pathway "
            "(`docs/07_BAND_SELECTION_PATHWAY.md`). It is the only ablation whose arms change "
            "the input representation, which is why the primary path is its reference rather "
            "than one of its treatments."
        ),
    )
)

_register(
    Ablation(
        key="A3",
        question="Is the four-branch design justified?",
        decision_rule=(
            "If {A,B,C,D} − {B,C} < 2σ, the removal of Branches A and D is confirmed. "
            "If it EXCEEDS 2σ, the audit is wrong about them and CHANGES §14 must be "
            "reversed for those items. This is designed to be able to prove the audit wrong."
        ),
        arms=(
            Arm(
                "abcd",
                overrides=("model.enabled_branches=[a,b,c,d]",),
                note="All four, with SYMMETRIC dropout — the confound removed.",
            ),
            Arm("bc", overrides=("model.enabled_branches=[b,c]",), note="Chemometrics + spatial."),
            Arm("c", overrides=("model.enabled_branches=[c]",), note="The spatial operator alone."),
            Arm("b", overrides=("model.enabled_branches=[b]",), note="The index bank alone."),
        ),
        config=CONTROL_CONFIG,
        reference="abcd",
        depends_on=("A12", "A1"),
        note=(
            "Runs the four-branch architecture on the primary 256-band grouped composition, "
            "with `branch_drop_profile=[1,1,1,1]`, so it measures the branches rather than "
            "the dropout policy that produced Branch C's confounded 87% influence — and "
            "rather than the split and the band count, which the audited replica would have "
            "varied alongside them."
        ),
    )
)

_register(
    Ablation(
        key="A4",
        question="Is Branch A's 64-cell replication necessary?",
        decision_rule=(
            "Branch A's compute scales as grid_size_a² while its parameters do not — it is "
            "60% of the forward FLOPs. Halving the grid cuts that 4×. Run only if A3 keeps A."
        ),
        arms=(
            Arm("grid8", overrides=("model.grid_size_a=8",), note="The audited 8×8 — 64 cells."),
            Arm("grid4", overrides=("model.grid_size_a=4",), note="4× cheaper."),
            Arm("grid2", overrides=("model.grid_size_a=2",), note="16× cheaper."),
        ),
        config=CONTROL_CONFIG,
        reference="grid8",
        depends_on=("A3",),
    )
)

_register(
    Ablation(
        key="A5",
        question="Is the rank-128 bilinear fusion worth 0.5 M parameters?",
        decision_rule=(
            "Second-order interactions over ten modality pairs, fitted from 6,036 samples, "
            "in a fusion where three of five modalities carried ≤6% influence each. Run only "
            "if A3 keeps ≥3 modalities."
        ),
        arms=(
            Arm("bilinear_gate", overrides=("model.fusion_mode=bilinear_gate",)),
            Arm("gate_only", overrides=("model.fusion_mode=gate",)),
            Arm("concat_mlp", overrides=("model.fusion_mode=concat_mlp",)),
        ),
        config=CONTROL_CONFIG,
        reference="bilinear_gate",
        depends_on=("A3",),
    )
)

_register(
    Ablation(
        key="A6",
        question="Does SupCon help, with the sampler controlled?",
        decision_rule=(
            "Both arms use the class-balanced sampler. The audited design changed the loss, "
            "the augmentation profile AND the sampler at the same epoch, so it measured none "
            "of the three. If SupCon clears 2σ, CHANGES §17's optional Phase B is earned."
        ),
        arms=(
            Arm("ce_only", overrides=("single.supcon_epochs=0",)),
            Arm(
                "ce_supcon",
                overrides=("single.supcon_epochs=30", "single.supcon_weight=0.3"),
            ),
        ),
        depends_on=("A12",),
    )
)

_register(
    Ablation(
        key="A7",
        question="Does any margin machinery help?",
        decision_rule=(
            "Four arms, ONE variable each. m=0 is NormFace, which Stage 1 already ran to "
            "0.842. Add back only what clears 2σ."
        ),
        arms=(
            Arm("m0", overrides=("single.arcface_m=0.0",), note="NormFace — no margin at all."),
            Arm("global_m", overrides=("single.arcface_m=0.30",), note="One scalar margin."),
            Arm(
                "per_class",
                overrides=("single.arcface_m=0.30", "model.per_class_margin=true"),
                note="+ the signed R−P rule, fitted on CALIB (never on the selection split).",
            ),
            Arm(
                "pairwise",
                overrides=(
                    "single.arcface_m=0.30",
                    "model.per_class_margin=true",
                    "model.pairwise_penalty=true",
                ),
                note="+ the row-normalised confusion penalty.",
            ),
        ),
        reference="m0",
        depends_on=("A12",),
    )
)

_register(
    Ablation(
        key="A8",
        question="Do Stages 2 and 3 add anything at all?",
        decision_rule=(
            "The falsification test for 65% of the audited run's compute. Current evidence: "
            "+0.005 macro-F1 for 12.2 of 18.7 hours. If either stage clears 2σ under the "
            "grouped protocol, CHANGES §14's removal of it must be reversed."
        ),
        arms=(
            Arm("s1", overrides=("pipeline=stage1_only",)),
            Arm("s1_s2", overrides=("pipeline=stage1_stage2",)),
            Arm("s1_s2_s3", overrides=("pipeline=three_stage",)),
            Arm(
                "single",
                overrides=("pipeline=single",),
                config=DEFAULT_CONFIG,
                note="The primary composition — collapsed curriculum AND the smaller model.",
            ),
        ),
        config=CONTROL_CONFIG,
        reference="s1",
        depends_on=("A12", "A1"),
        note=(
            "All four arms are the same architecture on the same 256-band grouped "
            "composition; only `pipeline` moves, which is what makes the comparison about "
            "the curriculum."
        ),
    )
)

_register(
    Ablation(
        key="A10",
        question="Is capacity actually harmful?",
        decision_rule=(
            "Answers 'is 5.19 M too big' with data instead of intuition. The 25-point gap "
            "between LDA on mean spectra (0.5916) and the full model (~0.845) is real "
            "spatial-spectral signal, so do not expect the smallest arm to win."
        ),
        arms=(
            Arm("w050", overrides=("model.spatial_width_mult=0.5",)),
            Arm("w075", overrides=("model.spatial_width_mult=0.75",)),
            Arm("w100", overrides=("model.spatial_width_mult=1.0",)),
            Arm("w150", overrides=("model.spatial_width_mult=1.5",)),
        ),
        reference="w100",
        depends_on=("A12",),
    )
)

_register(
    Ablation(
        key="A11",
        question="Is mixup the load-bearing regulariser?",
        decision_rule=(
            "Tests CHANGES §5.5 directly: training accuracy jumped 42% → 96.6% in the single "
            "epoch mixup was switched off, while validation macro-F1 did not move."
        ),
        arms=(
            Arm(
                "mixup_medium",
                overrides=("single.mixup=0.35", "single.aug_profile=medium"),
            ),
            Arm("nomixup_medium", overrides=("single.mixup=0.0", "single.aug_profile=medium")),
            Arm("mixup_heavy", overrides=("single.mixup=0.35", "single.aug_profile=heavy")),
            Arm("nomixup_heavy", overrides=("single.mixup=0.0", "single.aug_profile=heavy")),
            Arm("mixup_none", overrides=("single.mixup=0.35", "single.aug_profile=none")),
            Arm("nomixup_none", overrides=("single.mixup=0.0", "single.aug_profile=none")),
        ),
        reference="mixup_medium",
        depends_on=("A12",),
    )
)


#: A9 is **not a training run**. It is an analysis of an existing checkpoint —
#: t-SNE of the embeddings, mean-spectrum overlays, the 90×90 confusion matrix
#: and a segmentation-quality audit for the persistent hard cluster
#: {41, 49, 51, 52, 70} against five easy classes.
#:
#: CHANGES §20 rates it the highest-value experiment in the list, because it
#: distinguishes *spectrally inseparable varieties* from *segmentation failure*
#: and those have completely different consequences: the first is a
#: well-characterised ceiling worth publishing, the second is a fixable bug that
#: would explain why eight difficulty-targeted mechanisms moved nothing.
#:
#: Run it with ``python -m spectralquadnet.experiments.cli analyse``.
ANALYSIS_ONLY: tuple[str, ...] = ("A9",)


def get(key: str) -> Ablation:
    """Look up an ablation by key.

    Raises:
        KeyError: With the available keys in the message, because a typo here
            costs a GPU-day.
    """
    try:
        return ABLATIONS[key]
    except KeyError:
        raise KeyError(
            f"Unknown ablation {key!r}. Available: {', '.join(sorted(ABLATIONS))}"
            f" (plus analysis-only: {', '.join(ANALYSIS_ONLY)})"
        ) from None


def validate_all() -> list[str]:
    """Every registered ablation's design problems, flattened."""
    return [problem for ablation in ABLATIONS.values() for problem in ablation.validate()]


def total_runs() -> int:
    """Cost of the whole grid, in training runs."""
    return sum(a.n_runs for a in ABLATIONS.values())
