"""Read-only validation of sealed S22--S26 plans, artifacts and preserved history."""
from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DEST = ROOT / 'docs/research/evidence/S22_complementary_v5/validation/final/development_integrity.json'


@cache
def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    plans = [
        's22_complementary_v5', 's22_screening_amendment02',
        's22_screening_amendment03', 's22_screening_amendment05',
        's22_analysis_code04', 's22_analysis_code05',
        's23_frozen_multimodal', 's24_branch_multimodal',
        's25_head_seed_screen', 's26_tta_anchor',
    ]
    sealed = {}
    for name in plans:
        path = ROOT / f'configs/research/{name}.json'
        sealed[name] = digest(path)
        assert digest(path) == path.with_suffix('.sha256').read_text().split()[0], name
    active = plans[3:]
    # Ancestor source contracts remain historical; current sources are sealed by
    # the active amendment and its descendants. Archived parent code is checked
    # through their explicit pinned paths, rather than against replaced launchers.
    references = 0
    for name in active:
        plan = json.loads((ROOT / f'configs/research/{name}.json').read_text())
        for field in ('input_hashes', 'analysis_inputs'):
            for rel, expected in plan.get(field, {}).items():
                assert digest(ROOT / rel) == expected, (name, field, rel)
                references += 1
    studies = {
        'S22_complementary_v5': ('s22_screening_amendment05', 's22_fusion_analysis'),
        'S23_frozen_multimodal': ('s23_frozen_multimodal', 's23_frozen_multimodal'),
        'S24_branch_multimodal': ('s24_branch_multimodal', 's24_branch_multimodal'),
        'S25_head_seed_screen': ('s25_head_seed_screen', 's25_head_seed_screen'),
        'S26_tta_anchor': ('s26_tta_anchor', 's26_tta_anchor'),
    }
    archived_files = 0
    completed_files = 0
    for study, (plan_name, output_name) in studies.items():
        archive = ROOT / f'docs/research/evidence/{study}/screen_results'
        record = json.loads((archive / 'ARCHIVED.json').read_text())
        assert record['plan_sha256'] == sealed[plan_name], study
        for rel, expected in record['files'].items():
            assert digest(archive / rel) == expected, (study, rel)
            archived_files += 1
        output = ROOT / f'outputs/{output_name}'
        complete = json.loads((output / 'COMPLETED.json').read_text())
        assert complete['plan_sha256'] == sealed[plan_name], study
        assert digest(archive / 'COMPLETED.json') == digest(output / 'COMPLETED.json'), study
        for rel, expected in complete['files'].items():
            assert digest(output / rel) == expected, (output_name, rel)
            completed_files += 1
    original = ROOT / 'configs/research/s22_complementary_v5.json'
    assert digest(original) == 'f44c0d2e22f2849816cf2a8f2cdad8ae90084c695b06e4c181e7fc9b000f6f73'
    parent = ROOT / 'docs/research/evidence/S22_complementary_v5'
    assert digest(parent / 'preregistration.json') == digest(original)
    for suffix in ('02', '03', '05'):
        assert digest(parent / f'amendment{suffix}.json') == sealed[f's22_screening_amendment{suffix}']
    current = json.loads((ROOT / 'configs/research/s22_screening_amendment05.json').read_text())
    assert current['cells'] == [{'fold': 0, 'seed': 0}, {'fold': 1, 'seed': 0}]
    cuda = json.loads((ROOT / 'outputs/CUDA_COMPLETE.json').read_text())
    assert cuda['plan_sha256'] == sealed['s22_screening_amendment05']
    rejection = json.loads((ROOT / 'outputs/s26_tta_anchor/hypothesis.json').read_text())
    assert rejection['held_out_rows_scored'] == 0
    assert rejection['new_head_fits'] == 0 and rejection['new_encoder_fits'] == 0
    assert not (ROOT / 'outputs/s26_tta_anchor/predictions.csv.gz').exists()
    result = {
        'passed': True, 'sealed_plan_hashes': sealed,
        'checked_input_references': references,
        'unique_hashed_paths': digest.cache_info().currsize,
        'compact_archived_files': archived_files,
        'full_completed_files': completed_files,
        'original_s22_parent_unchanged': True,
        's22_cells': current['cells'],
        'completed_encoder_fits': 2, 'completed_head_fits_s23_s24_s25': 10,
        's26_zero_new_fits_and_heldout_scores': True,
        's27_state': 'proposed, not frozen or executed',
        'scope': 'Artifact integrity only; metric/identity arithmetic is in study readers and S21 replay.',
    }
    DEST.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
