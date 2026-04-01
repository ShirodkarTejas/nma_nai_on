# Micro-Publication Package

This package is a clean experiment path for the micro-publication.

## Why it exists

The older project code mixed together:

- reward redesign,
- environment complexity,
- training infrastructure,
- exploratory fixes,
- publication claims.

This package narrows the scope to the questions the paper needs to answer:

1. Does the controller adapt because of explicit substrate cues?
2. What changes when those cues are removed?
3. What changes when directional mechanics are introduced?

## Presets

- `baseline_short`
- `cue_ablation_short`
- `anisotropy_proxy_short`
- `anisotropy_full_short`

## Run examples

```bash
python3 micro_publication_main.py --experiment baseline_short
```

```bash
python3 micro_publication_main.py --experiment cue_ablation_short
```

```bash
python3 micro_publication_main.py --experiment anisotropy_proxy_short
```

```bash
python3 micro_publication_main.py --experiment anisotropy_full_short
```

```bash
python3 micro_publication_main.py --matrix_only
```

```bash
python3 micro_publication_main.py --run_matrix
```

## Notes

- Default runs are short by design: `20,000` training steps.
- Short presets save and run the expensive evaluation path only at the end of training.
- The full anisotropic mode uses a per-segment directional drag model in the rigid-link setting.
- This is intended as a clean experimental platform, not as a claim that the mechanics problem is fully solved.
- Per-run artifacts are isolated under `outputs/micro_publication/runs/<experiment_name>/`.
- Matrix manifests live under `outputs/micro_publication/`.
- Matrix comparisons live under `outputs/micro_publication/matrix/`.
