# Dallas Based (Base Model) Scripts

This folder is the canonical home for the Dallas-based ("base") replication
entrypoints.

## Canonical Files

- `scripts/base/run_base.sh` - runs all base model variants on the synthetic inputs
- `scripts/base/run_base_sample.sh` - runs a quick synthetic smoke test
- `scripts/base/make_synthetic_base_data.py` - regenerates the synthetic JSONL inputs
- `scripts/base/config_base.yaml` - base model configuration template
- `scripts/base/saver_base.py` - entrypoint for estimator JSON -> table export
- `scripts/base/summarizer_base.py` - entrypoint for summary statistics tables
- `scripts/base/plotter_base.py` - entrypoint for coefficient plotting
- `scripts/base/distance_decay_base.py` - entrypoint for distance-decay plotting
- `scripts/base/distance_robustness_check.py` - road-network distance robustness script (requires the optional `geo` extras)

## Notes

- All utility scripts (`saver_base.py`, `summarizer_base.py`, `plotter_base.py`,
  `distance_decay_base.py`) are self-contained in this folder together with
  their shared `table_utils.py`.
- Synthetic feature files live in `data/features/` and can be regenerated with
  `python scripts/base/make_synthetic_base_data.py`.
