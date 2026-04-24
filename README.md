# Dallas DCM Replication Package (Distance & Social Distance Base Paper)

Replication code and synthetic data for:

> Wang, Y. & Hipp, J. R. *Investigating how social and physical distance impact
> offender and victim mobility with Discrete Choice Modeling.* (Conditionally accept.)

This repository implements a GPU-accelerated discrete choice model (DCM) that
estimates how physical distance, racial dissimilarity, and income difference
between a person's residential block group and a potential target block group
shape:

- **offender mobility** — where offenders commit crimes, and
- **victim mobility** — where victimizations are observed,

across six crime types (burglary, motor vehicle theft, larceny, assault,
robbery, drug violations). The model is a conditional logit with the **full
Dallas choice set (959 census block groups, no sub-sampling)**, estimated via
maximum likelihood on a consumer-grade GPU.

![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![JAX](https://img.shields.io/badge/backend-JAX-orange.svg)

---

## What you can do with this repo

1. **Run the model** end-to-end on synthetic data that mirrors the analytic
   schema (no credentials, no large downloads, ≈1 minute on CPU).
2. **Inspect the method** — the JAX model, chunked MLE, and interaction
   functions live in `dcm/` and are ~300 LOC total.
3. **Re-run the base paper results** — swap the synthetic JSONL files for the
   real ones (instructions below) and re-run `scripts/base/run_base.sh`.
4. **Reuse the preprocessing pipeline** — the three-step pipeline under
   `scripts/base/preprocessing/` documents how raw Dallas Police Department
   records become analytic JSONL.

---

## Repository Layout

```
dallas_based_github/
├── dcm/                          # DCM engine (JAX, ~300 LOC)
│   ├── models.py                 # dcm_model, chunked-sum objective
│   ├── interactions.py           # distance, race, income interaction fns
│   ├── mle_utils.py              # SE (Hessian) and BIC
│   ├── protocols.py              # pydantic schemas + Config
│   └── tests.py                  # pytest suite (correctness + scale)
├── main.py                       # base pipeline entry point
├── scripts/base/
│   ├── config_base.yaml          # base model configuration
│   ├── run_base.sh               # all base + robustness variants
│   ├── run_base_sample.sh        # quick synthetic smoke test
│   ├── make_synthetic_base_data.py
│   ├── saver_base.py             # estimator JSON → CSV / LaTeX
│   ├── summarizer_base.py        # summary statistics tables
│   ├── plotter_base.py           # coefficient bar plots
│   ├── distance_decay_base.py    # distance-decay curves
│   ├── distance_robustness_check.py  # road-network robustness
│   ├── table_utils.py
│   └── preprocessing/            # raw data → JSONL pipeline
│       ├── geocode_addresses.py  # (optional) ArcGIS geocoding
│       ├── prepare_blocks.py     # census → blocks.jsonl
│       ├── prepare_agents.py     # geocoded CSVs → agent JSONL
│       ├── generate_sample_raw_data.py
│       └── mappers/              # race + crime category lookups
├── data/
│   ├── features/                 # synthetic JSONL inputs (shipped)
│   ├── estimators/base/          # populated by main.py
│   └── tables/base/              # populated by saver_base.py
├── setup.py                      # installable package
├── pyproject.toml
├── requirements.txt
├── run_base.sh                   # thin wrapper around scripts/base/run_base.sh
├── DATA_AVAILABILITY.md
├── REPLICATION_NOTES.md
├── CITATION.cff
└── LICENSE
```

---

## Quick Start (synthetic smoke test, ~1 minute)

```bash
git clone https://github.com/yqwang718/dallas_based_github.git
cd dallas_based_github

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

bash scripts/base/run_base_sample.sh
```

This regenerates the synthetic JSONL inputs, runs two small assault-only base
models (one for offenders, one for victims), and exports their estimator
tables.

Expected artifacts:

```
data/estimators/base/offenders_sample.json
data/estimators/base/victims_sample.json
data/tables/base/offenders_sample.csv
data/tables/base/victims_sample.csv
```

---

## Full Synthetic Replication

Run every base model variant from the manuscript (main + four robustness
specifications + post-COVID), all on the synthetic inputs:

```bash
bash scripts/base/run_base.sh
```

This produces one JSON file per (agent × specification) combination under
`data/estimators/base/`:

| File suffix              | Variant                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `offenders.json`         | main base model (L2-log distance, L1 race, \|Δincome\|)                 |
| `victims.json`           | main base model                                                         |
| `*_distance_l2.json`     | raw L2 distance in km                                                   |
| `*_race_bernasco.json`   | Bernasco (2009) aggregated racial dissimilarity                         |
| `*_race_disagg.json`     | Disaggregated (threshold) racial dissimilarity                          |
| `*_ses_dummy.json`       | Target-block log income only (no pairwise diff)                         |
| `*_post_covid.json`      | Re-estimation on post-COVID data (2020-03-24 to 2021-07-22)             |

Export them to publication-ready tables:

```bash
python scripts/base/saver_base.py data/estimators/base data/tables/base
python scripts/base/saver_base.py data/estimators/base data/tables/base --latex
```

---

## Swapping in the Real Analytic Data

The synthetic JSONL files under `data/features/` are intentional stand-ins.
To reproduce the published estimates, replace them with your copies of the
real analytic inputs while preserving the same file names and pydantic schemas
(see `dcm/protocols.py`):

```
data/features/
├── blocks.jsonl                  # BlockFeatures
├── offenders.jsonl               # AgentFeatures (2014-06-01 to 2020-03-23)
├── victims.jsonl                 # AgentFeatures
├── offenders_post_covid.jsonl
└── victims_post_covid.jsonl
```

Then re-run `bash scripts/base/run_base.sh`. No code changes are needed.

The original raw inputs are described in `DATA_AVAILABILITY.md`. Dallas Police
Department microdata is publicly available from the City of Dallas open data
portal; geocoded coordinates and derived JSONL features are not redistributed
here.

---

## Preprocessing (Raw Data → Analytic JSONL)

If you are starting from raw Dallas PD CSVs instead of pre-built JSONL,
`scripts/base/preprocessing/` has a three-step pipeline. See
`scripts/base/preprocessing/README.md` for the full schema, but the short
version is:

```bash
python scripts/base/preprocessing/generate_sample_raw_data.py \
    --output-dir data/raw_sample

python scripts/base/preprocessing/prepare_blocks.py \
    --census-gpkg data/raw_sample/census_merged_sample.gpkg \
    --output-jsonl data/raw_sample/blocks_test.jsonl

python scripts/base/preprocessing/prepare_agents.py \
    --offenders-csv data/raw_sample/offenders_geocoded_sample.csv \
    --incidents-csv data/raw_sample/incidents_geocoded_sample.csv \
    --census-gpkg   data/raw_sample/census_merged_sample.gpkg \
    --mappers-dir   scripts/base/preprocessing/mappers \
    --output-dir    data/raw_sample/features
```

The preprocessing scripts require the optional `geo` extras:

```bash
pip install -e .[geo]
```

Optional geocoding step (requires ArcGIS Online credentials):

```bash
pip install -e .[geocoding]
```

---

## Method at a Glance

For each individual $i$ (offender or victim) and candidate block group $j$,
the utility is

$$
U_{ij} = \beta_\text{distance} \log(D_{ij})
        + \beta_\text{race} R_{ij}
        + \beta_\text{income} I_{ij}
        + \sum_k \beta_{\text{control},k} C_{jk}
        + \varepsilon_{ij}
$$

with

- $D_{ij}$: Euclidean distance (m) from $i$'s home to block $j$'s centroid,
- $R_{ij}$: L1-norm aggregated racial dissimilarity between home and target
  block,
- $I_{ij}$: absolute difference in log median income,
- $C_{jk}$: eight block-level controls (population, employees, land size,
  household size, home-ownership %, underage %, attractions, transit stops).

The conditional logit probability uses the **full 959-block choice set**:

$$P_{ij} = \frac{\exp(U_{ij})}{\sum_m \exp(U_{im})}$$

Parameters are estimated by maximum likelihood with L-BFGS-B (SciPy) and JAX
auto-differentiation. A chunked-sum objective keeps GPU memory bounded even
at N ≈ 340k observations × K = 959 alternatives. SEs come from the inverse
observed information (Hessian); BIC comes from the converged loss.

One model per (agent-type × crime-type) is estimated independently — twelve
models total (six offender crimes, four victim crimes, plus two pooled
models). Victim models exclude burglary and drug violations since those offenses
do not yield a well-defined victim mobility signal.

---

## Testing

```bash
pip install -e .[dev]
pytest dcm/tests.py -v
```

The tests cover:

- chunked vs. vmapped MLE give the same loss and gradient (<1% tolerance),
- large-scale stability at 1M × 10K,
- end-to-end L-BFGS-B convergence at 100K × 5K,
- shape correctness of interaction primitives.

---

## Hardware Notes

The paper runs on an NVIDIA RTX 3070 (8 GB VRAM). For synthetic smoke tests
CPU is sufficient and takes ~1 minute. For the full dataset on CPU, expect
wall-clock times on the order of hours per model; on the RTX 3070, a few
minutes per model.

Tune `optimizer.chunk_size` in `scripts/base/config_base.yaml` if you hit
out-of-memory errors — smaller chunk sizes use less VRAM. Choose values
that are multiples of 64 (e.g. 256, 512, 1024, 2048, 4096, 8192, 16384)
to keep GPU kernels well-aligned; the default is `16384`.

---

## Citation

If you use this software or reproduce results, please cite both the software
and the paper. Citation metadata is in `CITATION.cff`.

```bibtex
@article{wang2026dallas,
  title   = {Investigating how social and physical distance impact offender and
             victim mobility with Discrete Choice Modeling},
  author  = {Wang, Yuqing and Hipp, John R.},
  journal = {Journal of Quantitative Criminology},
  year    = {2026},
  note    = {Conditionally accepted}
}
```

---

## License

This software is released under the
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)
— see [LICENSE](LICENSE).

- Permitted: use by individuals, educational institutions, public
  research organizations, and other noncommercial organizations for
  research, study, and other noncommercial purposes.
- Not permitted under this license: commercial use. For commercial
  licensing inquiries, please contact the authors (see
  [CITATION.cff](CITATION.cff)).

The copyright holders reserve the right to release this software, or
any future version of it, under different or additional license terms
(including commercial or dual-license terms) at any time.

---

## Contact

Issues and pull requests welcome. For questions about the method or the
analysis, contact Yuqing Wang (yqwang1@uci.edu).
