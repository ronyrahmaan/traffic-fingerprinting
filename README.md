# Traffic Fingerprinting: Reproduction and Extension

**Md A Rahman** · RA Skills Assessment · DSMI Lab, OU Polytechnic Institute

---

## Overview

This project does two things:

1. **Reproduces** the four classification algorithms from Kennedy et al. (IEEE CNS 2019) on their published Alexa voice command dataset
2. **Extends** the same pipeline to an original website traffic dataset I collected myself (50 sites, 100 visits each)

I also added three new feature sets beyond what the paper reports (CUMUL, timing, combined) and tested them with XGBoost and Random Forest. All code is written from scratch — the authors' GitHub repo was missing key files (`tools.py`, `utils.py`), so I re-implemented everything based on the algorithm descriptions in the papers they cited.

## Results

### Alexa Dataset (Reproducing Table I)

| Algorithm | Paper | Mine | Diff |
|-----------|-------|------|------|
| LL-Jaccard | 17.4% | 17.6% | +0.2pp |
| LL-NB | 33.8% | 34.1% | +0.3pp |
| VNG++ | 24.9% | 24.2% | −0.7pp |
| AdaBoost | 33.4% | 30.3% | −3.1pp |
| **Combined+RF (new)** | — | **39.2%** | — |

All four original algorithms match the paper within 0.2–3.1pp. My Combined+RF beats the paper's best result (LL-NB 33.8%) by +5.4pp and trains 160x faster than AdaBoost.

### Website Dataset (50 sites × 100 visits = 5,000 traces)

| Algorithm | Website | Alexa | Gap |
|-----------|---------|-------|-----|
| LL-Jaccard | **94.1%** | 17.6% | +76.5pp |
| Combined+RF | 92.7% | 39.2% | +53.5pp |
| Combined+XGB | 92.0% | 32.4% | +59.6pp |
| CUMUL+XGB | 91.4% | 30.8% | +60.6pp |
| AdaBoost | 89.7% | 30.3% | +59.4pp |
| Timing+XGB | 86.9% | 17.2% | +69.7pp |
| LL-NB | 84.6% | 34.1% | +50.5pp |
| VNG++ | 54.3% | 24.2% | +30.1pp |

Every algorithm does much better on website traffic. Websites load thousands of packets with distinct page structures, giving classifiers far more signal than short voice command exchanges.

## Project Structure

```
traffic-fingerprinting/
│
├── src/traffic_fingerprinting/        # Core library
│   ├── classifiers.py                 # All classifiers (LL-Jaccard, LL-NB, VNG++, AdaBoost, XGB, RF)
│   ├── features.py                    # Feature extraction from packet traces
│   ├── data_loader.py                 # Loaders for Alexa and website CSV formats
│   └── __init__.py
│
├── scripts/
│   ├── 1a_run_evaluation.py           # Basic Alexa reproduction (Table I)
│   ├── 1a_run_expanded_evaluation.py  # Full Alexa evaluation with improvements + figures
│   ├── 1a_save_results_and_figures.py # Save results JSON and generate plots
│   ├── 1a_generate_book_data.py       # Export data for report
│   ├── 1b_capture_website_traffic.py  # Automated traffic collection (Playwright + tcpdump)
│   ├── 1b_validate_website_dataset.py # Quality checks on collected data
│   └── 1b_run_website_evaluation.py   # Run all algorithms on website dataset
│
├── dashboard/                         # Interactive Streamlit dashboard
│   ├── app.py
│   └── requirements.txt
│
├── results/
│   ├── alexa_dataset/                 # Alexa reproduction results (JSON)
│   └── website_eval/                  # Website evaluation results (JSON)
│
├── figures/
│   ├── alexa/                         # Comparison plots, confusion matrices
│   └── website/                       # Per-site accuracy, cross-domain comparisons
│
├── .streamlit/config.toml             # Dashboard theme
├── pyproject.toml
└── README.md
```

## How to Run

```bash
cd traffic-fingerprinting
uv sync

# Reproduce Alexa results (Part 1a)
uv run python scripts/1a_run_expanded_evaluation.py

# Run website evaluation (Part 1b — requires data in data/collected/website_csv/)
uv run python scripts/1b_run_website_evaluation.py

# Launch the interactive dashboard
uv run streamlit run dashboard/app.py
```

To collect website traffic from scratch (requires root for tcpdump):

```bash
uv run playwright install chromium
sudo uv run python scripts/1b_capture_website_traffic.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| Python 3.12 | Runtime |
| uv | Package management |
| scikit-learn | Classification algorithms |
| XGBoost | Gradient boosting classifier |
| dpkt | PCAP packet parsing |
| Playwright | Browser automation for traffic collection |
| matplotlib | Static figures |
| streamlit | Interactive dashboard |
| plotly | Interactive charts |

## References

1. Kennedy, Thomas, and Wang. "I Can Hear Your Alexa: Voice Command Fingerprinting on Smart Home Speakers." IEEE CNS, 2019.
2. Panchenko et al. "Website Fingerprinting at Internet Scale." NDSS, 2016.
3. Sirinam et al. "Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning." ACM CCS, 2018.
