# Bi-Coupled Adaptive Network (BCAN)

Reference implementation of the main training pipeline for

> Trot after Market: A Bi-Coupled Adaptive Network for Stock Prediction via
> Relation-Adaptive Multimodal Alignment

submitted to *Big Data and Cognitive Computing* (manuscript BDCC-4645358).

This repository contains the proposed model only. The baseline models compared in
the manuscript are third-party implementations and are not redistributed here.

## Layout

```
US/   U.S. market pipeline (116 securities, 2010-2022, 9 rolling phases)
CN/   Chinese A-share pipeline (160 stocks, 2018-2025, 7 rolling phases)
```

The two directories are the same architecture under different asset naming. Each
contains the model, the training entry point, the backtest and the rolling-split
utility.

| File | Contents |
| --- | --- |
| `main_*.py` | Training / evaluation loop, PCDA pair construction, losses |
| `Model_*.py` | Full network: news branch, market branch, alignment, relation heads |
| `Layers_*.py` | News and market encoders, graph tensor, relation attention, GRUs |
| `backtesting.py` | Top-5 portfolio, IC/RankIC/NDCG@5, AR/SR/MDD, turnover |
| `data_preprocessing.py` | Feature construction, static graph, targets |
| `data_splits.py` | Rolling-window phase definition |
| `project_paths.py` | Asset resolution and the temporal contract |
| `run_*.py` | Convenience launcher scripts |
| `core_math_contract.json` | Function-level hashes of the numerical core |
| `VALIDATION_*.md` | What was checked locally and against the manuscript |

## Reference configuration

These are the defaults in the argument parser and are the settings reported in
Section 4 of the manuscript.

| Setting | Value |
| --- | --- |
| Input window | 5 trading days |
| Relation heads / head width | 8 / 512 |
| Consensus votes `Kc` | 6 |
| Retained neighbours `Kr` | 3 |
| Static-graph threshold `phi_s` | 0.5 |
| `lambda_align` | 1.0 |
| `lambda_pcda` | `1e-4`, zero for 5 epochs then linear ramp over epochs 6-10 |
| News GRU width | 768 |
| Market GRU hidden / output | 16 / 64 |
| Shared alignment width | 64 |
| Stock and relation GRU width | 256 |
| Dropout | 0.4 |
| Optimizer | Adam, lr `5e-4`, weight decay 0 |
| Gradient clipping | 0.25 |
| Max epochs / early-stopping patience | 300 / 30 |
| Gradient accumulation | 16 steps |
| Frozen-encoder microbatch | 32 |
| Checkpoint selection | minimum validation MAE, eligible from epoch 1 |
| Seed | 11380 |
| `lambda_align` / `lambda_pcda` above; `--static-mode level` |

Rolling windows: U.S. 3 years train / 1 year validation / 1 year test, 1-year step,
9 phases. China 36 months / 4 months / 8 months, 8-month step, 7 phases.

## Data

The processed panels are **not** included in this repository.

* **U.S. market.** Derived from the public FNSPID dataset
  (<https://github.com/Zdong104/FNSPID_Financial_News_Dataset>). Download it, then
  build the four panels this code expects (`*_price.csv`, `*_price_normalized.csv`,
  `*_target.csv`, `*_news.csv`) with the column layout documented in
  `data_preprocessing.py` and `project_paths.py`.
* **Chinese market.** Derived from CMIN-CN, Sina Finance and the Securities Times
  archive. These third-party sources are redistributed under their own terms, so the
  panels cannot be published here. Researchers with access to those sources can
  rebuild the panels with the same column layout.

A small synthetic sample that exercises the loader is provided in `data_sample/`.
It is not real market data and is only meant to let the pipeline start.

Text-input copyright: the individual news articles that form the text input remain
under the copyright of their publishers and are not redistributed.

## Running

```bash
pip install -r requirements.txt          # install PyTorch for your CUDA version first

cd US
python run_n2m_filtered_dailyrecurrent_us.py --phase 1 --device cuda
```

`--phase -1` runs every phase in turn. `--skip-backtest` stops after the predictive
metrics. `--static-mode return` switches the static graph from price levels to
training-period returns, which is the tested alternative reported in the manuscript.

## Outputs

Each phase writes its checkpoint, the per-phase prediction file and the metric and
backtest tables under `result/`. These are ignored by git.

## Licence

MIT. See `LICENSE`.

## Contact

Long Chen, Xi'an University of Posts and Telecommunications — chenlong@xupt.edu.cn
