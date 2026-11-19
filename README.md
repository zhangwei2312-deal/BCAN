# Bi-Coupled Adaptive Network (BCAN)

Reference implementation of the main training pipeline for

> Trot after Market: A Bi-Coupled Adaptive Network for Stock Prediction via
> Relation-Adaptive Multimodal Alignment

submitted to *Big Data and Cognitive Computing* (manuscript BDCC-4645358).

This repository contains the proposed model only. The baseline models compared in
the manuscript are third-party implementations and are not redistributed here.

**What is provided:** the main model, the training entry point, the backtest, the
rolling-split utility, the reference configuration, the expected input-file layout
and a small synthetic sample that exercises the loader.

**What is not provided:** the real market panels, the news corpora and the
pretrained encoder weights. Those have to be prepared separately; see *Data* below.

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
| `run_*.py` | Launcher for the formal runs |
| `core_math_contract.json` | Function-level hashes of the numerical core |
| `VALIDATION_*.md` | What was checked locally and against the manuscript |

The `verify_*.py` and `smoke_*.py` scripts referenced in `VALIDATION_*.md` were
internal checks and are not part of this release.

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
| Headline embedding | masked average pooling over the frozen encoder's token states (Eq. 1) |
| Dropout | 0.4 |
| Optimizer | Adam, lr `5e-4`, weight decay 0 |
| Gradient clipping | 0.25 |
| Max epochs / early-stopping patience | 300 / 30 |
| Gradient accumulation | 16 steps |
| Frozen-encoder microbatch | 32 |
| Checkpoint selection | minimum validation MAE, eligible from epoch 1 |
| Seed | 11380 |
| Static-graph construction | `--static-mode level` (price levels) |

Rolling windows: U.S. 3 years train / 1 year validation / 1 year test, 1-year step,
9 phases. China 36 months / 4 months / 8 months, 8-month step, 7 phases.

## Install

Each market directory carries its own dependency list, so install from inside it:

```bash
cd US                       # or: cd CN
pip install -r requirements.txt
```

Install PyTorch for your CUDA version first; see the comment at the top of
`requirements.txt`.

## Data

The real panels are not included. The loader expects four files per market, with the
column layout used by `data_preprocessing.py` and `project_paths.py`:

```
*_price.csv              Ticker, Date, Open, High, Low, Close, Volume
*_price_normalized.csv   the same dates with a normalized close column
*_target.csv             Ticker, Date, target   (next-day close-to-close return)
*_news.csv               Ticker, Date, Headlines   (headlines separated by '|')
```

* **U.S. market.** Derived from the public FNSPID dataset
  (<https://github.com/Zdong104/FNSPID_Financial_News_Dataset>). Download it and
  build the four panels with the column layout above.
* **Chinese market.** Derived from CMIN-CN, Sina Finance and the Securities Times
  archive. These third-party sources are redistributed under their own terms, so the
  panels cannot be published here. Researchers with access to those sources can
  rebuild the panels with the same layout.
* **Encoder weights.** U.S. runs expect a local copy of
  `huawei-noah/TinyBERT_General_4L_312D` in `./Tinybert-4layers-en`; Chinese runs
  expect `TinyBertZh-4L` in `./Tinybert-4layers`. Both are loaded with
  `AutoTokenizer` / `AutoModel` and stay frozen.

### Encoder checksums

The weights were not redistributed with this repository, so the checkpoints used
for the reported experiments are the local copies listed below. Their architecture
corresponds to the upstream releases named above, and the parameter tensors are
unchanged; the files are local serializations, which is why their hashes differ
from the hashes of the upstream files. The values below are SHA-256.

| Market | File                                    | Bytes      | SHA-256                                                            |
| ------ | --------------------------------------- | ---------- | ------------------------------------------------------------------ |
| U.S.   | `Tinybert-4layers-en/config.json`       | 409        | `9435ceadf314b412f627e32f8e27cb8db4996b6bab41e4d233da178cb39132b9` |
| U.S.   | `Tinybert-4layers-en/pytorch_model.bin` | 62,747,391 | `84ac219f2fdab6e7f54fc6db4d7a9493708990b006d58f50cc1dceeef2a12f8a` |
| U.S.   | `Tinybert-4layers-en/vocab.txt`         | 231,508    | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |
| China  | `Tinybert-4layers/config.json`          | 714        | `9d2a77948bef59f07c24cc5bf394f75fe45654d7c7bdc6070eb4aebcc3a9c74b` |
| China  | `Tinybert-4layers/model.safetensors`    | 45,689,248 | `c6892be327df420c644a4e30e93d2b82fd1f00a9a6f6bd8c33945f7ac530f1ff` |
| China  | `Tinybert-4layers/pytorch_model.bin`    | 45,702,349 | `45fdeb288dfa504ba5039757e1fdbca8145cbc33eeb5ab3e00bc7f2eb0938a04` |
| China  | `Tinybert-4layers/vocab.txt`            | 109,534    | `366fe2b1318c774fd28bdea58dbcfbd6d6aeae313fdcd9063d170cdc602fd51a` |

For the Chinese checkpoint both a `.bin` and a `.safetensors` copy are present.
`AutoModel.from_pretrained` loads the safetensors file when both exist, and the two
copies differ from each other by 13,101 bytes, which is the `position_ids` buffer
that the safetensors writer stores explicitly. The parameter tensors are identical
in the two files.

Verify locally with:

```bash
sha256sum Tinybert-4layers-en/config.json Tinybert-4layers-en/pytorch_model.bin Tinybert-4layers-en/vocab.txt
sha256sum Tinybert-4layers/*.safetensors Tinybert-4layers/*.bin Tinybert-4layers/config.json Tinybert-4layers/vocab.txt
```

A synthetic sample in `data_sample/` follows the same layout and lets the loader
start. It is not real market data.

Text-input copyright: the individual news articles that form the text input remain
under the copyright of their publishers and are not redistributed.

## Run

From inside a market directory:

```bash
cd US
python run_n2m_filtered_dailyrecurrent_us.py --phases 1
python run_n2m_filtered_dailyrecurrent_us.py --phases 1,2,3,4,5,6,7,8,9
```

`--phases` is required and takes a comma-separated list of phase IDs (US 1-9,
China 1-7). The launcher accepts `--device`, `--seed`, `--max-epoch`, `--wait-epoch`,
`--lambda-align`, `--bert-batch-size`, `--batch-train`, `--bert-path` and
`--data-dir`. Options handled inside `main_*.py`, such as `--skip-backtest` and
`--static-mode return`, must be passed to that script rather than to the launcher.

## Outputs

Each phase writes its checkpoint, the per-phase prediction file and the metric and
backtest tables under `result/`. These are ignored by git.

## Licence

MIT. See `LICENSE`.

## Contact

Long Chen, Xi'an University of Posts and Telecommunications — chenlong@xupt.edu.cn
