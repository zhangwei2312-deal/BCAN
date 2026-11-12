# Final US Daily-Recurrent audit

Date: 2026-08-31. Only this existing project was edited. No parallel project,
formal training run or model redesign was created.

Relation sensitivity analysis was removed from the final project because it is not part of the final experimental protocol.

## Modified files

| File | Change |
| --- | --- |
| backtesting.py | Remove the independent analysis pass, its three top-level helpers and nested correlation helper, the raw-close-only argument/loading path, related fields and prints. Preserve all three explicit-memory passes and prediction/portfolio calculations. |
| main_n2m_filtered_dailyrecurrent.py | Remove the corresponding argument chain, raw-close loading, result fields, console section and summary rows/column. Preserve the memory summary. |
| Layers_n2m_filtered_dailyrecurrent.py | Remove two assignments to an analysis-only runtime attribute. No Parameter or calculation change. |
| Model_n2m_filtered_dailyrecurrent.py | Remove that attribute's reset and activation-recomputation save/restore. Retain actual previous-day relation and recurrent hidden handling. |
| verify_package.py | Check the remaining backtest API and exactly three intervention inference calls; retain metric/output/date/PCDA checks. |
| verify_dailyrecurrent.py | Check actual relation outputs and propagation directly, including zero diagonal, shape and previous-day Relative-Static. Retain all three intervention and recomputation tests. |
| smoke_dailyrecurrent.py | Remove the extra analysis pass and its imports/outputs. Default to five test days, without changing model sizes, batch/TBPTT or training hyperparameters. |
| core_math_contract.json | Remove the deleted helper fingerprint and update only the constructor fingerprint affected by removal of a non-Parameter attribute. |
| README_DailyRecurrent.md | Document only retained experiments and the current short verification workflow. |
| VALIDATION_DailyRecurrent.md | This final audit. |

No files in the sibling Fixed_US project were edited. Its source hashes were
checked before and after. Dataset and local English TinyBERT assets are unchanged.

## Preserved core

- 116 stocks, nine U.S. rolling phases, strict t -> t+1.
- English TinyBERT: four layers, hidden312, vocabulary30522, pretrained/frozen/eval.
- News cleaning, deduplication, maximum30, packed headline GRU and no-news handling.
- OHLCV encoder, News-to-Market Self+Static base alignment.
- Eight-head, consensus6, mutual cap3, filtered bidirectional PCDA with the same weights and warmup/ramp.
- Relation bootstrap5, then one new day with persistent hidden; separate rolling5 stock-state branch.
- Dynamic, Relative-Static, bidirectional fusion, propagation, concat, residual and Linear output.
- Same three-term objective, optimizer, TBPTT, seeds and checkpoint selection.
- Same formal runner and required phase selection.

## Exact before/after numerical comparison

Both complete model/layer ASTs match after excluding only the removed attribute
assignments. The retained protected definitions and unchanged utility/data files
also passed fingerprint verification.

A fixed CPU fixture used seed11380, three stocks, all eight heads and unchanged
full hidden dimensions. Actual English TinyBERT was loaded. Comparison used
torch.equal with zero tolerance: 620 tensors and 53,088,743 tensor elements matched exactly.

Compared initialized state and all 185 parameter definitions; all six forward
outputs; MSE, base loss, bidirectional current-day PCDA loss and combined loss;
every parameter gradient including absent gradients; daily recurrent hidden;
per-head previous relations; post-backward/post-detach state; and RNG.

Full, No Explicit Relation Memory and Shuffled Explicit Relation Memory were each
tested in the numerical fixture, including the actual chronological inference
function. For the PCDA loss fixture, fixed nonzero pair weights exercised both
directions; the real-data smoke additionally exercised the actual consensus gate.

This proves equivalence for the controlled cleanup regression, not cross-device
training reproducibility or predictive performance.

## Verification

| Check | Result |
| --- | --- |
| compileall and runtime imports | PASS |
| verify_package.py | PASS |
| verify_fixed_implementation.py | PASS |
| verify_dailyrecurrent.py | PASS |
| CPU full-size one-window forward/backward | PASS |
| Exact cleanup regression | PASS: torch.equal, zero tolerance |
| Real P1 CPU short training, backward, optimizer and TBPTT | PASS |
| Current-day PCDA and chronological validation | PASS |
| Full / No Explicit / Shuffled Explicit inference and backtests | PASS |
| Canonical prediction and metric files | PASS: 580 stock-date rows, nine unchanged prediction columns |
| Formal main memory summary | PASS: three phase rows and three average rows; only retained fields |
| Temporary checkpoint save/load | PASS |
| Strict P9 dates | PASS: final prediction date 2022-12-29, count250 |
| AST import-usage audit | PASS for scoped removal; unrelated existing imports retained |

The tests directly checked normal fusion diagonals and propagation values/shapes,
rather than retaining a runtime matrix for an independent analysis.

The CPU short run used real P1 data with 116 stocks and eight heads, two 17-window
training passes at schedule epochs0 and5, batch/TBPTT16, five validation windows
per pass and five chronological prediction dates in each intervention mode.
No formal epoch sweep or full P1 training was launched.

## Output contract

Canonical prediction.csv and metrics.csv are retained, as are
predictions_phase1.csv, prediction_records_phase1.csv,
precision_at_5_daily_phase1.csv and the explicit-memory analysis summary.
Prediction columns and phase replacement behavior remain unchanged.

All prediction and portfolio metrics remain available:
MSE, RMSE, MAE, IC, RankIC, NDCG@5, Precision@5, MRR, Sharpe,
annualized return, total return, maximum drawdown and final value.
All three memory metric groups and Full-minus-intervention drops remain.
Precision@5 is still computed before the positive-prediction trading filter.

Only the removed experiment's result columns are absent from metrics/summary.
There is no extra inference pass for that experiment and no replacement runtime
attribute. Normal fusion validation observes actual calculation outputs in tests.

## Import and artifact audit

AST import usage was checked. The changed backtest and smoke have no unused named
imports. Existing unrelated imports were not refactored. scipy.stats.spearmanr
remains because canonical RankIC and backtest RankIC use it.

Source, README and configuration were searched for the deleted API names and
output terms: no remaining occurrences, except the single historical sentence in
this report. Temporary smoke outputs, the numerical reference and generated
Python caches were removed after inspection. Output directory creation and
checkpoint saving capabilities are intact.

## Formal P1 command (not executed)

From this project, with the intended Python environment activated:

```powershell
python run_n2m_filtered_dailyrecurrent_us.py --phases "1" --device cuda:0 --seed 11380 --lambda-align 0.1 --wait-epoch 20 --max-epoch 300 --exp-name "us_dailyrecurrent_phase1_lambda01_seed11380"
```

The command selects base coefficient0.1 and patience20; PCDA remains enabled
with coefficient1e-4. Formal GPU capacity was not tested by this CPU acceptance run.
