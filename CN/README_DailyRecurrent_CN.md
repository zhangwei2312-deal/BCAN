# CN Daily-Recurrent BCAN/DRFN

This is an independent A-share project. Input data and Chinese TinyBERT were
copied without modification. It does not load assets or checkpoints from any
sibling project. Use newly trained Daily-Recurrent checkpoints for formal results.

## Market and Inputs

- 160 stocks, 1,942 market dates, 2018-01-02 through 2025-12-31.
- Seven rolling phases: 36 training months, 4 validation months, 8 test months,
  sliding forward 8 months each phase.
- `cn_price.csv`, `cn_price_normalized.csv`, `cn_target.csv`, `cn_news.csv`.
- Normalized OHLCV uses daily cross-sectional sample z-scores, `ddof=1`.
- Target at date t is `Close[t+1] / Close[t] - 1` on the next global trading date.
- Splits use market dates. A supervised prediction is eligible only when its
  next global trading date belongs to the same split. P7 has 164 test prediction
  days, ending 2025-12-30 with target close on 2025-12-31.
- Static relations use training-period `Close_norm` Pearson correlation > 0.5.
- Chinese TinyBERT: `./Tinybert-4layers`, 4 layers, hidden 312, vocabulary 21128.
  Pretrained weights remain unchanged, frozen, and in evaluation mode.

`--data-dir` selects input CSVs only. `--bert-path` selects the explicit local
Chinese model. Both default to this project; there is no implicit fallback.
Checkpoints, results and logs always go inside this project under the experiment name.

## Daily Computation

1. The rolling five-day news and market encoders produce contextual representations.
   Headlines undergo exact deduplication, HTML unescape and whitespace cleanup,
   with at most 30 per stock/day. The headline GRU uses actual packed lengths.
   A no-news day outputs zero while preserving hidden state for a later news day
   inside that five-day encoding context.
2. Base alignment uses valid news anchors against market candidates, with self
   and static positives. Dynamic positives are handled separately by PCDA.
3. Detached previous-day pure Dynamic matrices, one per head, select today's
   PCDA pairs. Both modality directions filter their news endpoint independently.
4. Relation GRU bootstrap reads five days once. Each next window passes only its
   new terminal contextual vector and the carried hidden: `h_t = GRU_r(X_t, h_{t-1})`.
   Repeated days and non-stride-one windows raise an error.
5. Prediction GRU_s remains window-local: `V_s,t = GRU_s(X[t-4:t], h0=0)`.
6. Each of eight heads independently constructs current Dynamic `D_t`.
7. Relative Static is `sigmoid(alpha) * D_prev + (1-sigmoid(alpha)) * weighted_static`.
8. Bidirectional Dynamic/Relative-Static attention uses separate projections and sum.
9. The fused relation diagonal is removed before propagating the projected GRU_s state.
10. Head outputs use tanh, followed by eight-head concatenation, adaptive residual,
    then a **Linear** prediction output, without final output tanh.
11. Persist the relation hidden and **pure Dynamic D_t** for the next day.
    Final fused relations are used for current propagation, not as next-day memory.

## Loss and State

`MSE + lambda_align * base_alignment + effective_lambda_pcda(epoch) * PCDA`

PCDA: 8 heads, 6-of-8 positive consensus, epsilon 1e-8, mutual candidate test,
per-source cap 3, dynamic-only weight 1, static-overlap weight 0.5, temperature 0.1.
PCDA is enabled by default, with maximum coefficient 1e-4. Schedule epochs 0-4
have coefficient zero; epochs 5-9 ramp to 1e-4. Recurrence continues during warmup.
The directed row cap is unchanged and need not leave a symmetric retained matrix.

TBPTT uses `batch_train=16` chronological days: mean the actual chunk losses,
backward once, clip gradients, optimizer step, zero gradients, then detach hidden
without clearing its numeric value. A shorter final chunk uses its own mean.
Non-reentrant activation checkpointing restores explicit head memory during
recomputation and preserves RNG state. No layer dimensions are reduced.

Training epochs and independent validation/test/phase sequences begin with both
hidden and explicit memory reset. Official checkpoint selection remains minimum
validation MAE, with the existing early-stopping rule and no forced PCDA eligibility gate.

## Explicit Relation Memory Intervention

- **Full**: retain recurrent hidden and previous-day per-head Dynamic matrices.
- **No Explicit Relation Memory**: clear only per-head matrices before each day.
- **Shuffled Explicit Relation Memory**: apply one stock permutation to both
  rows and columns of all eight previous-day matrices, retaining hidden unchanged.

Each independent inference pass resets both kinds of state at its start and end.
The model is replayed chronologically when loading a checkpoint. Formal inference
does not optimize alignment loss. No sensitivity or coupling analysis runs here.

## Formal Outputs

`SavedModels/<exp-name>/phaseX/`, `result/<exp-name>/`, `logs/<exp-name>/`.
The runner refuses occupied experiment directories and duplicate phase IDs.

Canonical `prediction.csv` columns:
`seed, phase_id, Date, Ticker, prediction, target_return_t_plus_1, prediction_rank, is_top5, actual_positive`.
Saving a phase replaces that seed/phase's rows instead of duplicating them.
`metrics.csv` preserves prediction/ranking/trading metrics, including Precision@5.
The per-phase prediction, audit, daily Precision@5, training-log, result-summary,
memory-analysis-summary and checkpoint-manifest files remain available.

Precision@5 uses all valid stocks before the trading positive-prediction filter.
Trading remains equal-weight Top5 with positive predictions only and zero
transaction cost. Sharpe, annualization, drawdown, MSE/MAE/IC/RankIC/MRR/NDCG
calculations are unchanged from the input A-share implementation.

`runtime_manifest_phaseX.json` records the actual phase, seed, device, loss
coefficients, temporal settings, Chinese BERT contract, and software/GPU environment.

## Verification Only

```powershell
python -m compileall -q .
python verify_package.py
python verify_fixed_implementation.py
python verify_dailyrecurrent_cn.py
python smoke_dailyrecurrent_cn.py --device cpu --debug-daily-relation
python smoke_forward_fixed.py --device cuda:0
```

The short smoke uses all 160 stocks and full model dimensions, two 17-day training
calls at schedule epochs 0 and 5, five validation days, and five test days in each
of the three intervention modes. This is not a formal two-epoch training run or
a performance comparison. Its temporary checkpoint is deleted after roundtrip.
The one-window CUDA check is not proof that full TBPTT=16 fits GPU memory.

## Formal Phase 1 (Not Started Automatically)

Activate the intended PyTorch environment and run inside this project:

```powershell
python run_n2m_allstocks_dailyrecurrent_cn.py --phases "1" --device cuda:0 --seed 11380 --lambda-align 1.0 --max-epoch 300 --wait-epoch 30 --exp-name "cn_dailyrecurrent_phase1_seed11380"
```

Use `--phases "1,2,3,4,5,6,7"` and a new experiment name for all phases. Defaults
remain unchanged; `--debug-daily-relation` enables the temporal trace. Do not reuse
checkpoints trained with a window-local relation GRU as formal Daily-Recurrent results.
