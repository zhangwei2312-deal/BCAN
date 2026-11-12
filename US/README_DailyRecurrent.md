# Daily Recurrent Dynamic Relation (U.S.)

This directory is the standalone Daily-Recurrent US training and submission
project. Verification uses local protected-definition fingerprints in
`core_math_contract.json`; no sibling project is required.

## Assets and transfer

The four data CSVs and English TinyBERT are available locally:
`us_news.csv`, `us_price_normalized.csv`, `us_target.csv`, `us_price.csv`,
and `Tinybert-4layers-en/`. Transfer these with the code, or explicitly provide
`--data-dir` and `--bert-path`. There is no asset fallback configuration.
Missing resources produce an error with the expected paths; nothing is downloaded.

Expected model: huawei-noah/TinyBERT_General_4L_312D, four layers,
hidden size 312, English WordPiece vocabulary 30522. It is loaded with
`from_pretrained`, excluded from task-model parameter reset, frozen, and kept in
eval mode even during task training. The numerical verification compares its
parameters against the supplied local pretrained model.

All `SavedModels`, `result`, and `logs` paths remain in this project, never in
`--data-dir`. Use a distinct `--exp-name`; the runner refuses existing nonempty
experiment directories. `smoke_only_*` outputs are validation artifacts, not
formal experimental results.

## Actual computation

Relation branch:

```text
bootstrap Z1..Z5 -> GRU_r(hidden=None) -> h5 -> eight R5 heads
current Z6 + h5 -> GRU_r -> h6 -> eight R6 heads
current Z7 + h6 -> GRU_r -> h7 -> eight R7 heads
```

The model stores the pre-dropout hidden in `_relation_hidden`. After bootstrap it
asserts a single trading-day update and rejects overlapping-day reprocessing.
`_relation_window_dates` and `_relation_date` are metadata, not model inputs.
Targets and future dates never enter the state transition.

The news and market encoders retain their rolling context. Z6 is the current-day
fused representation computed from that context. The separate prediction branch,
`GRUs_s`, still receives all five timesteps; it is not persistent.

The protected downstream computation remains: Current Dynamic -> Relative Static
-> bidirectional branch attention -> separate projections plus sum -> diagonal
removal -> propagation of projected prediction features -> eight-head concat
-> adaptive residual -> linear prediction. Alpha remains trainable in Relative
Static. There is no new output tanh or replacement concat/GELU fusion.

## Alignment and loss

Previous-day detached individual head matrices select the current-day PCDA pairs:

```text
D5 heads -> positive weights -> PCDA(news6, market6)
D6 heads -> positive weights -> PCDA(news7, market7)
```

PCDA receives `x_news_align[-1:]` and `x_market_align[-1:]` only. Its eight-head
gate, cap, bidirectional modality loss, independent news-endpoint filters, weights,
and schedule are unchanged. Basic Self+Static alignment remains News -> Market
with valid-news filtering and its existing five-day scope.

```text
total = MSE + lambda_align * base_alignment + lambda_pcda(epoch) * PCDA
```

The PCDA defaults are enabled with lambda_pcda=1e-4 in both main and runner.
Alpha remains trainable in Relative Static. PCDA warmup/ramp remain 5/5. It is an auxiliary training loss, not an inner
inference optimizer. Relations and losses are evaluated in the same forward
graph; parameter updates occur at optimizer chunk boundaries.

## State and gradients

- Reset both hidden and relation matrices at each train/validation epoch, split,
  phase and independent inference pass.
- Bootstrap from the original past-only five-day context; hidden never crosses
  a split.
- Carry hidden values and their graph through a TBPTT chunk.
- Average chunk losses, backward once, clip, step Adam, clear gradients, then
  detach hidden while retaining values.
- Normalize the final partial chunk by its actual window count. No training
  `retain_graph=True` is introduced.
- Activation recomputation preserves RNG and restores head memory and cached
  Fusion after backward. It never replays the persistent relation-GRU update.
- Hidden, head relations and analysis cache are transient, not checkpoint
  parameters. Runtime manifests identify `relation_temporal_mode=daily_recurrent`.

Checkpoints contain model parameters, not persistent recurrent state.
Each independent inference sequence starts with an empty hidden and head memory.

## Daily statistics and runtime records

PCDA statistics average the current terminal trading days in each pass:
`pcda_edges_per_day`, `pcda_dyn_only_edges_per_day`,
`pcda_static_overlap_edges_per_day`, and `pcda_zero_edge_day_ratio`.
Training logs, result CSVs and aggregate summaries use these names only.

`runtime_manifest.json` records the invocation. Each phase also produces
`runtime_manifest_phase{phase_id}.json` with its actual phase identifier.
Records include daily recurrence and prediction context, PCDA scope/heads/
consensus/schedule, both alignment coefficients, English BERT properties,
linear output, TBPTT steps, seed/device and Python/PyTorch/CUDA/cuDNN versions.

## Explicit Relation Memory Intervention

Displayed modes are:

- Full: retain previous-day explicit relation matrices and persistent hidden.
- No Explicit Relation Memory: clear only explicit relation matrices each day;
  retain the persistent relation hidden and rolling prediction context.
- Shuffled Explicit Relation Memory: apply one shared stock permutation to both
  axes and all head matrices; retain the persistent relation hidden.

Internal keys remain `full`, `no_memory`, `shuffled_memory`. These interventions
do not remove every source of temporal information, and do not optimize PCDA at
test time. Existing portfolio metrics and drop formulas remain unchanged.

## Verification without formal training

```powershell
python verify_package.py
python verify_fixed_implementation.py
python verify_dailyrecurrent.py
python smoke_forward_fixed.py --device cpu
python smoke_forward_fixed.py --device cuda:0
python smoke_dailyrecurrent.py --device cpu --debug-daily-relation
```

The last command uses actual P1 data and the unchanged 116-stock/8-head model.
It runs two short training passes of 17 windows, exercising schedule epochs 0
and 5 with accumulation 16, plus five validation windows per pass. It then
performs five chronological test predictions in each of the three intervention
modes. The test-prefix length is configurable with --test-days.

Checks include backward/optimizer, TBPTT, current-day-only PCDA, temporary
checkpoint save/load, canonical prediction and metric outputs, zero relation
diagonals, correct propagation shapes and preserved previous-day relation state.
Short-prefix metrics are not evidence of forecasting or portfolio performance.
The smoke never creates a formal training checkpoint.

## Formal training

The runner requires explicit `--phases`; omitting it exits without training.
Use `--phases "1"` for P1 or `--phases "1,2,3,4,5,6,7,8,9"` for all phases.
Logs are named `logs/{exp_name}/dailyrecurrent_us_phase{phase_id}.log`.
Model/result namespaces and overwrite protection remain unchanged.

### P1 example (not automatically executed)

```powershell
python run_n2m_filtered_dailyrecurrent_us.py --phases "1" --device cuda:0 --seed 11380 --lambda-align 0.1 --wait-epoch 20 --max-epoch 300 --exp-name "us_dailyrecurrent_phase1_lambda01_seed11380"
```

This explicitly sets lambda_align=0.1 and patience=20. Base-alignment and
patience defaults remain 1.0 and 30 respectively. Seed, patience and base-alignment strength remain configurable.
TBPTT retains more activations than independent-window backward, so the short
smoke does not guarantee memory capacity for every GPU or full training run.
