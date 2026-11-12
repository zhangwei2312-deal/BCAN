"""Run formal Daily-Recurrent CN training.

Base alignment uses valid news rows as anchors. PCDA retains both modality
directions and filters the news endpoint independently in each direction.
PCDA uses positive consensus from at least six of eight heads. Outputs use a separate
experiment namespace.
"""

import argparse
import os
import subprocess
import sys
from project_paths import PROJECT_ROOT, resolve_assets


DEFAULT_EXP_PREFIX = "cn_dailyrecurrent_n2m_allstocks_bidir_pcda"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Daily-Recurrent CN training "
            "with isolated outputs."
        )
    )
    parser.add_argument(
        "--phases",
        type=str,
        required=True,
        help="Required comma-separated phase IDs, e.g. 1 or 1,2,3,4,5,6,7.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=11380)
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help=(
            "Output namespace. Defaults to "
            f"{DEFAULT_EXP_PREFIX}_seed<SEED>."
        ),
    )
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--wait-epoch", type=int, default=30)
    parser.add_argument(
        "--lambda-align",
        type=float,
        default=1.0,
        help="Weight of the base self/static News-to-Market alignment loss.",
    )
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--batch-train", type=int, default=16)
    parser.add_argument("--bert-path", type=str, default="./Tinybert-4layers")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--debug-daily-relation", action="store_true")
    return parser.parse_args()


def parse_phase_list(value):
    phases = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        phase = int(part)
        if phase < 1 or phase > 7:
            raise ValueError(f"Phase must be between 1 and 7, got {phase}")
        if phase in phases:
            raise ValueError(f"Repeated phase {phase} would overwrite its outputs.")
        phases.append(phase)
    if not phases:
        raise ValueError("No phases were provided.")
    return phases


def build_phase_command(args, phase):
    return [
        sys.executable,
        "main_n2m_allstocks_dailyrecurrent.py",
        "--phase",
        str(phase),
        "--lambda-align",
        str(args.lambda_align),
        "--use-pcda-align",
        "1",
        "--lambda-pcda-align",
        "0.0001",
        "--pcda-pos-k",
        "6",
        "--pcda-eps-sign",
        "1e-8",
        "--pcda-dyn-only-weight",
        "1.0",
        "--pcda-static-overlap-weight",
        "0.5",
        "--pcda-row-cap",
        "3",
        "--pcda-align-warmup-epoch",
        "5",
        "--pcda-align-ramp-epoch",
        "5",
        "--pcda-temperature",
        "0.1",
        "--exp-name",
        args.exp_name,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--max-epoch",
        str(args.max_epoch),
        "--wait-epoch",
        str(args.wait_epoch),
        "--lr",
        "5e-4",
        "--rnn-length",
        "5",
        "--heads-att",
        "8",
        "--hidn-att",
        "512",
        "--hidn-rnn",
        "256",
        "--hidn-news",
        "768",
        "--hidn-market",
        "64",
        "--dropout",
        "0.4",
        "--static-thresh",
        "0.5",
        "--bert-batch-size",
        str(args.bert_batch_size),
        "--batch-train",
        str(args.batch_train),
        "--bert-path",
        args.bert_path,
        "--data-dir",
        args.data_dir,
    ] + (["--debug-daily-relation"] if args.debug_daily_relation else [])


def run_phase(args, phase, logs_dir):
    print(f"\nDaily-Recurrent CN training: Phase {phase}", flush=True)
    print(f"Experiment name: {args.exp_name}", flush=True)
    print(f"Seed: {args.seed}", flush=True)

    command = build_phase_command(args, phase)
    log_path = logs_dir / f"dailyrecurrent_cn_phase{phase}.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        log_file.write("Command: " + " ".join(command) + "\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
            env=env,
            cwd=str(PROJECT_ROOT),
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if return_code != 0:
        print(f"\nFAILED: phase={phase}, return_code={return_code}", flush=True)
        sys.exit(return_code)

    print(f"\nFinished phase {phase}", flush=True)
    print(f"  result/{args.exp_name}/all_phases_results_phase{phase}.csv", flush=True)
    print(f"  result/{args.exp_name}/analysis_summary_phase{phase}.csv", flush=True)
    print(f"  SavedModels/{args.exp_name}/phase{phase}/", flush=True)


def main():
    args = parse_args()
    phases = parse_phase_list(args.phases)
    if args.exp_name is None:
        args.exp_name = f"{DEFAULT_EXP_PREFIX}_seed{args.seed}"
    if any(char in args.exp_name for char in ("/", "\\", ":")):
        raise ValueError(
            "--exp-name must be a simple directory name without /, \\, or :"
        )

    if args.exp_name in ('.', '..'):
        raise ValueError('Experiment name cannot be . or ..')
    args.data_dir, args.bert_path = map(str, resolve_assets(args.data_dir, args.bert_path))
    project_root = PROJECT_ROOT
    main_script = project_root / "main_n2m_allstocks_dailyrecurrent.py"
    if not main_script.exists():
        raise FileNotFoundError(
            "main_n2m_allstocks_dailyrecurrent.py not found. Run this script from this package root."
        )

    output_roots = [
        project_root / "SavedModels" / args.exp_name,
        project_root / "result" / args.exp_name,
        project_root / "logs" / args.exp_name,
    ]
    occupied = [path for path in output_roots if path.exists() and any(path.iterdir())]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite an existing experiment. Choose a new --exp-name:\n  "
            + "\n  ".join(str(path) for path in occupied)
        )

    logs_dir = project_root / "logs" / args.exp_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Daily-Recurrent CN training")
    print("Relation temporal mode: DailyRecurrent; bootstrap=5, subsequent updates=1")
    print("PCDA: previous-day eight heads -> current day only; TBPTT at optimizer chunks")
    print(f"Assets: {args.data_dir}; Chinese TinyBERT: {args.bert_path}")
    print("TinyBERT: pretrained weights preserved and frozen")
    print("News GRU: true headline lengths packed before recurrence")
    print("Backtest outputs: Precision@5 plus complete prediction CSV files")
    print("Base alignment: News anchors -> Market candidates")
    print("News-anchor filter: enabled (zero-news rows are excluded)")
    print("PCDA directions: News -> Market and Market -> News")
    print("PCDA filter: the news endpoint is filtered independently per direction")
    print(f"Phases: {phases}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Experiment: {args.exp_name}")
    print(f"Models: SavedModels/{args.exp_name}/phaseX")
    print(f"Results: result/{args.exp_name}/all_phases_results_phaseX.csv")
    print(f"Logs: logs/{args.exp_name}/dailyrecurrent_cn_phaseX.log")
    print("=" * 90)

    for phase in phases:
        run_phase(args, phase, logs_dir)

    print("\nAll requested phases finished successfully.", flush=True)


if __name__ == "__main__":
    main()
