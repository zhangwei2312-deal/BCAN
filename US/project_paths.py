"""Explicit local assets; experiment outputs stay in this standalone project."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILES = ('us_news.csv', 'us_price.csv', 'us_price_normalized.csv', 'us_target.csv')
TEMPORAL_CONFIG = {
    'relation_temporal_mode': 'daily_recurrent',
    'relation_bootstrap_length': 5,
    'relation_state_persistent': True,
    'relation_state_detach_policy': 'mean_chunk_loss_backward_then_optimizer_step_then_hidden_detach',
    'pcda_temporal_scope': 'previous_day_headwise_relations_to_current_day_only',
    'prediction_temporal_scope': 'rolling_5_day',
    'activation_checkpointing': 'non_reentrant_rng_preserved_head_memory_restored',
}


def _resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def resolve_assets(data_dir='.', bert_path='./Tinybert-4layers-en'):
    data = _resolve(data_dir)
    bert = _resolve(bert_path)
    missing = [str(data / f) for f in DATA_FILES if not (data / f).is_file()]
    if missing:
        raise FileNotFoundError('Missing data in ' + str(data) + '. Place the four CSV files in the project or explicitly pass --data-dir. Missing: ' + ', '.join(missing))
    validate_english_bert_files(bert)
    return data, bert


def validate_english_bert_files(path):
    path = _resolve(path)
    required = ('config.json', 'vocab.txt')
    missing = [name for name in required if not (path / name).is_file()]
    if missing or not any((path / name).is_file() for name in ('pytorch_model.bin', 'model.safetensors')):
        raise FileNotFoundError(f'Missing English TinyBERT at {path}. Supply ./Tinybert-4layers-en or explicit --bert-path (huawei-noah/TinyBERT_General_4L_312D).')
    config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
    vocab = (path / 'vocab.txt').read_text(encoding='utf-8').splitlines()
    actual = (config.get('num_hidden_layers'), config.get('hidden_size'), config.get('vocab_size'), len(vocab))
    if actual != (4, 312, 30522, 30522):
        raise RuntimeError(f'English TinyBERT required (4,312,30522,30522); got {actual} at {path}. Chinese models are forbidden.')
    if not {'the', 'market', 'stock', 'news'}.issubset(set(vocab)):
        raise RuntimeError('Expected English WordPiece vocabulary was not found.')
    return path


def validate_loaded_bert(encoder):
    cfg = encoder.bert.config
    actual = (cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size, encoder.tokenizer.vocab_size)
    frozen = not any(p.requires_grad for p in encoder.bert.parameters())
    if actual != (4, 312, 30522, 30522) or not frozen or encoder.bert.training:
        raise RuntimeError(f'Invalid English TinyBERT state: config={actual}, frozen={frozen}, training={encoder.bert.training}')
    print(f'English TinyBERT verified: layers={actual[0]}, tokenizer_vocab={actual[3]}, hidden={actual[1]}, frozen={frozen}, eval=True')


def experiment_dir(kind, exp_name=None):
    root = PROJECT_ROOT / kind
    if exp_name:
        if exp_name in ('.', '..') or any(c in exp_name for c in '/\\:'):
            raise ValueError('Experiment name must be a single directory name.')
        root = root / exp_name
    if not root.resolve().is_relative_to(PROJECT_ROOT):
        raise ValueError('Output path escapes the project.')
    return root
