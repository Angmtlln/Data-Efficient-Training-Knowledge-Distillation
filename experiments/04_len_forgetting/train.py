"""Train ModernBERT on one saved selection and record its predictions."""
import argparse
import csv
import fcntl
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch
import transformers
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments/01_cpu_selection/code'))
from common import digest, read_data


def metrics(pred):
    labels = pred.label_ids
    predictions = np.argmax(pred.predictions, axis=-1)
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'macro_f1': float(f1_score(labels, predictions, average='macro')),
        'positive_f1': float(f1_score(labels, predictions)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--training-seed', type=int, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--output', type=Path, default=REPO / 'results/len_forgetting/runs')
    parser.add_argument('--checkpoint-root', type=Path, default=REPO / 'outputs/len_forgetting/checkpoints')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if not args.cpu and not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        raise RuntimeError('No CUDA or MPS GPU visible. Explicit --cpu required.')

    selection = json.loads(args.selection.read_text())
    expected_hash = selection.pop('sha256')
    if digest(selection) != expected_hash:
        raise ValueError('Selection checksum mismatch')
    dataset, snapshot = read_data()
    if selection['pool_sha256'] != snapshot['splits']['pool']['sha256']:
        raise ValueError('Pool hash mismatch')
    indices = selection['indices']
    if len(indices) != 6534 or len(set(indices)) != 6534:
        raise ValueError('Invalid selection size')

    config = json.loads((REPO / 'experiments/01_cpu_selection/config.json').read_text())
    tokenization = json.loads((REPO / 'experiments/01_cpu_selection/data/tokenization.json').read_text())
    if tokenization['snapshot'] != snapshot or tokenization['model_revision'] != config['model_revision']:
        raise ValueError('Tokenized data/model version mismatch')
    tokenized = load_from_disk(str(REPO / 'experiments/01_cpu_selection/data/tokenized'))
    run_id = f"{selection['name']}__ss{selection['selection_seed']}__ts{args.training_seed}"
    if args.pilot:
        run_id += '__pilot'
    result_dir = args.output / run_id
    checkpoint_dir = args.checkpoint_root / run_id

    # A second worker waits for the same run, then reuses its completed result.
    lock_dir = REPO / 'outputs/len_forgetting/locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (lock_dir / f'{run_id}.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    result_dir.mkdir(parents=True, exist_ok=True)
    if (result_dir / 'metrics.json').exists():
        saved = json.loads((result_dir / 'metrics.json').read_text())
        if saved['selection_sha256'] != expected_hash:
            raise ValueError('Existing run uses different selection')
        print('Already complete', run_id, flush=True)
        return

    set_seed(args.training_seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(REPO / 'experiments/01_cpu_selection/data/tokenizer'), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model'], revision=config['model_revision'], num_labels=2, attn_implementation='sdpa'
    )
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda'
    elif not args.cpu and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    training = config['training']
    options = dict(
        output_dir=str(checkpoint_dir), num_train_epochs=training['epochs'],
        per_device_train_batch_size=training['batch_size'], per_device_eval_batch_size=32,
        learning_rate=training['learning_rate'], weight_decay=training['weight_decay'],
        warmup_ratio=training['warmup_ratio'], seed=args.training_seed, data_seed=args.training_seed,
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='accuracy', greater_is_better=True, save_total_limit=2,
        logging_steps=20, report_to=[], bf16=device == 'cuda' and torch.cuda.is_bf16_supported(),
        fp16=device == 'cuda' and not torch.cuda.is_bf16_supported(), use_cpu=args.cpu,
        dataloader_num_workers=0, dataloader_pin_memory=False, optim='adamw_torch', save_safetensors=True,
    )
    if args.pilot:
        options.update(max_steps=20, eval_strategy='steps', eval_steps=10, save_strategy='steps', save_steps=10)
    trainer = Trainer(
        model=model, args=TrainingArguments(**options), train_dataset=tokenized['pool'].select(indices),
        eval_dataset=tokenized['val'], processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics,
    )
    last_checkpoint = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    started = time.time()
    print('RUN', run_id, 'device', device, 'resume', last_checkpoint, flush=True)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    evaluated_split = 'val' if args.pilot else 'test'
    predictions = trainer.predict(tokenized[evaluated_split])
    probabilities = torch.tensor(predictions.predictions).softmax(-1).numpy()
    with (result_dir / 'predictions.csv').open('w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['orig_idx', 'label', 'prediction', 'p_positive'])
        writer.writerows(zip(dataset[evaluated_split]['orig_idx'], predictions.label_ids,
                             probabilities.argmax(axis=1), probabilities[:, 1]))
    result = {
        'name': selection['name'], 'selection_seed': selection['selection_seed'],
        'training_seed': args.training_seed, 'selection_sha256': expected_hash,
        'metrics': metrics(predictions), 'split': evaluated_split, 'pilot': args.pilot,
        'seconds': time.time() - started, 'global_step': trainer.state.global_step,
        'best_checkpoint': trainer.state.best_model_checkpoint, 'best_metric': trainer.state.best_metric,
        'device': device, 'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'model_revision': config['model_revision'], 'log_history': trainer.state.log_history,
    }
    (result_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
