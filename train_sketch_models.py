"""Train comparable CNN and CNN+GRU pilots on the official Quick, Draw! splits.

Example:
  python train_sketch_models.py --data quickdraw --manifest pilot_manifest.csv \
      --output experiment --download

The optional downloader fetches only Google's official Sketch-RNN files.
Use only trusted archives: Google's stroke arrays require NumPy's pickle loader.
"""
import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn

from prepare_sketches import CLASSES, LEVELS, format_prefix, valid


def read_manifest(path, train_n, valid_n):
    groups = defaultdict(list)
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            groups[row['split'], row['category']].append(int(row['source_index']))
    for split, limit in [('train', train_n), ('valid', valid_n)]:
        for name in CLASSES:
            ids = groups[split, name][:limit]
            if len(ids) != limit or len(set(ids)) != limit or min(ids) < 0:
                raise ValueError(f'Missing, repeated or invalid manifest IDs: {split}/{name}')
            groups[split, name] = ids
    return groups


def load_data(args):
    groups = read_manifest(args.manifest, args.train_per_class, args.valid_per_class)
    drawings = {'train': [], 'valid': []}
    for label, name in enumerate(CLASSES):
        path = args.data / f'{name}.npz'
        if not path.exists() and args.download:
            args.data.mkdir(parents=True, exist_ok=True)
            url = f'https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{name}.npz'
            temp = path.with_suffix('.part')
            with urlopen(url, timeout=60) as response, temp.open('wb') as stream:
                copyfileobj(response, stream)
            temp.replace(path)
        with np.load(path, allow_pickle=True, encoding='latin1') as archive:
            for split in drawings:
                raw = archive[split]
                for idx in groups[split, name]:
                    stroke = raw[idx]
                    if not valid(stroke):
                        raise ValueError(f'Invalid sketch in manifest: {split}/{name}/{idx}')
                    drawings[split].append((label, idx, stroke))
        print(f'Loaded {name}', flush=True)
    return drawings


def prepare(records):
    # Cache four views per drawing; model batches read only the chosen view.
    n = len(records)
    images = np.empty((n, len(LEVELS), 64, 64), dtype=np.uint8)
    longest = max(len(stroke) for _, _, stroke in records)
    seqs = np.zeros((n, len(LEVELS), longest, 3), dtype=np.float32)
    lengths = np.empty((n, len(LEVELS)), dtype=np.int64)
    labels = np.asarray([r[0] for r in records], dtype=np.int64)
    ids = np.asarray([r[1] for r in records], dtype=np.int64)
    for i, (_, _, stroke) in enumerate(records):
        for j, level in enumerate(LEVELS):
            image, seq = format_prefix(stroke, level)
            images[i, j] = np.asarray(image)
            seqs[i, j, :len(seq)] = seq
            lengths[i, j] = len(seq)
    return images, seqs, lengths, labels, ids


class SketchClassifier(nn.Module):
    def __init__(self, use_gru=False):
        super().__init__()
        self.use_gru = use_gru
        blocks = []
        for a, b in [(1, 16), (16, 32), (32, 64)]:
            blocks += [nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        self.cnn = nn.Sequential(*blocks)
        self.gru = nn.GRU(3, 64, batch_first=True) if use_gru else None
        self.head = nn.Sequential(nn.Linear(128 if use_gru else 64, 128),
                                  nn.ReLU(), nn.Linear(128, len(CLASSES)))

    def forward(self, image, sequence=None, lengths=None):
        features = self.cnn(image).mean(dim=(2, 3))
        if self.gru is not None:
            states, _ = self.gru(sequence)
            # With a one-way GRU, padding after this index cannot affect it.
            final = states[torch.arange(len(lengths), device=states.device), lengths - 1]
            features = torch.cat((features, final), dim=1)
        return self.head(features)


def batch(cache, ids, levels, device, use_gru):
    images, seqs, lengths, labels, _ = cache
    pixels = torch.from_numpy(images[ids, levels, None]).to(device=device, dtype=torch.float32)
    pixels = (255.0 - pixels) / 255.0
    pixels = pixels.contiguous(memory_format=torch.channels_last)
    y = torch.from_numpy(labels[ids]).to(device)
    if not use_gru:
        return (pixels,), y
    sizes = lengths[ids, levels]
    seq = torch.from_numpy(seqs[ids, levels, :int(sizes.max())]).to(device)
    return (pixels, seq, torch.from_numpy(sizes).to(device)), y


def evaluate(model, cache, args, device, record=False):
    model.eval()
    result, predictions = [], []
    n = len(cache[3])
    with torch.inference_mode():
        for level_i, level in enumerate(LEVELS):
            hit1 = hit3 = 0
            loss_sum = 0.0
            for start in range(0, n, args.batch_size):
                ids = np.arange(start, min(start + args.batch_size, n))
                inputs, y = batch(cache, ids, level_i, device, model.use_gru)
                scores = model(*inputs)
                top = scores.topk(3).indices
                hit1 += top[:, 0].eq(y).sum().item()
                hit3 += top.eq(y[:, None]).any(1).sum().item()
                loss_sum += nn.functional.cross_entropy(scores, y, reduction='sum').item()
                if record:
                    for i, true, guesses in zip(ids, y.tolist(), top.tolist()):
                        predictions.append({'label': CLASSES[true],
                                            'source_index': int(cache[4][i]),
                                            'completion': int(level),
                                            'top1': CLASSES[guesses[0]],
                                            'top2': CLASSES[guesses[1]],
                                            'top3': CLASSES[guesses[2]]})
            result.append({'completion': level, 'n': n, 'top1_correct': hit1,
                           'top3_correct': hit3, 'top1_pct': 100 * hit1 / n,
                           'top3_pct': 100 * hit3 / n, 'loss': loss_sum / n})
    return result, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('quickdraw'))
    p.add_argument('--manifest', type=Path, default=Path('pilot_manifest.csv'))
    p.add_argument('--output', type=Path, default=Path('experiment'))
    p.add_argument('--train-per-class', type=int, default=1000)
    p.add_argument('--valid-per-class', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=360)
    p.add_argument('--threads', type=int, default=5)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--download', action='store_true')
    args = p.parse_args()
    if min(args.train_per_class, args.valid_per_class, args.epochs, args.batch_size, args.threads) < 1:
        p.error('Counts, epochs, batch size and threads must be positive.')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Choose an empty output directory so earlier run files stay intact.')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
    config = {'torch': str(torch.__version__), 'device': str(device),
              'numpy': np.__version__, 'seed': args.seed,
              'train_originals': args.train_per_class * len(CLASSES),
              'validation_originals': args.valid_per_class * len(CLASSES),
              'train_per_class': args.train_per_class,
              'valid_per_class': args.valid_per_class, 'epochs': args.epochs,
              'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
              'optimizer': 'Adam', 'levels': LEVELS, 'classes': CLASSES}
    (args.output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    raw = load_data(args)
    train = prepare(raw.pop('train'))
    val = prepare(raw.pop('valid'))
    print(f'Cached {len(train[3])} training and {len(val[3])} validation drawings', flush=True)

    models, optimizers = {}, {}
    for key in ['cnn', 'cnn_gru']:
        torch.manual_seed(args.seed)
        model = SketchClassifier(use_gru=(key == 'cnn_gru')).to(
            device=device, memory_format=torch.channels_last)
        models[key] = model
        optimizers[key] = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    counts = {key: sum(p.numel() for p in net.parameters()) for key, net in models.items()}
    best = {}
    n = len(train[3])
    init_levels = np.arange(n) % len(LEVELS)
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(n)
        views = (init_levels + epoch - 1) % len(LEVELS)
        for key, model in models.items():
            started = time.perf_counter()
            model.train()
            opt, loss_sum = optimizers[key], 0.0
            for start in range(0, n, args.batch_size):
                ids = order[start:start + args.batch_size]
                inputs, targets = batch(train, ids, views[ids], device, model.use_gru)
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(*inputs), targets)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * len(ids)
            levels, _ = evaluate(model, val, args, device)
            mean_top1 = float(np.mean([r['top1_pct'] for r in levels]))
            entry = {'model': key, 'epoch': epoch, 'train_loss': loss_sum / n,
                     'mean_val_top1_pct': mean_top1, 'levels': levels,
                     'elapsed_seconds': round(time.perf_counter() - started, 2)}
            with (args.output / 'history.jsonl').open('a') as log:
                print(json.dumps(entry), file=log)
            if key not in best or mean_top1 > best[key]['mean_val_top1_pct']:
                best[key] = entry
                torch.save({'model': key, 'state_dict': model.state_dict(),
                            'classes': CLASSES, 'epoch': epoch}, args.output / f'{key}.pt')
            print(f'Epoch {epoch}/{args.epochs} {key}: loss {loss_sum/n:.4f}, '
                  f'mean val top-1 {mean_top1:.2f}%, '
                  f'{entry["elapsed_seconds"]:.1f} s', flush=True)

    for key, model in models.items():
        checkpoint = torch.load(args.output / f'{key}.pt', map_location=device,
                                weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        levels, predictions = evaluate(model, val, args, device, record=True)
        best[key]['levels'] = levels
        with (args.output / f'{key}_predictions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
    (args.output / 'metrics.json').write_text(
        json.dumps({'config': config, 'parameters': counts, 'best': best}, indent=2) + '\n')
    print(f'Results written to {args.output}', flush=True)


if __name__ == '__main__':
    main()
