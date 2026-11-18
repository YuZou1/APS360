"""Audit Quick, Draw! Sketch-RNN archives and create a reproducible pilot manifest.

Run with Python 3, numpy and Pillow. Use only the official Google archives:
https://storage.googleapis.com/quickdraw_dataset/sketchrnn/<category>.npz
The NumPy files use object arrays, so loading them requires trusted files.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ('airplane', 'apple', 'bicycle', 'car', 'cat',
           'dog', 'fish', 'flower', 'house', 'tree')
LEVELS = (25, 50, 75, 100)


def valid(stroke):
    return (
        stroke.ndim == 2 and stroke.shape[1] == 3 and len(stroke) >= 4
        and np.isfinite(stroke).all()
        and ((stroke[:, 2] == 0) | (stroke[:, 2] == 1)).all()
        and np.any((stroke[:-1, 2] == 0) & (stroke[1:, :2] != 0).any(axis=1))
    )


def format_prefix(stroke, percent):
    """Make one matching grayscale image and (dx, dy, pen-lift) sequence."""
    n = max(1, len(stroke) * percent // 100)
    observed = stroke[:n]
    points = np.cumsum(observed[:, :2], axis=0, dtype=np.float32)
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = max(float((hi - lo).max()), 1.0)
    xy = (points - (lo + hi) / 2) * (55.0 / span) + 31.5
    deltas = np.diff(xy, axis=0, prepend=xy[:1]) / 63
    sequence = np.column_stack((deltas, observed[:, 2])).astype(np.float32)
    image = Image.new('L', (64, 64), 255)
    draw = ImageDraw.Draw(image)
    pixels = [tuple(map(int, row)) for row in np.rint(xy)]
    for i, point in enumerate(pixels):
        draw.point(point, fill=0)
        if i and observed[i - 1, 2] == 0:
            draw.line((pixels[i - 1], point), fill=0, width=2)
    return image, sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(360)
    stats = []
    with (args.output / 'pilot_manifest.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('category', 'split', 'source_index'))
        for name in CLASSES:
            path = args.data / f'{name}.npz'
            with np.load(path, allow_pickle=True, encoding='latin1') as data:
                row = [name, path.stat().st_size]
                for split, target in [('train', 3000), ('valid', 500)]:
                    strokes = data[split]
                    accepted = []
                    rejected = 0
                    for idx in rng.permutation(len(strokes)):
                        stroke = strokes[idx]
                        if not valid(stroke):
                            rejected += 1
                            continue
                        writer.writerow((name, split, idx))
                        accepted.append(int(idx))
                        if len(accepted) == target:
                            break
                    row += [len(strokes), len(accepted), rejected]
                    if name == 'cat' and split == 'train':
                        # Inspect the first accepted source drawing, without
                        # choosing it based on a model prediction.
                        idx = accepted[0]
                        for level in LEVELS:
                            image, sequence = format_prefix(strokes[idx], level)
                            image.save(args.output / f'cat_{level}.png')
                            assert sequence.shape == (max(1, len(strokes[idx])*level//100), 3)
                        (args.output / 'example.txt').write_text(
                            f'cat train source index {idx}; {len(strokes[idx])} points\n',
                            encoding='utf-8')
                stats.append(row)
    with (args.output / 'pilot_stats.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('category', 'bytes', 'source_train', 'selected_train',
                         'skipped_train', 'source_valid', 'selected_valid', 'skipped_valid'))
        writer.writerows(stats)
    print((args.output / 'pilot_stats.csv').read_text())
    print((args.output / 'example.txt').read_text())


if __name__ == '__main__':
    main()
