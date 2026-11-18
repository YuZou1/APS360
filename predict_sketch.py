"""Reproduce top-three predictions on a Quick, Draw! validation drawing."""
import argparse
from pathlib import Path

import numpy as np
import torch

from prepare_sketches import CLASSES, LEVELS, valid
from train_sketch_models import SketchClassifier, batch, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('quickdraw'))
    parser.add_argument('--checkpoints', type=Path, default=Path('experiment'))
    parser.add_argument('--category', choices=CLASSES, default='airplane')
    parser.add_argument('--index', type=int, default=60)
    args = parser.parse_args()
    label = CLASSES.index(args.category)
    # The true label is only used to locate and display the archived example.
    with np.load(args.data / f'{args.category}.npz', allow_pickle=True,
                 encoding='latin1') as archive:
        pool = archive['valid']
        if not (0 <= args.index < len(pool)) or not valid(pool[args.index]):
            parser.error('Index must select a valid drawing in the validation split.')
        views = prepare([(label, args.index, pool[args.index])])

    torch.set_num_threads(2)
    models = {}
    for key in ('cnn', 'cnn_gru'):
        checkpoint = torch.load(args.checkpoints / f'{key}.pt', map_location='cpu',
                                weights_only=True)
        if checkpoint['model'] != key or tuple(checkpoint['classes']) != CLASSES:
            raise ValueError(f'Checkpoint architecture or class order mismatch: {key}')
        net = SketchClassifier(use_gru=(key == 'cnn_gru'))
        net.to(memory_format=torch.channels_last)
        net.load_state_dict(checkpoint['state_dict'])
        models[key] = net.eval()

    with torch.inference_mode():
        for i, level in enumerate(LEVELS):
            print(f'{level}% (ground truth: {args.category}):')
            for key, net in models.items():
                x, _ = batch(views, np.array([0]), i, 'cpu', net.use_gru)
                guesses = net(*x).topk(3).indices[0].tolist()
                print(f'  {key:7} ' + ', '.join(CLASSES[g] for g in guesses))


if __name__ == '__main__':
    main()
