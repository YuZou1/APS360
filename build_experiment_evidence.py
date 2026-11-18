"""Turn saved validation logs and predictions into the report evidence."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prepare_sketches import LEVELS, format_prefix


def validate_saved_predictions(path, results):
    rows = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['label'], int(row['source_index']), int(row['completion']))
            assert key not in rows
            rows[key] = (row['top1'], row['top2'], row['top3'])
    counts = defaultdict(lambda: [0, 0, 0])
    for (label, _, level), guesses in rows.items():
        counts[level][0] += 1
        counts[level][1] += int(label == guesses[0])
        counts[level][2] += int(label in guesses)
    for r in results:
        assert counts[r['completion']] == [r['n'], r['top1_correct'], r['top3_correct']]
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((a.run / 'metrics.json').read_text())
    history = [json.loads(s) for s in (a.run / 'history.jsonl').read_text().splitlines()]
    keys = ['cnn', 'cnn_gru']
    names = ['CNN', 'CNN + GRU']
    colors = ['#285275', '#a84d28']
    best = metrics['best']
    preds = {
        k: validate_saved_predictions(a.run / f'{k}_predictions.csv', best[k]['levels'])
        for k in keys
    }
    assert preds['cnn'].keys() == preds['cnn_gru'].keys()
    originals = {
        f: {(lab, idx) for lab, idx, completion in preds['cnn'] if completion == f}
        for f in LEVELS
    }
    assert all(originals[f] == originals[100] for f in LEVELS)
    config = metrics['config']
    assert len(history) == len(keys) * config['epochs']
    assert len(originals[100]) == config['validation_originals']
    for k in keys:
        assert [h['epoch'] for h in history if h['model'] == k] == list(
            range(1, config['epochs'] + 1))
        assert best[k]['mean_val_top1_pct'] == max(
            h['mean_val_top1_pct'] for h in history if h['model'] == k)

    fig, ax = plt.subplots(figsize=(5.1, 1.58))
    for k, name, color, marker in zip(keys, names, colors, ['o', 's']):
        epochs = [h for h in history if h['model'] == k]
        ax.plot([h['epoch'] for h in epochs], [h['mean_val_top1_pct'] for h in epochs],
                marker=marker, markersize=3, color=color, linewidth=1.4, label=name)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dce3e7', linewidth=.55)
    ax.set(xlabel='Epoch', ylabel='Validation top-1 (%)',
           xticks=list(range(1, config['epochs'] + 1)))
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.legend(frameon=False, loc='best', fontsize=8)
    fig.tight_layout(pad=.42)
    fig.savefig(a.output / 'training_curves.png', dpi=275)
    plt.close(fig)

    val = sorted(key for key in preds['cnn'] if key[2] == 50)
    better = [k for k in val if preds['cnn'][k][0] != k[0] == preds['cnn_gru'][k][0]]
    worse = [k for k in val if preds['cnn_gru'][k][0] != k[0] == preds['cnn'][k][0]]
    if better and worse:
        samples = [better[0], worse[0]]
        selection = 'The two examples were selected to show disagreements in both directions.'
    else:
        samples = val[:2]
        selection = 'These are two validation sketches in source order.'
    figures, sample_records = [], []
    for i, key in enumerate(samples, 1):
        lab, idx, level = key
        with np.load(a.data / f'{lab}.npz', allow_pickle=True, encoding='latin1') as arc:
            image, _ = format_prefix(arc['valid'][idx], level)
        image.save(a.output / f'prediction_case_{i}.png')
        cnn = preds['cnn'][key]
        gru = preds['cnn_gru'][key]
        sample_records.append({'label': lab, 'validation_index': idx, 'completion': level,
                               'cnn_top3': cnn, 'cnn_gru_top3': gru})
        figures.append(
            rf"\begin{{minipage}}{{.46\linewidth}}\centering"
            rf"\fbox{{\includegraphics[width=1.3cm]{{prediction_case_{i}.png}}}}\\[1pt]"
            rf"\small\textbf{{{lab.capitalize()}}} (validation row {idx})\\"
            rf"CNN: {', '.join(cnn)}\\"
            rf"CNN + GRU: {', '.join(gru)}"
            r"\end{minipage}"
        )
    (a.output / 'prediction_examples.tex').write_text(
        r'\hfill'.join(figures) + '\n', encoding='utf-8')
    (a.run / 'illustrative_examples.json').write_text(json.dumps({
        'rule': selection, 'cnn_gru_only_correct_at_50': len(better),
        'cnn_only_correct_at_50': len(worse), 'examples': sample_records
    }, indent=2) + '\n')

    training = config['train_originals']
    validation = config['validation_originals']
    means = [best[k]['mean_val_top1_pct'] for k in keys]
    first = [best[k]['levels'][0]['top1_pct'] for k in keys]
    full = [best[k]['levels'][-1]['top1_pct'] for k in keys]
    text = {}
    for k, name in zip(keys, ['Cnn', 'Gru']):
        text[f'{name}Epoch'] = str(best[k]['epoch'])
        text[f'{name}Params'] = f'{metrics["parameters"][k]:,}'
    text.update({
        'CnnMean': f'{means[0]:.2f}', 'GruMean': f'{means[1]:.2f}',
        'CnnFirst': f'{first[0]:.2f}', 'GruFirst': f'{first[1]:.2f}',
        'CnnFull': f'{full[0]:.2f}', 'GruFull': f'{full[1]:.2f}',
        'RunTrain': f'{training:,}', 'RunValid': f'{validation:,}',
        'RunEpochs': str(config['epochs']), 'RunBatch': str(config['batch_size']),
        'CnnOnlyFifty': str(len(worse)), 'GruOnlyFifty': str(len(better)),
        'CaseSelection': selection,
    })
    for key, prefix in zip(keys, ['Cnn', 'Gru']):
        for metric, suffix in [('top1_pct', 'One'), ('top3_pct', 'Three')]:
            text[f'{prefix}Table{suffix}'] = ' & '.join(
                f'{r[metric]:.2f}' for r in best[key]['levels'])
    latex = '\n'.join(rf'\newcommand{{\{k}}}{{{v}}}' for k, v in text.items())
    (a.output / 'run_summary.tex').write_text(latex + '\n', encoding='utf-8')
    print(json.dumps({'mean_top1_pct': dict(zip(keys, means)),
                      'validation_examples': sample_records}, indent=2))


if __name__ == '__main__':
    main()
