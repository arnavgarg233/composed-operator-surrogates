import argparse
import json
import os
from pathlib import Path

import numpy as np


def plot(payload, output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / 's_of_t_mpl_cache'))
    import matplotlib

    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('Use a PNG output for deterministic figure bytes')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not output.parent.is_dir():
        raise ValueError('Output parent must exist')
    if payload.get('schema') != 'pde-s-of-t-v1':
        raise ValueError('Expected pde-s-of-t-v1 JSON')
    times = np.asarray(payload['times'], dtype=float)
    ensemble = payload['ensemble']
    c = {k: np.asarray(v, dtype=float) for k, v in ensemble['curves'].items()}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    axes[0].semilogy(times[1:], c['r_pool'][1:], label='Available signal r', color='#0072B2')
    axes[0].semilogy(times[1:], c['epsilon_pool'][1:], label='Contrast error epsilon', color='#D55E00')
    axes[0].set(xlabel='Schedule time t', ylabel='Relative RMS', title='Matched truth and error')
    axes[0].axvline(1, color='0.7', linestyle=':', linewidth=1)
    axes[0].legend(fontsize=8)
    for index, item in enumerate(payload['per_seed'].values()):
        values = {k: np.asarray(v, dtype=float) for k, v in item['curves'].items()}
        axes[1].loglog(values['r_pool'][1:], values['S_pool'][1:], color='0.7', alpha=0.7,
                       linewidth=0.8, label='Individual seeds' if index == 0 else None)
        axes[2].plot(times[1:], values['margin_vs_common_pattern'][1:], color='0.7',
                     alpha=0.7, linewidth=0.8)
    valid = (c['r_pool'] > 0) & (c['S_pool'] > 0)
    axes[1].loglog(c['r_pool'][valid], c['S_pool'][valid], color='#0072B2', label='Prediction ensemble')
    fit = ensemble['fits']['pooled']
    if fit['slope'] is not None and valid.any():
        domain = np.geomspace(c['r_pool'][valid].min(), c['r_pool'][valid].max(), 100)
        axes[1].loglog(domain, np.exp(fit['intercept']) * domain**fit['slope'], '--',
                       color='#D55E00', label=f"OLS slope {fit['slope']:.3f}")
    axes[1].axhline(1, color='0.3', linestyle=':', linewidth=1)
    axes[1].set(xlabel='Available signal r', ylabel='Pooled S', title='All positive frames, time order')
    axes[1].legend(fontsize=8)
    ci = c['margin_ci95_pointwise']
    axes[2].fill_between(times[1:], ci[0, 1:], ci[1, 1:], color='#0072B2', alpha=0.15)
    axes[2].plot(times[1:], c['margin_vs_common_pattern'][1:], color='#0072B2', label='Ensemble mean-unit margin')
    axes[2].axhline(0, color='0.3', linestyle=':', linewidth=1)
    axes[2].axvline(1, color='0.7', linestyle=':', linewidth=1)
    axes[2].set(xlabel='Schedule time t', ylabel='Mean-unit S minus 1', title='Below zero beats common pattern')
    axes[2].legend(fontsize=8)
    if payload.get('synthetic_only'):
        fig.suptitle('Synthetic verification only, not learner results', fontsize=11)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [axis.get_tightbbox(renderer) for axis in axes]
    if any(boxes[i].overlaps(boxes[i+1]) for i in range(len(boxes)-1)):
        plt.close(fig)
        raise RuntimeError('Panel labels overlap')
    with output.open('xb') as stream:
        fig.savefig(stream, format=output.suffix.lstrip('.'), dpi=180,
                    metadata={'Software': 'pde-s-of-t'})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Headless S(t) figure from analyzer JSON only.')
    parser.add_argument('json_file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.json_file.read_text()), args.output)


if __name__ == '__main__':
    main()
