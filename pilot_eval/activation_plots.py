"""Lazy Matplotlib renderer for saved activation reports; no model libraries."""
import numpy as np


def plot_report(directory, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pilot_eval.activation_prepare import STEPS, VIEWS, PROJECTIONS
    from pilot_eval.run import _write_json

    folder = directory / 'plots'
    folder.mkdir(parents=True, exist_ok=True)
    records = result['measurements']
    STEPS = result.get('checkpoint_steps', STEPS)
    palette = ['#6c757d', '#467ba3', '#55a089', '#bf924d', '#a76b91']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#fafbfc'})
    outputs = []

    def save(fig, name):
        for extension in ['png', 'svg']:
            path = folder / (name + '.' + extension)
            fig.savefig(path, dpi=160, bbox_inches='tight')
            outputs.append(path.name)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
    for row, view in enumerate(VIEWS):
        for column, weighting in enumerate(['token', 'example']):
            axis = axes[row, column]
            for step, color in zip(STEPS, palette):
                cells = [r for r in records if r['kind'] == 'block' and r['view'] == view
                         and r['weighting'] == weighting and r['step'] == step]
                y = [np.nan if r['relative_write'] is None else r['relative_write'] for r in cells]
                bounds = np.array([r['interval_95'] or [np.nan, np.nan] for r in cells])
                axis.plot(range(28), y, color=color, label=result.get('variant_labels', {}).get(str(step), f'Step {step}'))
                axis.fill_between(range(28), bounds[:, 0], bounds[:, 1], color=color, alpha=.13)
            axis.set_title(f'{view} · {weighting} weighting')
            axis.set_ylabel('Norm(mean Δh) / mean norm(h)')
            axis.set_ylim(bottom=0)
            axis.grid(axis='y', alpha=.15)
            axis.set_xlabel('Decoder block (0–27)')
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle('Effective block write · paired 95% example bootstrap intervals', y=1.01)
    fig.tight_layout()
    save(fig, 'block-depth')

    values = [r['relative_write'] for r in records if r['kind'] == 'module' and r['relative_write'] is not None]
    maximum = max(values, default=0.)
    # Zero-only evidence still gets a nondegenerate drawing range; recorded max stays zero.
    vmax = maximum if maximum > 0 else 1.
    for weighting in ['token', 'example']:
        fig, axes = plt.subplots(3, len(STEPS), figsize=(3.2 * len(STEPS), 15), sharey=True)
        for row, view in enumerate(VIEWS):
            for column, step in enumerate(STEPS):
                axis = axes[row, column]
                grid = np.full((28, 7), np.nan)
                for cell in records:
                    if cell['kind'] == 'module' and cell['view'] == view and cell['step'] == step and cell['weighting'] == weighting:
                        grid[cell['layer'], PROJECTIONS.index(cell['projection'])] = (
                            np.nan if cell['relative_write'] is None else cell['relative_write'])
                cmap = plt.get_cmap('Blues').copy()
                cmap.set_bad('#d9dde3')
                drawing = axis.imshow(grid, aspect='auto', vmin=0, vmax=vmax, cmap=cmap, interpolation='nearest')
                axis.set_title(f"{view} · {result.get('variant_labels', {}).get(str(step), 'step ' + str(step))}", fontsize=8)
                axis.set_xticks(range(7), [p.removesuffix('_proj') for p in PROJECTIONS], rotation=45)
                axis.set_yticks([0, 7, 14, 21, 27])
                if column == 0:
                    axis.set_ylabel('Decoder block')
        fig.suptitle(f'Direct module / ordinary output magnitude · {weighting} weighting\nShared scale across both weightings; gray = undefined', y=.99)
        fig.subplots_adjust(top=.92, bottom=.06, wspace=.25, hspace=.25, right=.91)
        bar = fig.colorbar(drawing, cax=fig.add_axes([.93, .15, .012, .68]))
        bar.set_label('Mean direct norm / mean ordinary norm')
        save(fig, f'module-heatmaps-{weighting}')

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
    for row, view in enumerate(VIEWS):
        for column, weighting in enumerate(['token', 'example']):
            axis = axes[row, column]
            cells = [r for r in records if r['kind'] == 'block' and r['view'] == view and r['weighting'] == weighting and r['step'] == 0]
            axis.plot(range(28), [r['mean_base_norm'] for r in cells], color=palette[1], label='Mean baseline norm')
            axis.plot(range(28), [r['mean_base_vector_norm'] for r in cells], color=palette[3], label='Norm(mean baseline)')
            axis.set_title(f'{view} · {weighting} weighting')
            axis.set_ylabel('Baseline magnitude')
            axis.set_xlabel('Decoder block')
            axis.grid(axis='y', alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Denominator sensitivity · shared untuned block reference', y=1.01)
    fig.tight_layout()
    save(fig, 'block-denominators')
    _write_json(folder / 'manifest.json', {'files': outputs,
        'heatmap_scale': {'min': 0, 'data_max': maximum, 'display_max': vmax,
                          'shared_across': 'checkpoint/view/weighting', 'undefined_color': '#d9dde3'},
        'source_table': '../results/measurements.csv',
        'intervals': '2000 paired example replicates; seed 42; defined replicates only'})
