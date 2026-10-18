"""Presentation-only publication figures from returned, frozen Stage B statistics.

No new inferential estimate or interpolation is computed here: points, uncertainty,
and Holm values come directly from the governed statistics report.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/qai-matplotlib-stage-L')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/twelfth-selection-v12/phase_B_operational_v1/statistics-return-v1/statistics.json'
OUT = ROOT / 'outputs/twelfth-selection-v12/phase_B_evidence_v1'
OUT.mkdir(parents=True, exist_ok=True)
data = json.loads(SOURCE.read_text())
assert data['status'] == 'PASS_STAGE_B_COMPLETE_ALL_K_AND_METHODS'
assert len(data['long_table']) == 196 and len(data['contrasts']) == 96

METHODS = ('interaction_tabu', 'shortgpt_bi_adapted', 'sleb_iterative_adapted', 'random')
COLORS = dict(zip(METHODS, ('#0072B2', '#E69F00', '#009E73', '#CC79A7')))
NAMES = {'interaction_tabu': 'Combination search',
         'shortgpt_bi_adapted': 'ShortGPT BI',
         'sleb_iterative_adapted': 'SLEB iterative', 'random': 'Random'}
MARKERS = dict(zip(METHODS, ('o', 's', '^', 'D')))
MODELS = ('qwen', 'mistral')
TASKS = ('piqa', 'hellaswag', 'gsm8k')
WORKLOADS = ('B1S512T128', 'B4S2048T128')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'savefig.facecolor': 'white'})

def save(fig, name):
    for suffix, options in (('.pdf', {}), ('.svg', {}), ('.png', {'dpi': 340})):
        fig.savefig(OUT / (name + suffix), bbox_inches='tight', **options)
    plt.close(fig)

def rows(model, workload='B1S512T128'):
    return [r for r in data['long_table'] if r['model'] == model and r['workload'] == workload]

def dense(model, workload='B1S512T128'):
    matches = [r for r in rows(model, workload) if r['deleted_layers'] == 0]
    assert len(matches) == 1
    return matches[0]

legend = [Line2D([0], [0], marker=MARKERS[m], linestyle='', color=COLORS[m],
                 label=NAMES[m], markersize=7) for m in METHODS]

# All fixed seed outcomes at the five discrete deletion counts; no lines between k.
fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.4), sharex=True)
for i, model in enumerate(MODELS):
    subset = rows(model)
    assert len(subset) == 49
    for j, task in enumerate(TASKS):
        ax = axes[i, j]
        base = dense(model)[task + '_accuracy']
        ax.axhline(base, color='#263746', linestyle='--', linewidth=1.1)
        ax.scatter([0], [base], marker='P', s=90, color='#263746', zorder=4)
        for method_index, method in enumerate(METHODS):
            points = [r for r in subset if r['method'] == method]
            assert len(points) == 12
            for seed_index, seed in enumerate((11, 23, 37)):
                selected = [r for r in points if r['seed'] == seed]
                assert {r['deleted_layers'] for r in selected} == {1, 2, 3, 4}
                for row in selected:
                    x = row['deleted_layers'] + (method_index - 1.5) * .13 + (seed_index - 1) * .035
                    ax.scatter(x, row[task + '_accuracy'], color=COLORS[method],
                               marker=MARKERS[method], s=32, alpha=.82,
                               edgecolor='white', linewidth=.45)
        ax.set_xticks((0, 1, 2, 3, 4))
        ax.set_xlim(-.35, 4.35)
        ax.set_ylim(0, 1)
        ax.set_title(model.upper() + ' · ' + task.upper())
        ax.grid(axis='y', color='#D9E1E8', linewidth=.6)
        if j == 0:
            ax.set_ylabel('Question accuracy')
        if i == 1:
            ax.set_xlabel('Whole layers deleted (discrete k)')
fig.legend(handles=legend + [Line2D([0], [0], marker='P', color='#263746',
                                   linestyle='', label='Dense')],
           loc='lower center', ncol=5, bbox_to_anchor=(.5, -.012), frameon=False)
fig.tight_layout(rect=(0, .04, 1, 1))
save(fig, 'figure_1_accuracy_by_discrete_k_all_seeds')

# Every configured method/seed/k in the measured quality versus serving plane.
fig, axes = plt.subplots(2, 2, figsize=(17.8, 11.0))
for i, model in enumerate(MODELS):
    for j, workload in enumerate(WORKLOADS):
        ax = axes[i, j]
        baseline = dense(model, workload)['piqa_accuracy']
        selected = [r for r in rows(model, workload) if r['deleted_layers'] > 0]
        assert len(selected) == 48
        for row in selected:
            lo, hi = row['speedup_CI_nominal95']
            speedup = row['speedup']
            ax.errorbar((row['piqa_accuracy'] - baseline) * 100, speedup,
                        yerr=[[max(0, speedup - lo)], [max(0, hi - speedup)]],
                        fmt=MARKERS[row['method']], color=COLORS[row['method']],
                        markeredgecolor='white', markersize=4 + row['deleted_layers'] * .8,
                        elinewidth=.65, capsize=1.5, alpha=.73)
        ax.axhline(1, color='#263746', linestyle='--', linewidth=1)
        ax.axvline(0, color='#263746', linestyle=':', linewidth=1)
        ax.grid(color='#D9E1E8', linewidth=.5)
        ax.set_title(model.upper() + ' · ' + workload)
        ax.set_xlabel('PIQA accuracy change from Dense (percentage points)')
        if j == 0:
            ax.set_ylabel('Dense/candidate fixed-length wall speedup')
fig.legend(handles=legend, loc='lower center', ncol=4,
           bbox_to_anchor=(.5, -.012), frameon=False)
fig.tight_layout(rect=(0, .04, 1, 1))
save(fig, 'figure_2_quality_serving_all_48_logical_points')

# Holm-96 family: all 96 prespecified model × k × method × task contrasts.
fig, axes = plt.subplots(2, 4, figsize=(22.5, 12.4), sharex=True)
for i, model in enumerate(MODELS):
    for j, k in enumerate((1, 2, 3, 4)):
        ax = axes[i, j]
        selected = [c for c in data['contrasts'] if c['model'] == model and c['k'] == k]
        assert len(selected) == 12
        selected.sort(key=lambda c: (TASKS.index(c['task']), METHODS.index(c['method'])))
        for y, cell in enumerate(selected):
            point = cell['accuracy_difference'] * 100
            lo, hi = (v * 100 for v in cell['nominal95CI'])
            method = cell['method']
            ax.errorbar(point, y, xerr=[[max(0, point - lo)], [max(0, hi - point)]],
                        fmt=MARKERS[method], color=COLORS[method], markersize=6,
                        markeredgecolor='#263746' if cell['reject_Holm96'] else 'white',
                        elinewidth=1, capsize=2)
        ax.axvline(0, color='#263746', linestyle='--', linewidth=.9)
        ax.set_yticks(range(12), [c['task'].upper() + ' · ' +
                                   NAMES[c['method']] for c in selected]
                      if j == 0 else [''] * 12)
        ax.invert_yaxis()
        ax.set_title(model.upper() + f' · k={k}')
        ax.grid(axis='x', color='#D9E1E8', linewidth=.5)
        if i == 1:
            ax.set_xlabel('Accuracy difference from Dense (percentage points)')
fig.tight_layout(rect=(0, .075, 1, 1))
fig.text(.5, .015, 'Error bars: nominal 95% bootstrap intervals; dark marker edge: Holm-96 rejection at 0.05.',
         ha='center', fontsize=9, color='#263746')
save(fig, 'figure_3_holm96_all_accuracy_contrasts')

# Tail risk is a descriptive E1 quality quantity; every logical seed is retained.
fig, axes = plt.subplots(1, 2, figsize=(16.2, 6.5), sharey=True)
for ax, model in zip(axes, MODELS):
    selected = [r for r in rows(model) if r['deleted_layers'] > 0]
    assert len(selected) == 48
    for row in selected:
        ax.scatter(100 * row['parameter_removed_ratio'], row['C4_positive_tail_CVaR90'],
                   color=COLORS[row['method']], marker=MARKERS[row['method']],
                   s=32 + row['deleted_layers'] * 7, alpha=.77,
                   edgecolor='white', linewidth=.4)
    ax.set_xlabel('Parameters removed (%)')
    ax.set_title(model.upper())
    ax.grid(color='#D9E1E8', linewidth=.5)
axes[0].set_ylabel('C4 positive NLL delta tail (CVaR90)')
fig.legend(handles=legend, loc='lower center', ncol=4,
           bbox_to_anchor=(.5, -.02), frameon=False)
fig.tight_layout(rect=(0, .05, 1, 1))
save(fig, 'figure_4_parameter_removal_c4_tail_risk')

# Lease occupancy includes failed runs; historical shared priors remain separate.
fig, axes = plt.subplots(1, 2, figsize=(16.8, 8.8), width_ratios=(2.8, 1.0))
for ax, historical in zip(axes, (False, True)):
    leases = [x for x in data['cost_lease_lineage'] if x['shared_or_historical'] == historical]
    labels = [x['role'].replace('-', ' ') for x in leases]
    values = [x['GPUh'] for x in leases]
    colors = ['#D55E00' if 'fail' in x['role'].lower() else
              '#56B4E9' if historical else '#0072B2' for x in leases]
    ax.barh(labels, values, color=colors, edgecolor='#263746', linewidth=.4)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel('Controller lease occupancy (GPU-hours)')
    ax.set_title('Shared historical prior' if historical else 'Incremental A + B runs, including failures')
    ax.grid(axis='x', color='#D9E1E8', linewidth=.5)
    for y, value in enumerate(values):
        ax.text(value, y, f' {value:.3f}', va='center', fontsize=8)
fig.tight_layout()
save(fig, 'figure_5_full_lease_cost_lineage')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

script = Path(__file__).resolve()
manifest = {'schema_version': 1,
            'project_id': 'b85b4b60-c0b8-4cc5-af06-81b4c63eb30a',
            'presentation_only': True,
            'source': {'path': str(SOURCE.relative_to(ROOT)), 'sha256': sha(SOURCE),
                       'status': 'previously observed E1 follow-up'},
            'script': {'path': str(script.relative_to(ROOT)), 'sha256': sha(script)},
            'no_new_inference_or_interpolation': True,
            'outputs': [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                         'bytes': path.stat().st_size}
                        for path in sorted(OUT.iterdir())
                        if path.suffix in ('.pdf', '.svg', '.png')]}
(OUT / 'figure_data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'figures': 5, 'files': len(manifest['outputs'])}))
