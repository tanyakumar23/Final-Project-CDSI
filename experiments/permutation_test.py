# sign-flip permutation test for the neural change index at scene boundaries
# loads the cached PSTH data from event_boundary_analysis.py and tests whether
# the mean change index per region is significantly different from zero.
# i use a sign-flip test because we don't have per-cut spike rates cached, only
# per-condition means - so the sign flip on per-unit CIs is the cleanest option here.
import os, pickle, numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

AREAS_ORDER = ['amygdala', 'hippocampus', 'ACC', 'preSMA', 'vmPFC']
AREA_COLORS = {
    'amygdala': '#e41a1c',
    'hippocampus': '#377eb8',
    'ACC': '#4daf4a',
    'preSMA': '#984ea3',
    'vmPFC': '#ff7f00',
}
N_PERM = 10000
np.random.seed(42)

from config import OUT_DIR
cache_path = OUT_DIR + '/analysis_cache.pkl'
out_dir = OUT_DIR

with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

bin_centers = cache['bin_centers']
psth_data = cache['psth_data']
cuts_df = cache['cuts_df']

POST_WIN = (bin_centers >= 0) & (bin_centers <= 1.0)
n_change = int(cuts_df['is_change'].sum())   # 13
n_total = len(cuts_df)                        # 93

print(f'Scene changes: {n_change} / {n_total} total cuts')
print(f'Running {N_PERM} permutations per region...\n')

fig, axes = plt.subplots(1, len(AREAS_ORDER), figsize=(14, 3.5))
fig.suptitle(f'Change Index: Observed vs Null (permutation, n={N_PERM})',
             fontsize=10, fontweight='bold')

results = {}

for ax, area in zip(axes, AREAS_ORDER):
    color = AREA_COLORS[area]

    ch_data = np.array(psth_data[area]['change'])  # shape (n_units, n_bins)
    co_data = np.array(psth_data[area]['cont'])

    if len(ch_data) == 0:
        ax.set_title(area)
        continue

    # observed change index per unit then mean
    r_ch = ch_data[:, POST_WIN].mean(axis=1)
    r_co = co_data[:, POST_WIN].mean(axis=1)
    denom = r_ch + r_co
    valid = denom > 0
    obs_ci = np.mean((r_ch[valid] - r_co[valid]) / denom[valid])

    # cache only has per-condition means, not per-cut spike rates, so i can't
    # do a full cut-label shuffle - sign-flip on per-unit CIs works fine here

    per_unit_ci = (r_ch[valid] - r_co[valid]) / denom[valid]

    null_means = np.zeros(N_PERM)
    for i in range(N_PERM):
        signs = np.random.choice([-1, 1], size=len(per_unit_ci))
        null_means[i] = np.mean(per_unit_ci * signs)

    p_perm = np.mean(null_means <= obs_ci)   # one-tailed: obs < 0

    results[area] = {
        'obs': obs_ci,
        'null': null_means,
        'p_perm': p_perm,
        'n_units': int(valid.sum()),
    }

    # plot null distribution
    ax.hist(null_means, bins=60, color='#999999', alpha=0.7,
            density=True, label='null distribution')
    ax.axvline(obs_ci, color=color, lw=2.5,
               label=f'observed\nCI={obs_ci:.4f}')
    ax.axvline(0, color='k', lw=0.8, ls=':')

    pstr = f'p={p_perm:.4f}' if p_perm >= 0.0001 else 'p<0.0001'
    ax.set_title(f'{area}\n{pstr}, n={valid.sum()} units', fontsize=8)
    ax.set_xlabel('Change index (null)', fontsize=7)
    if ax == axes[0]:
        ax.set_ylabel('Density')
    ax.legend(fontsize=6, frameon=False)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig5_permutation_test.png'), dpi=200, bbox_inches='tight')
plt.close()

print('\npermutation test results:')
for area in AREAS_ORDER:
    if area in results:
        r = results[area]
        print(f'{area}: obs_CI={r["obs"]:.4f}, p={r["p_perm"]:.4f}, n={r["n_units"]}')

