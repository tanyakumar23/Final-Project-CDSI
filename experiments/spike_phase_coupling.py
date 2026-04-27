# spike-theta phase coupling at scene boundaries
# this script looks at whether hippocampal neurons fire at a preferred phase of the
# local theta oscillation (4-8 Hz), and whether that preference changes at scene cuts.
# for each unit i classify spikes into three windows: scene change (0-1s after a hard cut),
# continuity cut (0-1s after a continuity cut), or baseline (more than 3s from any cut).
# then i look up the theta phase at each spike time and compute the mean vector length (MVL)
# which is basically how consistent the preferred firing phase is - 0 means random,
# 1 means all spikes at exactly the same phase.
# rayleigh test checks if the distribution is significantly non-uniform.
# watson-williams tests if the preferred phase is different between conditions.
# the main result is the polar histogram showing where in the theta cycle spikes tend to land.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import butter, filtfilt, hilbert
from scipy import stats
from pynwb import NWBHDF5IO

from config import NWB_DIR, SCENECUTS_CSV
OUT_FIG_DIR = './figures'
OUT_RES_DIR = './results'
os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_RES_DIR, exist_ok=True)

LFP_FS = 1000
THETA_LO, THETA_HI = 4, 8
FILTER_ORDER = 4
EPOCH_WIN = (0.0, 1.0)   # post-cut window for "change" and "cont" spikes
BASE_EXCL = 3.0          # 5s left too few baseline spikes, using 3 instead
N_BINS_POLAR = 18           # polar histogram bins
N_PERM = 5000
np.random.seed(3)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def mean_vector_length(phases):
    """mean vector length - phase-locking strength in [0,1]"""
    if len(phases) == 0:
        return np.nan, np.nan
    z = np.exp(1j * phases)
    mvl = np.abs(z.mean())
    pref = np.angle(z.mean())
    return float(mvl), float(pref)


def rayleigh_test(phases):
    """rayleigh test for circular uniformity, returns (Z, p)"""
    n = len(phases)
    if n < 5:
        return np.nan, 1.0
    r = np.abs(np.exp(1j * phases).mean())
    Z = n * r ** 2
    p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))
    return float(Z), float(np.clip(p, 0, 1))


def watson_williams_test(phases_a, phases_b):
    """watson-williams test for equal circular means, returns (F, p_approx)"""
    n1, n2 = len(phases_a), len(phases_b)
    if n1 < 3 or n2 < 3:
        return np.nan, 1.0
    r1 = np.abs(np.exp(1j * phases_a).mean())
    r2 = np.abs(np.exp(1j * phases_b).mean())
    pooled = np.concatenate([phases_a, phases_b])
    r_pool = np.abs(np.exp(1j * pooled).mean())
    n = n1 + n2
    R1, R2, R = n1 * r1, n2 * r2, n * r_pool
    F = (n - 2) * (R1 + R2 - R) / (n - R1 - R2 + 1e-30)
    p = 1 - stats.f.cdf(F, 1, n - 2)
    return float(F), float(np.clip(p, 0, 1))


def main():
    sc = pd.read_csv(SCENECUTS_CSV).reset_index(drop=True)
    new_sc = np.where(np.diff(sc['scene_id']))[0] + 1
    is_ch = np.zeros(len(sc), dtype=bool)
    is_ch[0] = True
    is_ch[new_sc] = True
    ch_times = sc.loc[is_ch, 'shot_start_t'].values
    co_times = sc.loc[~is_ch, 'shot_start_t'].values
    all_cuts = np.sort(np.concatenate([ch_times, co_times]))

    nwb_files = sorted(glob(os.path.join(NWB_DIR, 'sub-*', '*.nwb')))
    rows = []

    # collect phases across sessions for polar plots
    agg_phases = {'change': [], 'cont': [], 'base': []}

    for fpath in nwb_files:
        sid = os.path.basename(fpath).replace('.nwb', '')
        print(f'Processing {sid} ...')
        try:
            with NWBHDF5IO(fpath, 'r', load_namespaces=True) as io:
                nwb = io.read()
                el = nwb.electrodes.to_dataframe()

                # hippocampal neurons
                units = nwb.units.to_dataframe()
                if len(units) == 0:
                    print(f'  {sid}: no units'); continue

                hipp_el = el[el['location'].str.contains('hippo|Hip|CA', case=False, na=False)]
                if len(hipp_el) == 0:
                    print(f'  {sid}: no hipp electrodes'); continue

                # match units to hippocampal electrodes
                def _is_hipp(e):
                    return bool(e['location'].str.contains(
                        'hippo|Hip|CA', case=False, na=False).any())
                hipp_unit_mask = units['electrodes'].apply(_is_hipp)
                hipp_units = units[hipp_unit_mask]
                if len(hipp_units) == 0:
                    print(f'  {sid}: no hipp units'); continue

                # LFP for theta phase
                try:
                    lfp_obj = nwb.processing['ecephys']['LFP_macro'].electrical_series['ElectricalSeries']
                except (KeyError, AttributeError):
                    print(f'  {sid}: no LFP'); continue

                lfp_el = lfp_obj.electrodes.to_dataframe()
                hipp_lfp_pos = list(np.where(lfp_el['location'].str.contains(
                    'hippo|Hip|CA', case=False, na=False))[0])
                if not hipp_lfp_pos:
                    continue

                lfp_data = np.asarray(lfp_obj.data[:])
                n_lfp = lfp_data.shape[0]
                lfp_ts = (lfp_obj.starting_time or 0.0) + np.arange(n_lfp) / lfp_obj.rate
                raw_lfp = lfp_data[:, hipp_lfp_pos[0]].astype(float)

        except Exception as e:
            print(f'  {sid}: {e}'); continue

        # theta phase for full session
        nyq = LFP_FS / 2.0
        b, a = butter(FILTER_ORDER, [THETA_LO / nyq, THETA_HI / nyq], btype='band')
        theta_filt = filtfilt(b, a, raw_lfp)
        theta_phase = np.angle(hilbert(theta_filt))   # (n_samples,)

        # classify spikes per hip unit
        for uid, urow in hipp_units.iterrows():
            spike_times = np.array(urow['spike_times'])
            if len(spike_times) < 20:
                continue

            phases_ch, phases_co, phases_base = [], [], []

            for t_sp in spike_times:
                # classify spike into condition
                dt_ch = t_sp - ch_times
                dt_co = t_sp - co_times

                in_ch = np.any((dt_ch >= EPOCH_WIN[0]) & (dt_ch < EPOCH_WIN[1]))
                in_co = np.any((dt_co >= EPOCH_WIN[0]) & (dt_co < EPOCH_WIN[1]))

                dt_all = t_sp - all_cuts
                near_any = np.any(np.abs(dt_all) < BASE_EXCL)

                # theta phase at spike time
                idx = int(np.searchsorted(lfp_ts, t_sp))
                if idx <= 0 or idx >= len(theta_phase):
                    continue
                ph = theta_phase[idx]

                # print(f'spike at {t_sp:.2f}: in_ch={in_ch} in_co={in_co}')
                if in_ch:
                    phases_ch.append(ph)
                elif in_co:
                    phases_co.append(ph)
                elif not near_any:
                    phases_base.append(ph)

            phases_ch = np.array(phases_ch)
            phases_co = np.array(phases_co)
            phases_base = np.array(phases_base)

            mvl_ch, pref_ch = mean_vector_length(phases_ch)
            mvl_co, pref_co = mean_vector_length(phases_co)
            mvl_base, pref_base = mean_vector_length(phases_base)

            z_ch, p_ray_ch = rayleigh_test(phases_ch)
            z_co, p_ray_co = rayleigh_test(phases_co)

            F_ww, p_ww = watson_williams_test(phases_ch, phases_co) \
                if len(phases_ch) >= 5 and len(phases_co) >= 5 else (np.nan, 1.0)

            ci_mvl = (mvl_ch - mvl_co) / (abs(mvl_ch) + abs(mvl_co) + 1e-30) \
                if not (np.isnan(mvl_ch) or np.isnan(mvl_co)) else np.nan

            rows.append({
                'session': sid, 'unit': uid,
                'n_spikes_ch': len(phases_ch),
                'n_spikes_co': len(phases_co),
                'n_spikes_base': len(phases_base),
                'MVL_change': mvl_ch, 'pref_phase_change': pref_ch,
                'MVL_cont': mvl_co, 'pref_phase_cont': pref_co,
                'MVL_base': mvl_base, 'pref_phase_base': pref_base,
                'MVL_CI': ci_mvl,
                'rayleigh_Z_ch': z_ch, 'rayleigh_p_ch': p_ray_ch,
                'rayleigh_Z_co': z_co, 'rayleigh_p_co': p_ray_co,
                'watson_williams_F': F_ww, 'watson_williams_p': p_ww,
            })
            agg_phases['change'].extend(phases_ch.tolist())
            agg_phases['cont'].extend(phases_co.tolist())
            agg_phases['base'].extend(phases_base.tolist())

        print(f'  {sid}: {len([r for r in rows if r["session"]==sid])} units processed')

    if not rows:
        print('No units processed.'); return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_RES_DIR, 'spike_phase_results.csv'), index=False)

    ci_vals = df['MVL_CI'].dropna().values
    if len(ci_vals) >= 3:
        t_obs, p_t = stats.ttest_1samp(ci_vals, 0)
        null = np.array([np.mean(np.random.choice([-1,1], len(ci_vals)) * ci_vals)
                         for _ in range(N_PERM)])
        p_p = (np.abs(null) >= abs(ci_vals.mean())).mean()
        print(f'\nMVL CI: mean={ci_vals.mean():.3f}  t={t_obs:.2f}  '
              f'p_ttest={p_t:.4f}  p_perm={p_p:.4f}')
    else:
        p_p = np.nan
        print('Not enough units for group test.')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                             subplot_kw=dict(projection='polar'))
    labels = ['Scene Change', 'Continuity Cut', 'Baseline']
    colors = ['#d62728', '#1f77b4', '#2ca02c']
    for ax, (cat, label, col) in zip(axes,
                                      [('change', labels[0], colors[0]),
                                       ('cont',   labels[1], colors[1]),
                                       ('base',   labels[2], colors[2])]):
        phases = np.array(agg_phases[cat])
        if len(phases) == 0:
            ax.set_title(f'{label}\n(no data)', fontsize=8)
            continue
        bins = np.linspace(-np.pi, np.pi, N_BINS_POLAR + 1)
        counts, _ = np.histogram(phases, bins=bins)
        bin_centres = (bins[:-1] + bins[1:]) / 2
        ax.bar(bin_centres, counts / counts.sum(),
               width=2 * np.pi / N_BINS_POLAR,
               color=col, alpha=0.7, edgecolor='k', lw=0.3)
        mvl, pref = mean_vector_length(phases)
        ax.annotate('', xy=(pref, mvl), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='k', lw=2))
        z_r, p_r = rayleigh_test(phases)
        ax.set_title(f'{label}\nn={len(phases):,}  MVL={mvl:.3f}\n'
                     f'Rayleigh p={p_r:.3f}', fontsize=8)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticks(np.linspace(0, 2*np.pi, 5)[:-1])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2'], fontsize=7)

    fig.suptitle('Spike-Theta Phase Coupling at Scene Boundaries\n'
                 '(hippocampal units)', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_FIG_DIR, 'figS8_phase_coupling_polar.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    df_valid = df.dropna(subset=['MVL_change', 'MVL_cont'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.scatter(df_valid['MVL_change'], df_valid['MVL_cont'],
               c='steelblue', s=20, alpha=0.5, zorder=3)
    lim = max(df_valid[['MVL_change', 'MVL_cont']].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('MVL (Scene Change)', fontsize=8)
    ax.set_ylabel('MVL (Continuity Cut)', fontsize=8)
    ax.set_title('Spike Phase-Locking per Unit', fontsize=9)

    ax = axes[1]
    ci_all = df_valid['MVL_CI'].values
    ax.hist(ci_all, bins=20, color='steelblue', edgecolor='k', lw=0.3)
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.axvline(np.nanmean(ci_all), color='#d62728', lw=1.5,
               label=f'mean={np.nanmean(ci_all):.3f}')
    ax.set_xlabel('MVL Change Index (ch – co)', fontsize=8)
    ax.set_ylabel('Number of units', fontsize=8)
    ax.set_title(f'MVL CI distribution\np_perm = {p_p:.3f}', fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle('Hippocampal Spike Phase-Locking: MVL Change Index',
                 fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_FIG_DIR, 'figS9_mvl_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'\nSaved figures to {OUT_FIG_DIR}')
    print(f'Saved results to {OUT_RES_DIR}/spike_phase_results.csv')


main()
