'''Hippocampal theta reset and saccade amplitude at scene cuts; CI, memory prediction.'''
# this is the main analysis script for the bmovie event boundary project.
# basic idea: scene cuts in the movie are either hard scene changes (new location/time)
# or continuity cuts (same scene, just different camera angle). the question is whether
# the brain treats these differently, specifically whether hippocampal theta resets more
# at actual scene boundaries.
# for each subject i extract hippocampal LFP, bandpass to 4-8 Hz (theta), take the
# hilbert envelope to get instantaneous amplitude, then epoch around each cut type.
# a change index is computed as (change - cont) / (|change| + |cont|) so it's bounded
# between -1 and 1. positive = more theta at scene changes, negative = suppression.
# same thing is done for saccade amplitude using the eye tracking data.
# at the end i test whether these indices predict recognition memory (AUC from the
# old/new recognition task each subject did after the movie).
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import butter, filtfilt, hilbert
from scipy import stats
from pynwb import NWBHDF5IO
from sklearn.metrics import roc_auc_score

# change to your path
NWB_DIR = 'E:/000623'
SCENECUTS_CSV = 'E:/bmovie-release-NWB-BIDS/assets/annotations/scenecut_info.csv'
OUT_FIG_DIR = './figures'
OUT_RES_DIR = './results'

LFP_FS = 1000            # Hz, confirmed from NWB
THETA_LO = 4               # Hz
THETA_HI = 8               # Hz
FILTER_ORDER = 4               # Butterworth order
PRE_S = 1.0             # epoch pre-event window (s)
POST_S = 2.0             # epoch post-event window (s)
BL_WIN = (-0.5, 0.0)    # baseline window (s)
RESP_WIN = (0.0, 1.0)    # response window (s)
SAC_WIN = (0.0, 1.0)    # saccade amplitude collection window (s)

N_PERM = 5000
np.random.seed(0)

os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_RES_DIR, exist_ok=True)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def butter_bandpass(signal_1d, lo=THETA_LO, hi=THETA_HI, fs=LFP_FS, order=FILTER_ORDER):
    """zero-phase butterworth bandpass filter"""
    nyq = fs / 2.0
    # order 2 was too smooth, 4 works better
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, signal_1d)


def analytic_envelope(filtered_1d):
    """hilbert envelope - instantaneous amplitude of filtered signal"""
    return np.abs(hilbert(filtered_1d))


def epoch_trace(trace, t_lfp, event_times, pre=PRE_S, post=POST_S, fs=LFP_FS):
    """cut trace into epochs around event_times, returns (n_valid_events, n_win_samples) or None"""
    n_pre = int(pre * fs)
    n_post = int(post * fs)
    epochs = []
    for t0 in event_times:
        idx0 = int(np.searchsorted(t_lfp, t0))
        start = idx0 - n_pre
        stop = idx0 + n_post
        if start < 0 or stop > len(trace):
            continue
        epochs.append(trace[start:stop])
    return np.array(epochs) if epochs else None


def baseline_pct_change(epochs, t_axis, bl_win=BL_WIN):
    """percent change relative to baseline window for each epoch"""
    bl_mask = (t_axis >= bl_win[0]) & (t_axis < bl_win[1])
    if bl_mask.sum() == 0:
        return epochs
    bl_mean = epochs[:, bl_mask].mean(axis=1, keepdims=True)
    bl_mean = np.where(np.abs(bl_mean) < 1e-12, 1e-12, bl_mean)
    return (epochs - bl_mean) / np.abs(bl_mean) * 100.0


def resp_mean(epochs_norm, t_axis, resp_win=RESP_WIN):
    """mean normalized power in the response window"""
    mask = (t_axis >= resp_win[0]) & (t_axis < resp_win[1])
    if mask.sum() == 0:
        return np.nan
    return float(epochs_norm[:, mask].mean())


def change_index(r_change, r_cont):
    """symmetric index: (change - cont) / (|change| + |cont|)"""
    denom = abs(r_change) + abs(r_cont)
    if denom < 1e-12:
        return np.nan
    return (r_change - r_cont) / denom


def get_hip_channel_cols(nwb, lfp_es):
    """return hippocampal channel indices in the LFP ElectricalSeries"""
    try:
        elec_inds = np.array(lfp_es.electrodes.data[:], dtype=int)
        all_locs = np.array(nwb.electrodes.to_dataframe()['location'].values)
        ch_locs = all_locs[elec_inds]   # one per LFP column
        hip_mask = np.array(['hippocampus' in str(l).lower()
                               for l in ch_locs])
        return np.where(hip_mask)[0], ch_locs
    except Exception as e:
        return np.array([], dtype=int), np.array([])


def get_lfp_time(lfp_es):
    """return timestamp array for an LFP ElectricalSeries"""
    if lfp_es.timestamps is not None:
        return np.asarray(lfp_es.timestamps[:], dtype=float)
    n = lfp_es.data.shape[0]
    return lfp_es.starting_time + np.arange(n, dtype=float) / lfp_es.rate


def get_saccade_amp_per_cut(sac_t, sac_amp, cut_times, win=SAC_WIN):
    """mean saccade amplitude in win around each cut, NaN if no saccades"""
    out = np.full(len(cut_times), np.nan)
    for i, t0 in enumerate(cut_times):
        mask = (sac_t >= t0 + win[0]) & (sac_t < t0 + win[1])
        if mask.sum() > 0:
            out[i] = sac_amp[mask].mean()
    return out


def compute_auc(trials_df):
    """recognition AUC from NWB trials table"""
    recog = trials_df[trials_df['stim_phase'] == 'recognition']
    if len(recog) == 0:
        return np.nan
    y_true = recog['stimulus_file'].str.startswith('old').astype(int).values
    y_score = recog['actual_response'].values
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def frame_times_to_cut_times(frame_times_arr, frame_idx_series):
    """convert shot_start_fr to session timestamps"""
    return frame_times_arr[frame_idx_series.to_numpy(int) - 1, 1]


cuts_df = pd.read_csv(SCENECUTS_CSV).reset_index(drop=True)
new_scenes = np.where(np.diff(cuts_df['scene_id']))[0] + 1
is_change = np.zeros(len(cuts_df), dtype=bool)
is_change[0] = True
is_change[new_scenes] = True
cuts_df['is_change'] = is_change

print(f'Scene cut annotations: {is_change.sum()} scene changes, '
      f'{(~is_change).sum()} continuity cuts  ({len(cuts_df)} total)')

# time axis for epochs, centred at 0 = cut onset
t_epoch = np.arange(-PRE_S, POST_S, 1.0 / LFP_FS)


nwb_files = [f for f in sorted(glob(os.path.join(NWB_DIR, 'sub-*/sub-*CSR1*.nwb')))
             if 'CS53' not in os.path.basename(f)]   # CS53 excluded: 74-84% missing data
print(f'\nFound {len(nwb_files)} encoding (CSR1) sessions (CS53 excluded)\n')

session_records = []
grand_change_pool = []   # change condition epochs, averaged per session
grand_cont_pool = []     # cont condition epochs, averaged per session

for nwb_path in nwb_files:
    subj = os.path.basename(nwb_path).split('_')[0]   # e.g. 'sub-CS41'
    print(f'  {subj} ...', end=' ', flush=True)

    rec = {'subject': subj, 'theta_ci': np.nan, 'saccade_ci': np.nan,
           'auc': np.nan, 'n_hip_ch': 0, 'n_sac': 0}

    try:
        with NWBHDF5IO(nwb_path, 'r') as io:
            nwb = io.read()

            frame_arr = np.column_stack((
                nwb.stimulus['movieframe_time'].data[:],
                nwb.stimulus['movieframe_time'].timestamps[:]
            )).astype(float)

            trials_df = nwb.trials.to_dataframe()
            enc_rows = trials_df[trials_df['stim_phase'] == 'encoding']
            if len(enc_rows) == 0:
                print('no encoding, skip')
                session_records.append(rec)
                continue
            enc_start = float(enc_rows['start_time'].values[0])
            enc_stop = float(enc_rows['stop_time'].values[0])

            rec['auc'] = compute_auc(trials_df)

            cut_times = frame_times_to_cut_times(frame_arr, cuts_df['shot_start_fr'])
            change_times = cut_times[is_change]
            cont_times = cut_times[~is_change]

            # filter: only cuts within encoding (+ small margins for epoch)
            def enc_mask(times):
                return times[(times > enc_start + PRE_S) &
                             (times < enc_stop - POST_S)]

            change_times = enc_mask(change_times)
            cont_times = enc_mask(cont_times)

            beh = nwb.processing['behavior']
            sac_ts = beh['Saccade']['TimeSeries']
            sac_times = np.asarray(sac_ts.timestamps[:], dtype=float)
            sac_data = np.asarray(sac_ts.data[:])
            # print(sac_data[:5, :])  # was checking which column is amplitude
            sac_amp = sac_data[:, 5] if sac_data.ndim > 1 else sac_data  # col 5 = amplitude (deg)

            # keep only saccades during encoding
            enc_sac = (sac_times >= enc_start) & (sac_times <= enc_stop)
            sac_times_enc = sac_times[enc_sac]
            sac_amp_enc = sac_amp[enc_sac]
            rec['n_sac'] = int(enc_sac.sum())

            # mean amplitude per cut
            amp_change = get_saccade_amp_per_cut(sac_times_enc, sac_amp_enc, change_times)
            amp_cont = get_saccade_amp_per_cut(sac_times_enc, sac_amp_enc, cont_times)

            mu_ch_amp = np.nanmean(amp_change)
            mu_co_amp = np.nanmean(amp_cont)
            rec['saccade_ci'] = change_index(mu_ch_amp, mu_co_amp)
            rec['mean_sac_change'] = mu_ch_amp
            rec['mean_sac_cont'] = mu_co_amp

            if 'ecephys' not in nwb.processing:
                print('no ecephys, theta skipped')
                session_records.append(rec)
                continue

            lfp_es = nwb.processing['ecephys']['LFP_macro']['ElectricalSeries']
            hip_cols, ch_locs = get_hip_channel_cols(nwb, lfp_es)
            rec['n_hip_ch'] = int(len(hip_cols))

            if len(hip_cols) == 0:
                print('0 hip channels, theta skipped')
                session_records.append(rec)
                continue

            # load only hippocampal columns from HDF5
            raw_v = np.asarray(lfp_es.data[:, hip_cols], dtype=float) * 1e6  # V to uV
            t_lfp = get_lfp_time(lfp_es)

            # restrict to encoding window (with epoch margins)
            enc_mask_lfp = (t_lfp >= enc_start - PRE_S - 0.1) & \
                           (t_lfp <= enc_stop + POST_S + 0.1)
            raw_v = raw_v[enc_mask_lfp, :]
            t_lfp = t_lfp[enc_mask_lfp]

            # filter + envelope per channel, then average across channels
            envelopes = np.zeros_like(raw_v)
            for c in range(raw_v.shape[1]):
                filt = butter_bandpass(raw_v[:, c])
                envelopes[:, c] = analytic_envelope(filt)

            theta_trace = envelopes.mean(axis=1)   # average across hip channels

            epochs_ch = epoch_trace(theta_trace, t_lfp, change_times)
            epochs_co = epoch_trace(theta_trace, t_lfp, cont_times)

            if epochs_ch is None or epochs_co is None:
                print('insufficient epochs, theta skipped')
                session_records.append(rec)
                continue

            # baseline normalize (percent signal change)
            norm_ch = baseline_pct_change(epochs_ch, t_epoch[:epochs_ch.shape[1]])
            norm_co = baseline_pct_change(epochs_co, t_epoch[:epochs_co.shape[1]])

            # grand-average pool
            grand_change_pool.append(norm_ch.mean(axis=0))
            grand_cont_pool.append(norm_co.mean(axis=0))

            # theta change index (response window)
            t_ax = t_epoch[:norm_ch.shape[1]]
            r_ch = resp_mean(norm_ch, t_ax)
            r_co = resp_mean(norm_co, t_ax)
            rec['theta_ci'] = change_index(r_ch, r_co)
            rec['theta_resp_change'] = r_ch
            rec['theta_resp_cont'] = r_co

            n_hip = rec['n_hip_ch']
            theta_ci_str = f"theta_CI={rec['theta_ci']:.3f}"
            sac_ci_str = f"sac_CI={rec['saccade_ci']:.3f}"
            print(f'OK  ({n_hip} hip ch,  {theta_ci_str},  {sac_ci_str})')

    except Exception as e:
        print(f'ERROR: {e}')

    session_records.append(rec)

res_df = pd.DataFrame(session_records)
res_df.to_csv(os.path.join(OUT_RES_DIR, 'session_results.csv'), index=False)
print(f'\nSaved session_results.csv  ({len(res_df)} sessions)')

theta_valid = res_df.dropna(subset=['theta_ci'])
sac_valid = res_df.dropna(subset=['saccade_ci'])
both_valid = res_df.dropna(subset=['theta_ci', 'saccade_ci'])
mem_theta = res_df.dropna(subset=['theta_ci', 'auc'])
mem_sac = res_df.dropna(subset=['saccade_ci', 'auc'])

print(f'\nSessions with theta data: {len(theta_valid)}')
print(f'Sessions with saccade data: {len(sac_valid)}')
print(f'Sessions with both: {len(both_valid)}')


def permutation_test_mean(values, n_perm=N_PERM):
    """sign-flip permutation test against H0=mean==0, returns (obs, null, p)"""
    obs = np.mean(values)
    null = np.array([np.mean(values * np.random.choice([-1, 1], size=len(values)))
                     for _ in range(n_perm)])
    p = np.mean(np.abs(null) >= np.abs(obs))
    return obs, null, p


print('group-level statistics')

# theta CI vs 0
if len(theta_valid) >= 5:
    t_ci = theta_valid['theta_ci'].values
    obs_t, null_t, p_t = permutation_test_mean(t_ci)
    t_stat, p_t_param = stats.ttest_1samp(t_ci, 0)
    d_theta = obs_t / t_ci.std()
    print(f'\nTheta change index (n={len(theta_valid)} sessions):')
    print(f'  mean +/- sem: {obs_t:.4f} +/- {t_ci.std()/np.sqrt(len(t_ci)):.4f}')
    print(f'  permutation: p={p_t:.4f}')
    print(f'  t-test: t={t_stat:.3f}, p={p_t_param:.4f}')
    print(f"  cohen's d: {d_theta:.3f}")
else:
    p_t, obs_t, d_theta = np.nan, np.nan, np.nan
    print('not enough sessions for theta stats')

# saccade amplitude CI vs 0
if len(sac_valid) >= 5:
    s_ci = sac_valid['saccade_ci'].values
    obs_s, null_s, p_s = permutation_test_mean(s_ci)
    t_stat_s, p_s_param = stats.ttest_1samp(s_ci, 0)
    d_sac = obs_s / s_ci.std()
    print(f'\nSaccade amplitude change index (n={len(sac_valid)} sessions):')
    print(f'  mean +/- sem: {obs_s:.4f} +/- {s_ci.std()/np.sqrt(len(s_ci)):.4f}')
    print(f'  permutation: p={p_s:.4f}')
    print(f'  t-test: t={t_stat_s:.3f}, p={p_s_param:.4f}')
    print(f"  cohen's d: {d_sac:.3f}")
else:
    p_s, obs_s, d_sac = np.nan, np.nan, np.nan
    print('not enough sessions for saccade stats')

# theta CI vs saccade CI
if len(both_valid) >= 5:
    r_ts, p_ts = stats.pearsonr(both_valid['theta_ci'], both_valid['saccade_ci'])
    print(f'\nTheta CI vs Saccade CI (n={len(both_valid)}): r={r_ts:.3f}, p={p_ts:.4f}')
else:
    r_ts, p_ts = np.nan, np.nan

# memory prediction
if len(mem_theta) >= 5:
    r_tm, p_tm = stats.pearsonr(mem_theta['theta_ci'], mem_theta['auc'])
    print(f'\nTheta CI vs Memory AUC (n={len(mem_theta)}): r={r_tm:.3f}, p={p_tm:.4f}')
else:
    r_tm, p_tm = np.nan, np.nan

if len(mem_sac) >= 5:
    r_sm, p_sm = stats.pearsonr(mem_sac['saccade_ci'], mem_sac['auc'])
    print(f'Saccade CI vs Memory AUC (n={len(mem_sac)}): r={r_sm:.3f}, p={p_sm:.4f}')
else:
    r_sm, p_sm = np.nan, np.nan

# multiple regression: theta + saccade predicting memory
mem_both = res_df.dropna(subset=['theta_ci', 'saccade_ci', 'auc'])
if len(mem_both) >= 6:
    X = np.column_stack([mem_both['theta_ci'].values,
                         mem_both['saccade_ci'].values,
                         np.ones(len(mem_both))])
    y = mem_both['auc'].values
    coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print(f'\nMultiple regression (theta CI + saccade CI predicting AUC, n={len(mem_both)}):')
    print(f'  R2 = {r2:.3f}')
    print(f'  beta theta = {coef[0]:.4f}')
    print(f'  beta saccade = {coef[1]:.4f}')
else:
    r2 = np.nan

print('plotting theta epoch...')

if grand_change_pool and grand_cont_pool:
    ga_ch = np.array(grand_change_pool)   # (n_sessions, n_samples)
    ga_co = np.array(grand_cont_pool)

    # align to shortest epoch in case of rounding differences
    min_len = min(ga_ch.shape[1], ga_co.shape[1], len(t_epoch))
    ga_ch = ga_ch[:, :min_len]
    ga_co = ga_co[:, :min_len]
    t_plot = t_epoch[:min_len]

    mean_ch = ga_ch.mean(axis=0)
    sem_ch = ga_ch.std(axis=0) / np.sqrt(ga_ch.shape[0])
    mean_co = ga_co.mean(axis=0)
    sem_co = ga_co.std(axis=0) / np.sqrt(ga_co.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f'Hippocampal Theta at Scene Cuts\n'
        f'(n={ga_ch.shape[0]} sessions with hippocampal LFP)',
        fontsize=10, fontweight='bold'
    )

    # left: both conditions
    ax = axes[0]
    ax.plot(t_plot, mean_ch, color='#377eb8', lw=2, label='scene change')
    ax.fill_between(t_plot, mean_ch - sem_ch, mean_ch + sem_ch,
                    color='#377eb8', alpha=0.25)
    ax.plot(t_plot, mean_co, color='#377eb8', lw=1.2, ls='--', alpha=0.6,
            label='continuity cut')
    ax.fill_between(t_plot, mean_co - sem_co, mean_co + sem_co,
                    color='#377eb8', alpha=0.1)
    ax.axvline(0, color='k', lw=0.8, ls=':', label='cut onset')
    ax.axvspan(*RESP_WIN, color='gold', alpha=0.1, zorder=0, label='response win')
    ax.set_xlabel('Time relative to cut onset (s)')
    ax.set_ylabel('Theta amplitude (% change from baseline)')
    ax.set_title('Scene change vs. continuity cut')
    ax.legend(fontsize=7, frameon=False)

    # right: difference trace
    diff = mean_ch - mean_co
    sem_diff = np.sqrt(sem_ch**2 + sem_co**2)
    ax = axes[1]
    ax.plot(t_plot, diff, color='#984ea3', lw=2)
    ax.fill_between(t_plot, diff - sem_diff, diff + sem_diff,
                    color='#984ea3', alpha=0.2)
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.8, ls=':')
    ax.axvspan(*RESP_WIN, color='gold', alpha=0.1, zorder=0)
    if not np.isnan(p_t):
        ax.set_title(f'Difference (change - cont)\n'
                     f'Theta CI={obs_t:.3f}, p={p_t:.4f} (perm., n_sess={len(theta_valid)})')
    else:
        ax.set_title('Difference (change - cont)')
    ax.set_xlabel('Time relative to cut onset (s)')
    ax.set_ylabel('delta theta amplitude (% change)')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_FIG_DIR, 'fig1_theta_epoch.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig1_theta_epoch')
else:
    print('  Skipped (no sessions with hippocampal LFP)')


print('plotting change index distributions...')

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
fig.suptitle('Scene-Cut Change Indices across Sessions\n'
             '(positive = higher at scene changes than continuity cuts)',
             fontsize=10, fontweight='bold')

for ax, (col, label, color, ci_vals, p_val, d_val) in zip(axes, [
    ('theta_ci',   'Hippocampal\nTheta CI',         '#377eb8',
     theta_valid['theta_ci'].values if len(theta_valid) >= 5 else np.array([]),
     p_t, d_theta),
    ('saccade_ci', 'Saccade Amplitude CI\n(deg)',    '#ff7f00',
     sac_valid['saccade_ci'].values if len(sac_valid) >= 5 else np.array([]),
     p_s, d_sac),
]):
    if len(ci_vals) == 0:
        ax.set_title(label + '\n(insufficient data)')
        continue

    x = np.zeros(len(ci_vals))
    jit = np.random.normal(0, 0.05, len(ci_vals))
    ax.scatter(x + jit, ci_vals, color=color, s=40, alpha=0.7, zorder=3,
               edgecolors='white', lw=0.5)
    mu = ci_vals.mean()
    sem = ci_vals.std() / np.sqrt(len(ci_vals))
    ax.errorbar([0], [mu], yerr=[sem], fmt='_', color='k',
                markersize=18, lw=2, capsize=6, zorder=4)
    ax.axhline(0, color='k', lw=0.8, ls='--')

    pstr = f'p={p_val:.4f}' if not np.isnan(p_val) and p_val >= 0.0001 else 'p<0.0001' if not np.isnan(p_val) else 'p=n/a'
    d_str = f"d={d_val:.2f}" if not np.isnan(d_val) else ''
    n_str = f'n={len(ci_vals)} sessions'
    ax.set_title(f'{label}\n{pstr}, {d_str}\n{n_str}',
                 fontsize=9)
    ax.set_ylabel('Change index')
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)

plt.tight_layout()
fig.savefig(os.path.join(OUT_FIG_DIR, 'fig2_change_indices.png'),
            dpi=200, bbox_inches='tight')
plt.close(fig)
print('  Saved fig2_change_indices')


print('plotting theta vs saccade CI...')

fig, ax = plt.subplots(figsize=(5, 4.5))
if len(both_valid) >= 5:
    x_vals = both_valid['theta_ci'].values
    y_vals = both_valid['saccade_ci'].values

    ax.scatter(x_vals, y_vals, s=55, alpha=0.8, color='#5e4fa2',
               edgecolors='white', lw=0.5, zorder=3)

    # regression line
    m, b = np.polyfit(x_vals, y_vals, 1)
    xline = np.linspace(x_vals.min(), x_vals.max(), 50)
    ax.plot(xline, m * xline + b, color='#5e4fa2', lw=1.5, ls='--', alpha=0.7)

    pstr_ts = f'p={p_ts:.3f}' if p_ts >= 0.001 else 'p<0.001'
    ax.set_title(f'Theta CI vs. Saccade Amplitude CI\n'
                 f'r={r_ts:.2f}, {pstr_ts}  (n={len(both_valid)} sessions)',
                 fontsize=9)
else:
    ax.set_title('Insufficient data for correlation')

ax.axhline(0, color='k', lw=0.7, ls=':')
ax.axvline(0, color='k', lw=0.7, ls=':')
ax.set_xlabel('Hippocampal theta change index')
ax.set_ylabel('Saccade amplitude change index')
plt.tight_layout()
fig.savefig(os.path.join(OUT_FIG_DIR, 'fig3_theta_saccade.png'),
            dpi=200, bbox_inches='tight')
plt.close(fig)
print('  Saved fig3_theta_saccade')


print('plotting memory prediction...')

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle('Scene Boundary Neural / Oculomotor Responses and Recognition Memory',
             fontsize=10, fontweight='bold')

for ax, (df_sub, xcol, xlabel, color, r_val, p_val) in zip(axes, [
    (mem_theta, 'theta_ci', 'Hippocampal theta CI', '#377eb8', r_tm, p_tm),
    (mem_sac, 'saccade_ci', 'Saccade amplitude CI', '#ff7f00', r_sm, p_sm),
]):
    if len(df_sub) >= 5:
        x_v = df_sub[xcol].values
        y_v = df_sub['auc'].values
        ax.scatter(x_v, y_v, s=45, alpha=0.8, color=color,
                   edgecolors='white', lw=0.5, zorder=3)
        m, b = np.polyfit(x_v, y_v, 1)
        xline = np.linspace(x_v.min(), x_v.max(), 50)
        ax.plot(xline, m * xline + b, color=color, lw=1.5, ls='--', alpha=0.7)
        pstr = f'p={p_val:.3f}' if not np.isnan(p_val) and p_val >= 0.001 else \
               'p<0.001' if not np.isnan(p_val) else 'p=n/a'
        bold = 'bold' if (not np.isnan(p_val) and p_val < 0.05) else 'normal'
        ax.set_title(f'{xlabel}\nr={r_val:.2f}, {pstr}  (n={len(df_sub)})',
                     fontsize=9, fontweight=bold)
    else:
        ax.set_title(f'{xlabel}\n(insufficient data)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Recognition AUC')
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color='k', lw=0.6, ls=':', alpha=0.5, label='chance')

plt.tight_layout()
fig.savefig(os.path.join(OUT_FIG_DIR, 'fig4_memory.png'),
            dpi=200, bbox_inches='tight')
plt.close(fig)
print('  Saved fig4_memory')


print('plotting per-session theta CI...')

if len(theta_valid) >= 3:
    sorted_df = theta_valid.sort_values('theta_ci')
    subj_lbls = [s.replace('sub-', '') for s in sorted_df['subject']]
    ci_vals = sorted_df['theta_ci'].values
    colors = ['#377eb8' if v < 0 else '#e41a1c' for v in ci_vals]

    fig, ax = plt.subplots(figsize=(max(6, len(ci_vals) * 0.45), 4))
    ax.bar(np.arange(len(ci_vals)), ci_vals, color=colors, alpha=0.75, zorder=3)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(np.arange(len(ci_vals)))
    ax.set_xticklabels(subj_lbls, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Theta change index')
    ax.set_title(f'Hippocampal Theta CI per Session\n'
                 f'(blue = suppression at scene changes, red = enhancement)\n'
                 f'Mean={obs_t:.3f}, p={p_t:.4f}' if not np.isnan(p_t) else f'Mean={obs_t:.3f}',
                 fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_FIG_DIR, 'fig5_per_session_theta.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig5_per_session_theta')

