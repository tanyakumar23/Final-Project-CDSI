# morlet CWT spectrogram of hipp LFP at scene boundaries
# this computes a broadband time-frequency representation of hippocampal LFP
# around scene cuts, using morlet wavelets (40 log-spaced frequencies from 1-150 Hz).
# the reason for wavelets instead of just bandpassing is that i want to see what happens
# across the whole frequency range, not just theta. scipy removed morlet2 in a newer
# version so i reimplemented it manually using fftconvolve.
# for each session i epoch the power matrix around scene change and continuity cut times,
# baseline correct using the -0.5 to 0s window, then average across channels and sessions.
# significance is tested with a 2D cluster permutation test (Maris & Oostenveld 2007)
# which corrects for multiple comparisons across the time-frequency plane.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from glob import glob
from scipy.signal import butter, filtfilt, fftconvolve
from scipy import stats
from scipy.ndimage import label as ndlabel
from pynwb import NWBHDF5IO

from config import NWB_DIR, SCENECUTS_CSV
OUT_FIG_DIR = './figures'
OUT_RES_DIR = './results'
os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_RES_DIR, exist_ok=True)

LFP_FS = 1000          # Hz
PRE_S = 1.0
POST_S = 2.0
BL_WIN = (-0.5, 0.0)   # baseline window (s) - seems standard
N_PERM = 1000
np.random.seed(1)

# 40 log-spaced freqs, 1-150 Hz
FREQS = np.logspace(np.log10(1), np.log10(150), 40)
W0 = 6.0   # Morlet wavelet central frequency (cycles)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def _morlet_wav(M, w0, s):
    # morlet2 was removed from scipy so doing it manually
    x = (np.arange(M) - (M - 1) / 2.0) / s
    return np.exp(1j * w0 * x) * np.exp(-0.5 * x ** 2) * np.pi ** (-0.25) / np.sqrt(s)


def morlet_power(signal_1d, freqs=FREQS, fs=LFP_FS, w0=W0):
    """returns power (n_freqs, n_samples) via morlet CWT"""
    power = np.zeros((len(freqs), len(signal_1d)))
    for i, f in enumerate(freqs):
        s = w0 * fs / (2 * np.pi * f)   # scale
        M = int(10 * s) | 1             # odd wavelet length, revisit if edge artefacts show up
        wav = _morlet_wav(M, w0, s)
        coef = fftconvolve(signal_1d, wav[::-1].conj(), mode='same')
        power[i] = np.abs(coef) ** 2
    return power


def epoch_tfr(power_2d, t_lfp, event_times, pre=PRE_S, post=POST_S, fs=LFP_FS):
    """cut power matrix into epochs around events, returns array or None"""
    n_pre = int(pre * fs)
    n_post = int(post * fs)
    epochs = []
    for t0 in event_times:
        idx0 = int(np.searchsorted(t_lfp, t0))
        start = idx0 - n_pre
        stop = idx0 + n_post
        if start < 0 or stop > power_2d.shape[1]:
            continue
        epochs.append(power_2d[:, start:stop])
    return np.array(epochs) if epochs else None   # (n, n_freqs, n_win)


def baseline_norm_tfr(epochs, t_axis, bl_win=BL_WIN):
    """percent change baseline normalization for TFR epochs"""
    bl_mask = (t_axis >= bl_win[0]) & (t_axis < bl_win[1])
    bl_mean = epochs[:, :, bl_mask].mean(axis=2, keepdims=True)  # (n,nf,1)
    bl_mean = np.where(bl_mean < 1e-20, 1e-20, bl_mean)
    return (epochs - bl_mean) / bl_mean * 100.0   # percent change


def cluster_perm_test(map_a, map_b, n_perm=N_PERM, threshold_p=0.05):
    """nonparametric cluster permutation test across sessions"""
    n = map_a.shape[0]
    diff = map_a - map_b   # (n_sess, nf, nt)
    t_obs = diff.mean(0) / (diff.std(0) / np.sqrt(n) + 1e-30)  # (nf, nt)

    # threshold t-map
    t_thr = stats.t.ppf(1 - threshold_p / 2, df=n - 1)
    pos = t_obs > t_thr
    neg = t_obs < -t_thr

    def max_cluster_stat(binary_map, t_map):
        labeled, n_clust = ndlabel(binary_map)
        if n_clust == 0:
            return 0.0
        return max(t_map[labeled == k].sum() for k in range(1, n_clust + 1))

    obs_stat = max(max_cluster_stat(pos, t_obs),
                   abs(max_cluster_stat(neg, -t_obs)))

    null_stats = []
    pooled = np.concatenate([map_a, map_b], axis=0)
    for _ in range(n_perm):
        idx = np.random.permutation(2 * n)
        p_a = pooled[idx[:n]]
        p_b = pooled[idx[n:]]
        d = p_a - p_b
        t_p = d.mean(0) / (d.std(0) / np.sqrt(n) + 1e-30)
        pos_p = t_p > t_thr
        neg_p = t_p < -t_thr
        null_stats.append(max(max_cluster_stat(pos_p, t_p),
                              abs(max_cluster_stat(neg_p, -t_p))))
    null_stats = np.array(null_stats)
    p_val = (null_stats >= obs_stat).mean()
    return t_obs, p_val, obs_stat


def main():
    sc = pd.read_csv(SCENECUTS_CSV).reset_index(drop=True)
    new_sc = np.where(np.diff(sc['scene_id']))[0] + 1
    is_ch = np.zeros(len(sc), dtype=bool)
    is_ch[0] = True
    is_ch[new_sc] = True
    ch_t = sc.loc[is_ch, 'shot_start_t'].values
    co_t = sc.loc[~is_ch, 'shot_start_t'].values
    nwb_files = sorted(glob(os.path.join(NWB_DIR, 'sub-*', '*.nwb')))

    t_axis = np.linspace(-PRE_S, POST_S, int((PRE_S + POST_S) * LFP_FS),
                          endpoint=False)

    session_change = []   # (n_freqs, n_time) per session
    session_cont = []
    session_ids = []

    for fpath in nwb_files:
        sid = os.path.basename(fpath).replace('.nwb', '')
        print(f'Processing {sid} ...')

        try:
            with NWBHDF5IO(fpath, 'r', load_namespaces=True) as io:
                nwb = io.read()

                # find hipp channels
                try:
                    lfp_obj = nwb.processing['ecephys']['LFP_macro'].electrical_series['ElectricalSeries']
                except (KeyError, AttributeError):
                    print(f'  {sid}: no LFP object found, skipping')
                    continue

                lfp_el = lfp_obj.electrodes.to_dataframe()
                hipp_mask = lfp_el['location'].str.contains('hippo|Hip|CA', case=False, na=False)
                hipp_idx = list(np.where(hipp_mask)[0])
                if not hipp_idx:
                    print(f'  {sid}: no hippocampal channels, skipping')
                    continue

                lfp_data = np.asarray(lfp_obj.data[:])
                n_lfp = lfp_data.shape[0]
                lfp_ts = (lfp_obj.starting_time or 0.0) + np.arange(n_lfp) / lfp_obj.rate

        except Exception as e:
            print(f'  {sid}: error reading NWB: {e}')
            continue

        # per-channel TFR
        ch_tfrs_change = []
        ch_tfrs_cont = []

        for ci in hipp_idx[:4]:   # limit to first 4 channels for speed
            raw_ch = lfp_data[:, ci].astype(float)

            # crude line-noise notch (60 Hz)
            b60, a60 = butter(2, [58 / 500, 62 / 500], btype='bandstop')
            raw_ch = filtfilt(b60, a60, raw_ch)
            # tried 120 Hz too but it didn't change anything so i dropped it

            print(f'  Computing CWT for channel {ci} ...')
            power = morlet_power(raw_ch)   # (n_freqs, n_samples) - this is slow, probably fine for now

            # epoch and bl-correct
            ep_ch = epoch_tfr(power, lfp_ts, ch_t)
            ep_co = epoch_tfr(power, lfp_ts, co_t)
            if ep_ch is None or ep_co is None:
                continue

            ep_ch_norm = baseline_norm_tfr(ep_ch, t_axis)
            ep_co_norm = baseline_norm_tfr(ep_co, t_axis)

            ch_tfrs_change.append(ep_ch_norm.mean(axis=0))   # (nf, nt)
            ch_tfrs_cont.append(ep_co_norm.mean(axis=0))

        if not ch_tfrs_change:
            continue

        session_change.append(np.mean(ch_tfrs_change, axis=0))  # (nf, nt)
        session_cont.append(np.mean(ch_tfrs_cont,   axis=0))
        session_ids.append(sid)
        print(f'  {sid}: done ({len(ch_tfrs_change)} hipp channels)')

    if not session_change:
        print('No sessions processed.')
        return

    session_change = np.array(session_change)   # (n_sess, nf, nt)
    session_cont = np.array(session_cont)

    grand_change = session_change.mean(0)   # (nf, nt)
    grand_cont = session_cont.mean(0)
    grand_diff = grand_change - grand_cont

    # cluster perm test
    print('Running cluster permutation test ...')
    t_obs, p_val, obs_stat = cluster_perm_test(session_change, session_cont)
    print(f'  Cluster permutation p = {p_val:.4f}  (obs_stat = {obs_stat:.3f})')

    # theta 4-8 Hz mean power in 0-1s window
    theta_mask = (FREQS >= 4) & (FREQS <= 8)
    t_resp = (t_axis >= 0) & (t_axis < 1.0)
    rows = []
    for i, sid in enumerate(session_ids):
        rows.append({
            'session': sid,
            'theta_pct_change': float(session_change[i][theta_mask][:, t_resp].mean()),
            'theta_pct_cont': float(session_cont[i][theta_mask][:, t_resp].mean()),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_RES_DIR, 'spectrogram_results.csv'), index=False)

    vmax = np.percentile(np.abs([grand_change, grand_cont]), 95)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, data, title in zip(axes,
                                [grand_change, grand_cont],
                                ['Scene Change', 'Continuity Cut']):
        im = ax.imshow(data, aspect='auto', origin='lower',
                       extent=[-PRE_S, POST_S, 0, len(FREQS)],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.axvline(0, color='k', lw=1.5, ls='--')
        ax.set_yticks(np.linspace(0, len(FREQS), 6))
        ax.set_yticklabels([f'{FREQS[int(np.clip(i, 0, len(FREQS)-1))]:.0f}'
                            for i in np.linspace(0, len(FREQS), 6)])
        ax.set_xlabel('Time from cut (s)', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.axhline(
            np.searchsorted(FREQS, 4), color='w', lw=0.8, ls=':', alpha=0.6)
        ax.axhline(
            np.searchsorted(FREQS, 8), color='w', lw=0.8, ls=':', alpha=0.6)
    axes[0].set_ylabel('Frequency (Hz)', fontsize=8)
    plt.colorbar(im, ax=axes[1], label='Power change (%)', shrink=0.8)
    fig.suptitle(
        f'Hippocampal LFP Spectrogram at Scene Boundaries\n'
        f'n = {len(session_ids)} sessions',
        fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_FIG_DIR, 'figS1_lfp_spectrogram.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    vd = np.percentile(np.abs(grand_diff), 95)
    im = ax.imshow(grand_diff, aspect='auto', origin='lower',
                   extent=[-PRE_S, POST_S, 0, len(FREQS)],
                   cmap='RdBu_r', vmin=-vd, vmax=vd)
    ax.axvline(0, color='k', lw=1.5, ls='--', label='cut onset')
    ax.set_yticks(np.linspace(0, len(FREQS), 6))
    ax.set_yticklabels([f'{FREQS[int(np.clip(i, 0, len(FREQS)-1))]:.0f}'
                        for i in np.linspace(0, len(FREQS), 6)])
    ax.set_xlabel('Time from cut (s)', fontsize=8)
    ax.set_ylabel('Frequency (Hz)', fontsize=8)
    ax.set_title(
        f'Scene Change - Continuity Cut  |  cluster p = {p_val:.3f}',
        fontsize=9)
    plt.colorbar(im, ax=ax, label='delta power (%)', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_FIG_DIR, 'figS2_spectrogram_diff.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'\nSaved figures to {OUT_FIG_DIR}')
    print(f'Saved results to {OUT_RES_DIR}/spectrogram_results.csv')


main()
