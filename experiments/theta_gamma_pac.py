# theta-gamma PAC (Tort MI) at scene boundaries
# phase-amplitude coupling (PAC) measures whether the amplitude of high-frequency gamma
# (30-80 Hz) is modulated by the phase of lower-frequency theta (4-8 Hz).
# i use the modulation index from Tort et al. 2010 which bins gamma amplitude by theta
# phase and computes the KL divergence from a uniform distribution. higher MI = stronger
# coupling between theta phase and gamma amplitude.
# the question is whether this coupling is stronger at scene changes vs continuity cuts,
# which would suggest theta is actively organizing gamma activity at event boundaries.
# i also compute a sliding-window version to see how PAC evolves over time around cuts,
# and a comodulogram (MI across all phase/amplitude frequency pairs) to check if the
# effect is specific to theta-gamma or shows up at other frequency combinations.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import butter, filtfilt, hilbert
from scipy import stats
from pynwb import NWBHDF5IO

# change to your path
NWB_DIR = 'E:/000623'
SCENECUTS_CSV = 'E:/bmovie-release-NWB-BIDS/assets/annotations/scenecut_info.csv'
OUT_FIG_DIR = './figures'
OUT_RES_DIR = './results'
os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_RES_DIR, exist_ok=True)

LFP_FS = 1000      # Hz
THETA_LO, THETA_HI = 4, 8    # Hz, phase band
GAMMA_LO, GAMMA_HI = 30, 80    # Hz, amplitude band
FILTER_ORDER = 4
PRE_S = 1.0
POST_S = 2.0
N_BINS = 18            # phase bins for MI - maybe try more later
N_PERM = 5000
SLIDE_WIN_S = 0.5          # sliding window for time-resolved PAC
SLIDE_STEP_S = 0.1
np.random.seed(21)

# comodulogram ranges
PHASE_BANDS = [(lo, lo + 4) for lo in range(2, 20, 2)]   # 2-4, 4-6, ... 18-20 Hz
AMP_BANDS = [(lo, lo + 20) for lo in range(20, 160, 10)]  # 20-40, ... 140-160 Hz

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def bandpass(signal_1d, lo, hi, fs=LFP_FS, order=FILTER_ORDER):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, signal_1d)
    # tried sosfilt but filtfilt was simpler


def modulation_index(phase_signal, amp_signal, n_bins=N_BINS):
    """Tort 2010 modulation index, returns scalar MI in [0,1]"""
    phase = np.angle(hilbert(phase_signal))
    amp = np.abs(hilbert(amp_signal))

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    amp_bins = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (phase >= bin_edges[b]) & (phase < bin_edges[b + 1])
        amp_bins[b] = amp[mask].mean() if mask.sum() > 0 else 0.0

    if amp_bins.sum() < 1e-30:
        return 0.0
    p = amp_bins / amp_bins.sum()
    # normalize by log(n_bins) to get MI in [0,1] - took me a while to get this right
    q = np.ones(n_bins) / n_bins
    kl = np.sum(p * np.log((p + 1e-30) / q))
    mi = kl / np.log(n_bins)
    return float(np.clip(mi, 0, 1))


def pac_epoch(signal_1d, t_lfp, event_times, pre=PRE_S, post=POST_S, fs=LFP_FS,
              phase_lo=THETA_LO, phase_hi=THETA_HI,
              amp_lo=GAMMA_LO,   amp_hi=GAMMA_HI):
    """MI per epoch around event_times, returns one value per valid epoch"""
    n_pre = int(pre * fs)
    n_post = int(post * fs)
    mis = []
    for t0 in event_times:
        idx0 = int(np.searchsorted(t_lfp, t0))
        start = idx0 - n_pre
        stop = idx0 + n_post
        if start < 0 or stop > len(signal_1d):
            continue
        seg = signal_1d[start:stop]
        ph = bandpass(seg, phase_lo, phase_hi)
        am = bandpass(seg, amp_lo, amp_hi)
        mis.append(modulation_index(ph, am))
    return np.array(mis)


def pac_sliding(signal_1d, fs=LFP_FS, win_s=SLIDE_WIN_S, step_s=SLIDE_STEP_S,
                pre_s=PRE_S, post_s=POST_S,
                phase_lo=THETA_LO, phase_hi=THETA_HI,
                amp_lo=GAMMA_LO,   amp_hi=GAMMA_HI):
    """time-resolved PAC in sliding windows, returns (t_centres, MI_array)"""
    n_win = int(win_s * fs)
    n_step = int(step_s * fs)
    n_tot = int((pre_s + post_s) * fs)
    t_centres, mis = [], []
    start = 0
    while start + n_win <= n_tot:
        seg = signal_1d[start:start + n_win]
        ph = bandpass(seg, phase_lo, phase_hi)
        am = bandpass(seg, amp_lo, amp_hi)
        mis.append(modulation_index(ph, am))
        t_centres.append((start + n_win / 2) / fs - pre_s)
        start += n_step
    return np.array(t_centres), np.array(mis)


def main():
    sc = pd.read_csv(SCENECUTS_CSV).reset_index(drop=True)
    new_sc = np.where(np.diff(sc['scene_id']))[0] + 1
    is_ch = np.zeros(len(sc), dtype=bool)
    is_ch[0] = True
    is_ch[new_sc] = True
    ch_t = sc.loc[is_ch, 'shot_start_t'].values
    co_t = sc.loc[~is_ch, 'shot_start_t'].values
    nwb_files = sorted(glob(os.path.join(NWB_DIR, 'sub-*', '*.nwb')))

    rows = []
    all_tc_change = []   # time-resolved MI per session
    all_tc_cont = []
    como_mat_ch = np.zeros((len(PHASE_BANDS), len(AMP_BANDS)))
    como_mat_co = np.zeros_like(como_mat_ch)
    n_como = 0

    for fpath in nwb_files:
        sid = os.path.basename(fpath).replace('.nwb', '')
        print(f'Processing {sid} ...')
        try:
            with NWBHDF5IO(fpath, 'r', load_namespaces=True) as io:
                nwb = io.read()
                try:
                    lfp_obj = nwb.processing['ecephys']['LFP_macro'].electrical_series['ElectricalSeries']
                except (KeyError, AttributeError):
                    print(f'  {sid}: no LFP, skipping'); continue

                lfp_el = lfp_obj.electrodes.to_dataframe()
                hipp_mask = lfp_el['location'].str.contains('hippo|Hip|CA', case=False, na=False)
                hipp_idx = list(np.where(hipp_mask)[0])
                if not hipp_idx:
                    print(f'  {sid}: no hippocampal channels, skipping'); continue

                lfp_data = np.asarray(lfp_obj.data[:])
                n_lfp = lfp_data.shape[0]
                lfp_ts = (lfp_obj.starting_time or 0.0) + np.arange(n_lfp) / lfp_obj.rate
        except Exception as e:
            print(f'  {sid}: error: {e}'); continue

        ch_mis_resp, co_mis_resp = [], []
        tc_ch_all, tc_co_all = [], []

        for ci in hipp_idx[:3]:   # first 3 hipp channels per session
            raw = lfp_data[:, ci].astype(float)

            # MI per full epoch
            mis_ch = pac_epoch(raw, lfp_ts, ch_t)
            mis_co = pac_epoch(raw, lfp_ts, co_t)
            if len(mis_ch) == 0 or len(mis_co) == 0:
                continue

            ch_mis_resp.append(mis_ch.mean())
            co_mis_resp.append(mis_co.mean())

            # time-resolved PAC across epochs
            n_pre_samp = int(PRE_S * LFP_FS)
            n_post_samp = int(POST_S * LFP_FS)
            tc_ch_ep, tc_co_ep = [], []
            for ev_arr, store in [(ch_t, tc_ch_ep), (co_t, tc_co_ep)]:
                for t0 in ev_arr:
                    idx0 = int(np.searchsorted(lfp_ts, t0))
                    s, e = idx0 - n_pre_samp, idx0 + n_post_samp
                    if s < 0 or e > len(raw):
                        continue
                    t_c, mi_c = pac_sliding(raw[s:e])
                    store.append(mi_c)
            if tc_ch_ep:
                tc_ch_all.append(np.mean(tc_ch_ep, axis=0))
            if tc_co_ep:
                tc_co_all.append(np.mean(tc_co_ep, axis=0))

            # comodulogram, first channel only
            if ci == hipp_idx[0]:
                for pi, (plo, phi) in enumerate(PHASE_BANDS):
                    for ai, (alo, ahi) in enumerate(AMP_BANDS):
                        mis_tmp = pac_epoch(raw, lfp_ts, ch_t,
                                            phase_lo=plo, phase_hi=phi,
                                            amp_lo=alo, amp_hi=ahi)
                        mis_tmp2 = pac_epoch(raw, lfp_ts, co_t,
                                             phase_lo=plo, phase_hi=phi,
                                             amp_lo=alo, amp_hi=ahi)
                        if len(mis_tmp) and len(mis_tmp2):
                            como_mat_ch[pi, ai] += mis_tmp.mean()
                            como_mat_co[pi, ai] += mis_tmp2.mean()
                n_como += 1

        if not ch_mis_resp:
            continue

        mean_ch = np.mean(ch_mis_resp)
        mean_co = np.mean(co_mis_resp)
        ci_val = (mean_ch - mean_co) / (abs(mean_ch) + abs(mean_co) + 1e-30)
        rows.append({'session': sid, 'MI_change': mean_ch,
                     'MI_cont': mean_co, 'CI': ci_val})
        print(f'  {sid}: MI change={mean_ch:.4f}  cont={mean_co:.4f}  CI={ci_val:.3f}')

        if tc_ch_all:
            all_tc_change.append(np.mean(tc_ch_all, axis=0))
        if tc_co_all:
            all_tc_cont.append(np.mean(tc_co_all, axis=0))

    if not rows:
        print('No sessions completed.'); return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_RES_DIR, 'pac_results.csv'), index=False)

    ci_vals = df['CI'].values
    t_obs, p_ttest = stats.ttest_1samp(ci_vals, 0)
    null = np.array([np.mean(np.random.choice([-1, 1], size=len(ci_vals)) * ci_vals)
                     for _ in range(N_PERM)])
    p_perm = (np.abs(null) >= np.abs(ci_vals.mean())).mean()
    print(f'\nGroup MI CI: mean={ci_vals.mean():.3f}  t={t_obs:.2f}  '
          f'p_ttest={p_ttest:.4f}  p_perm={p_perm:.4f}')

    '''plot code from keles et al repo - Keles, U., Tran, T. T., Norcia, A. M., Quian Quiroga, R., & Rutishauser, U. (2024). A multimodal dataset of human neurophysiology during movie watching. Scientific Data, 11, 67.'''
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.scatter(df['MI_change'], df['MI_cont'], c='steelblue', s=50, zorder=3)
    lim = max(df[['MI_change', 'MI_cont']].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('PAC MI (Scene Change)', fontsize=8)
    ax.set_ylabel('PAC MI (Continuity Cut)', fontsize=8)
    ax.set_title('Theta-Gamma PAC per Session', fontsize=9)

    ax = axes[1]
    ax.bar(['Scene Change', 'Continuity Cut'],
           [df['MI_change'].mean(), df['MI_cont'].mean()],
           yerr=[df['MI_change'].sem(), df['MI_cont'].sem()],
           color=['#d62728', '#1f77b4'], capsize=4, width=0.5)
    ax.set_ylabel('Mean MI (+/- SEM)', fontsize=8)
    ax.set_title(f'Group PAC MI\np_perm = {p_perm:.3f}', fontsize=9)

    fig.suptitle(f'Theta-Gamma Phase-Amplitude Coupling at Scene Boundaries\n'
                 f'n = {len(df)} sessions',
                 fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_FIG_DIR, 'figS3_pac_change_index.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    if all_tc_change and all_tc_cont:
        t_slide, _ = pac_sliding(np.zeros(int((PRE_S + POST_S) * LFP_FS)))
        grand_tc_ch = np.mean(all_tc_change, axis=0)
        grand_tc_co = np.mean(all_tc_cont, axis=0)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(t_slide, grand_tc_ch, color='#d62728', lw=1.5, label='Scene Change')
        ax.plot(t_slide, grand_tc_co, color='#1f77b4', lw=1.5, label='Continuity Cut')
        ax.axvline(0, color='k', lw=1, ls='--')
        ax.fill_between(t_slide,
                        grand_tc_ch - np.std(all_tc_change, axis=0) / np.sqrt(len(all_tc_change)),
                        grand_tc_ch + np.std(all_tc_change, axis=0) / np.sqrt(len(all_tc_change)),
                        color='#d62728', alpha=0.2)
        ax.fill_between(t_slide,
                        grand_tc_co - np.std(all_tc_cont, axis=0) / np.sqrt(len(all_tc_cont)),
                        grand_tc_co + np.std(all_tc_cont, axis=0) / np.sqrt(len(all_tc_cont)),
                        color='#1f77b4', alpha=0.2)
        ax.set_xlabel('Time from cut (s)', fontsize=8)
        ax.set_ylabel('Modulation Index', fontsize=8)
        ax.set_title('Time-Resolved Theta-Gamma PAC', fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_FIG_DIR, 'figS4_pac_timecourse.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    if n_como > 0:
        como_diff = (como_mat_ch - como_mat_co) / n_como
        phase_centres = [(lo + hi) / 2 for lo, hi in PHASE_BANDS]
        amp_centres = [(lo + hi) / 2 for lo, hi in AMP_BANDS]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, mat, title in zip(axes,
                                   [como_mat_ch / n_como,
                                    como_mat_co / n_como,
                                    como_diff],
                                   ['Scene Change', 'Continuity Cut', 'Difference (Ch-Co)']):
            vmax = np.percentile(np.abs(mat), 95)
            im = ax.imshow(mat.T, aspect='auto', origin='lower',
                           extent=[phase_centres[0], phase_centres[-1],
                                   amp_centres[0], amp_centres[-1]],
                           cmap='hot' if title != 'Difference (Ch-Co)' else 'RdBu_r',
                           vmin=0 if title != 'Difference (Ch-Co)' else -vmax,
                           vmax=vmax)
            ax.set_xlabel('Phase Frequency (Hz)', fontsize=8)
            ax.set_ylabel('Amplitude Frequency (Hz)', fontsize=8)
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, label='MI', shrink=0.8)
        fig.suptitle('Phase-Frequency Comodulogram', fontsize=9, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_FIG_DIR, 'figS5_comodulogram.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f'\nSaved figures to {OUT_FIG_DIR}')
    print(f'Saved results to {OUT_RES_DIR}/pac_results.csv')


main()
