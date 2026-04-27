# population PSTH and change index analysis at scene boundaries
# three things in this script: (1) firing rate PSTH per brain region aligned to scene cuts
# (change vs continuity), (2) same thing but split by whether there's an emotional face
# in the post-cut frames - mainly looking at amygdala here, (3) correlate the boundary
# response with recognition AUC across subjects. part 3 came out null.
# the face emotion labeling uses the short_faceannots.pkl from the BIDS assets folder.
# run with: python event_boundary_analysis.py --nwb_dir E:/000623
#            --scenecuts .../scenecut_info.csv --faceannots .../short_faceannots.pkl

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
from sklearn.metrics import roc_auc_score
from pynwb import NWBHDF5IO

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


PSTH_BIN = 0.05          # 50 ms bins
PSTH_PRE = 1.0           # 1 s before cut
PSTH_POST = 2.0          # 2 s after cut
EMOTION_POST_FRAMES = 25  # 1s of frames at 25 Hz - could go longer but cuts are dense
AREAS_ORDER = ['amygdala', 'hippocampus', 'ACC', 'preSMA', 'vmPFC']
AREA_COLORS = {
    'amygdala': '#e41a1c',
    'hippocampus': '#377eb8',
    'ACC': '#4daf4a',
    'preSMA': '#984ea3',
    'vmPFC': '#ff7f00',
}


def strip_hemisphere(area_str):
    for prefix in ('Left ', 'Right '):
        if area_str.startswith(prefix):
            return area_str[len(prefix):]
    return area_str


def compute_psth(spike_times, event_times, pre, post, bin_size):
    """psth for one neuron across all events, returns (bin_centers, mean_rate, sem_rate, trial_rates)"""
    edges = np.arange(-pre, post + bin_size, bin_size)
    bin_centers = edges[:-1] + bin_size / 2
    n_bins = len(bin_centers)
    trial_rates = np.zeros((len(event_times), n_bins))
    for i, t0 in enumerate(event_times):
        rel = spike_times - t0
        counts, _ = np.histogram(rel, bins=edges)
        trial_rates[i] = counts / bin_size   # Hz
    mean_r = trial_rates.mean(axis=0)
    sem_r = trial_rates.std(axis=0) / np.sqrt(len(event_times))
    return bin_centers, mean_r, sem_r, trial_rates


def compute_auc(trials_df):
    """recognition AUC from trials table, recognition phase only"""
    recog = trials_df[trials_df['stim_phase'] == 'recognition'].copy()
    if len(recog) == 0:
        return np.nan
    y_true = recog['stimulus_file'].str.startswith('old').astype(int).values  # 1=old, 0=new
    y_score = recog['actual_response'].values
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def get_dominant_emotion(face_annots, frame_start, n_frames=25):
    """most common emotion label in the n_frames after frame_start, returns 'neutral' if none found"""
    emotion_counts = {}
    for fr in range(int(frame_start), int(frame_start) + n_frames):
        key = f'frame_{fr}'
        if key not in face_annots:
            continue
        for person, info in face_annots[key].items():
            emo = info.get('emotion', 'neutral')
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
    if not emotion_counts:
        return 'neutral'
    return max(emotion_counts, key=emotion_counts.get)


def label_cut_emotions(cuts_df, face_annots, n_frames=25):
    """assign dominant emotion to each cut based on post-cut frames"""
    emotions = []
    for _, row in cuts_df.iterrows():
        frame_start = int(row['shot_start_fr'])
        emo = get_dominant_emotion(face_annots, frame_start, n_frames)
        emotions.append(emo)
    return emotions


def main(nwb_dir, scenecuts_file, faceannots_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # load annotations
    cuts_df = pd.read_csv(scenecuts_file).reset_index(drop=True)

    # scene change = first shot of a new scene_id
    new_scenes = np.where(np.diff(cuts_df['scene_id']))[0] + 1
    is_change = np.zeros(len(cuts_df), dtype=bool)
    is_change[0] = True
    is_change[new_scenes] = True
    cuts_df['is_change'] = is_change
    cuts_df['cut_label'] = np.where(is_change, 'change', 'cont')

    with open(faceannots_file, 'rb') as f:
        face_annots = pickle.load(f)

    # emotion label per cut
    cuts_df['emotion'] = label_cut_emotions(cuts_df, face_annots, n_frames=EMOTION_POST_FRAMES)
    # binary: emotional vs neutral
    cuts_df['has_emotion'] = cuts_df['emotion'] != 'neutral'

    nwb_files = sorted(glob(os.path.join(nwb_dir, 'sub-*/*.nwb')))
    print(f'Found {len(nwb_files)} NWB files')

    # storage across sessions
    # psth per area
    psth_data = {area: {'change': [], 'cont': []} for area in AREAS_ORDER}

    # emotion split per area
    emo_data = {area: {'emotional': [], 'neutral': []} for area in AREAS_ORDER}

    # per-subject memory and neural
    subj_memory = {}   # subj_id -> AUC
    subj_neural = {}   # subj_id -> {area: mean_change_response}

    bins = np.arange(-PSTH_PRE, PSTH_POST + PSTH_BIN, PSTH_BIN)
    bin_centers = bins[:-1] + PSTH_BIN / 2

    for nwb_path in nwb_files:
        subj_id = os.path.basename(nwb_path).split('_')[0]  # e.g. sub-CS41
        print(f'  Processing {os.path.basename(nwb_path)} ...')

        with NWBHDF5IO(nwb_path, 'r') as io:
            nwb = io.read()
            units_df = nwb.units.to_dataframe()
            elec_df = nwb.electrodes.to_dataframe()
            trials_df = nwb.trials.to_dataframe()

            # frame to timestamp lookup
            frame_times = np.column_stack((
                nwb.stimulus['movieframe_time'].data[:],
                nwb.stimulus['movieframe_time'].timestamps[:]
            )).astype(float)

        # cut times in this session
        cut_times = frame_times[cuts_df['shot_start_fr'].to_numpy(int) - 1, 1]

        change_times = cut_times[cuts_df['is_change'].values]
        cont_times = cut_times[~cuts_df['is_change'].values]
        emo_times = cut_times[cuts_df['has_emotion'].values]
        neu_times = cut_times[~cuts_df['has_emotion'].values]

        enc_start = trials_df[trials_df['stim_phase'] == 'encoding']['start_time'].values[0]
        enc_stop = trials_df[trials_df['stim_phase'] == 'encoding']['stop_time'].values[0]

        # auc for this subject
        auc = compute_auc(trials_df)
        if not np.isnan(auc):
            subj_memory[subj_id] = auc

        if subj_id not in subj_neural:
            subj_neural[subj_id] = {area: [] for area in AREAS_ORDER}

        for _, unit in units_df.iterrows():
            sp = unit['spike_times']
            # keep only spikes during encoding
            sp = sp[(sp >= enc_start) & (sp <= enc_stop)]
            if len(sp) < 10:  # 10 spike min - too low and the PSTH is just noise
                continue

            area_raw = elec_df.iloc[unit['electrode_id']]['location']
            area = strip_hemisphere(area_raw)
            if area not in AREAS_ORDER:
                continue

            # psth
            _, _, _, tr_change = compute_psth(sp, change_times, PSTH_PRE, PSTH_POST, PSTH_BIN)
            _, _, _, tr_cont = compute_psth(sp, cont_times, PSTH_PRE, PSTH_POST, PSTH_BIN)
            psth_data[area]['change'].append(tr_change.mean(axis=0))
            psth_data[area]['cont'].append(tr_cont.mean(axis=0))

            # emotion modulation
            _, _, _, tr_emo = compute_psth(sp, emo_times, PSTH_PRE, PSTH_POST, PSTH_BIN)
            _, _, _, tr_neu = compute_psth(sp, neu_times, PSTH_PRE, PSTH_POST, PSTH_BIN)
            emo_data[area]['emotional'].append(tr_emo.mean(axis=0))
            emo_data[area]['neutral'].append(tr_neu.mean(axis=0))

            # memory - mean firing in 0-1s post scene change
            post_bins = (bin_centers >= 0) & (bin_centers <= 1.0)
            mean_resp = tr_change.mean(axis=0)[post_bins].mean()
            subj_neural[subj_id][area].append(mean_resp)

    # fig 1: population PSTH
    print('plotting psth...')
    fig, axes = plt.subplots(1, len(AREAS_ORDER), figsize=(14, 3), sharey=False)
    fig.suptitle('Population PSTH at Event Boundaries by Brain Region', fontsize=10, fontweight='bold')

    for ax, area in zip(axes, AREAS_ORDER):
        color = AREA_COLORS[area]
        for cond, ls, alpha_fill in [('change', '-', 0.3), ('cont', '--', 0.15)]:
            data = np.array(psth_data[area][cond])
            if len(data) == 0:
                continue
            mean_r = data.mean(axis=0)
            sem_r = data.std(axis=0) / np.sqrt(len(data))
            ax.plot(bin_centers, mean_r, color=color, ls=ls,
                    lw=1.5, label=cond)
            ax.fill_between(bin_centers, mean_r - sem_r, mean_r + sem_r,
                            color=color, alpha=alpha_fill)

        ax.axvline(0, color='k', lw=0.8, ls=':')
        ax.set_title(f'{area}\n(n={len(psth_data[area]["change"])} units)', fontsize=8)
        ax.set_xlabel('Time re. cut (s)')
        if ax == axes[0]:
            ax.set_ylabel('Firing rate (Hz)')
        ax.legend(fontsize=6, frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig1_population_psth.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig1_population_psth')

    # fig 2: emotion modulation
    print('plotting emotion modulation...')
    fig, axes = plt.subplots(1, len(AREAS_ORDER), figsize=(14, 3), sharey=False)
    fig.suptitle('Emotion-Modulated Neural Responses at Event Boundaries', fontsize=10, fontweight='bold')

    for ax, area in zip(axes, AREAS_ORDER):
        color = AREA_COLORS[area]
        for cond, ls, alpha_fill in [('emotional', '-', 0.3), ('neutral', '--', 0.15)]:
            data = np.array(emo_data[area][cond])
            if len(data) == 0:
                continue
            mean_r = data.mean(axis=0)
            sem_r = data.std(axis=0) / np.sqrt(len(data))
            ax.plot(bin_centers, mean_r, color=color, ls=ls,
                    lw=1.5, label=cond)
            ax.fill_between(bin_centers, mean_r - sem_r, mean_r + sem_r,
                            color=color, alpha=alpha_fill)

        ax.axvline(0, color='k', lw=0.8, ls=':')
        ax.set_title(f'{area}\n(n={len(emo_data[area]["emotional"])} units)', fontsize=8)
        ax.set_xlabel('Time re. cut (s)')
        if ax == axes[0]:
            ax.set_ylabel('Firing rate (Hz)')
        ax.legend(fontsize=6, frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2_emotion_modulation.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig2_emotion_modulation')

    # fig 2b: amygdala per emotion - requires second pass through NWB files
    print('plotting amygdala per-emotion...')
    emotion_cats = ['afraid', 'angry', 'happy', 'surprised', 'neutral']
    emo_cat_data = {e: [] for e in emotion_cats}

    for nwb_path in nwb_files:
        with NWBHDF5IO(nwb_path, 'r') as io:
            nwb = io.read()
            units_df = nwb.units.to_dataframe()
            elec_df = nwb.electrodes.to_dataframe()
            trials_df = nwb.trials.to_dataframe()
            frame_times = np.column_stack((
                nwb.stimulus['movieframe_time'].data[:],
                nwb.stimulus['movieframe_time'].timestamps[:]
            )).astype(float)

        enc_start = trials_df[trials_df['stim_phase'] == 'encoding']['start_time'].values[0]
        enc_stop = trials_df[trials_df['stim_phase'] == 'encoding']['stop_time'].values[0]
        cut_times = frame_times[cuts_df['shot_start_fr'].to_numpy(int) - 1, 1]

        for _, unit in units_df.iterrows():
            sp = unit['spike_times']
            sp = sp[(sp >= enc_start) & (sp <= enc_stop)]
            if len(sp) < 10:
                continue
            area_raw = elec_df.iloc[unit['electrode_id']]['location']
            area = strip_hemisphere(area_raw)
            if area != 'amygdala':
                continue
            for emo in emotion_cats:
                emo_mask = cuts_df['emotion'].values == emo
                times_emo = cut_times[emo_mask]
                if len(times_emo) == 0:
                    continue
                _, _, _, tr = compute_psth(sp, times_emo, PSTH_PRE, PSTH_POST, PSTH_BIN)
                emo_cat_data[emo].append(tr.mean(axis=0))

    emo_colors = {
        'afraid': '#d73027',
        'angry': '#fc8d59',
        'happy': '#1a9850',
        'surprised': '#756bb1',
        'neutral': '#969696',
    }

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for emo in emotion_cats:
        data = np.array(emo_cat_data[emo])
        if len(data) == 0:
            continue
        mean_r = data.mean(axis=0)
        sem_r = data.std(axis=0) / np.sqrt(len(data))
        n_cuts = int(cuts_df['emotion'].eq(emo).sum())
        ax.plot(bin_centers, mean_r, color=emo_colors[emo],
                lw=1.5, label=f'{emo} (n={n_cuts} cuts)')
        ax.fill_between(bin_centers, mean_r - sem_r, mean_r + sem_r,
                        color=emo_colors[emo], alpha=0.15)

    ax.axvline(0, color='k', lw=0.8, ls=':')
    ax.set_xlabel('Time re. cut (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title('Amygdala neurons: response by emotion at boundary', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2b_amygdala_per_emotion.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig2b_amygdala_per_emotion')

    # fig 3: memory prediction - this came out null for all regions
    print('plotting memory prediction...')

    fig, axes = plt.subplots(1, len(AREAS_ORDER), figsize=(14, 3.5))
    fig.suptitle('Event Boundary Neural Response vs. Recognition Memory (AUC)', fontsize=10, fontweight='bold')

    for ax, area in zip(axes, AREAS_ORDER):
        color = AREA_COLORS[area]
        x_vals, y_vals, subj_labels = [], [], []

        for subj, auc in subj_memory.items():
            neural_vals = subj_neural[subj][area]
            if len(neural_vals) == 0:
                continue
            x_vals.append(np.mean(neural_vals))
            y_vals.append(auc)
            subj_labels.append(subj)

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        ax.scatter(x_vals, y_vals, color=color, s=30, zorder=3, alpha=0.8)

        if len(x_vals) >= 4:
            r, p = stats.pearsonr(x_vals, y_vals)
            m, b = np.polyfit(x_vals, y_vals, 1)
            xline = np.linspace(x_vals.min(), x_vals.max(), 50)
            ax.plot(xline, m * xline + b, color=color, lw=1.2, ls='--')
            pstr = f'p={p:.3f}' if p >= 0.001 else 'p<0.001'
            ax.set_title(f'{area}\nr={r:.2f}, {pstr}', fontsize=8)
        else:
            ax.set_title(f'{area}\n(n<4)', fontsize=8)

        ax.set_xlabel('Mean firing rate\nat scene changes (Hz)', fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel('Recognition AUC')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig3_memory_prediction.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig3_memory_prediction')

    # fig 4: change index bar chart per region
    print('plotting change index...')
    POST_WIN = (bin_centers >= 0) & (bin_centers <= 1.0)

    area_change_idx = {area: [] for area in AREAS_ORDER}

    for nwb_path in nwb_files:
        with NWBHDF5IO(nwb_path, 'r') as io:
            nwb = io.read()
            units_df = nwb.units.to_dataframe()
            elec_df = nwb.electrodes.to_dataframe()
            trials_df = nwb.trials.to_dataframe()
            frame_times = np.column_stack((
                nwb.stimulus['movieframe_time'].data[:],
                nwb.stimulus['movieframe_time'].timestamps[:]
            )).astype(float)

        enc_start = trials_df[trials_df['stim_phase'] == 'encoding']['start_time'].values[0]
        enc_stop = trials_df[trials_df['stim_phase'] == 'encoding']['stop_time'].values[0]
        cut_times = frame_times[cuts_df['shot_start_fr'].to_numpy(int) - 1, 1]
        change_times = cut_times[cuts_df['is_change'].values]
        cont_times = cut_times[~cuts_df['is_change'].values]

        for _, unit in units_df.iterrows():
            sp = unit['spike_times']
            sp = sp[(sp >= enc_start) & (sp <= enc_stop)]
            if len(sp) < 10:
                continue
            area_raw = elec_df.iloc[unit['electrode_id']]['location']
            area = strip_hemisphere(area_raw)
            if area not in AREAS_ORDER:
                continue

            _, _, _, tr_ch = compute_psth(sp, change_times, PSTH_PRE, PSTH_POST, PSTH_BIN)
            _, _, _, tr_co = compute_psth(sp, cont_times, PSTH_PRE, PSTH_POST, PSTH_BIN)

            r_ch = tr_ch[:, POST_WIN].mean()
            r_co = tr_co[:, POST_WIN].mean()
            denom = r_ch + r_co
            if denom > 0:
                ci = (r_ch - r_co) / denom
                area_change_idx[area].append(ci)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    means, sems, pvals = [], [], []
    for area in AREAS_ORDER:
        vals = np.array(area_change_idx[area])
        if len(vals) == 0:
            means.append(0); sems.append(0); pvals.append(1)
            continue
        means.append(vals.mean())
        sems.append(vals.std() / np.sqrt(len(vals)))
        _, p = stats.ttest_1samp(vals, 0)
        pvals.append(p)

    x = np.arange(len(AREAS_ORDER))
    bars = ax.bar(x, means, color=[AREA_COLORS[a] for a in AREAS_ORDER],
                  yerr=sems, capsize=4, error_kw={'lw': 1.2}, zorder=3)
    ax.axhline(0, color='k', lw=0.8)

    for xi, (m, s, p) in enumerate(zip(means, sems, pvals)):
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        # place label just beyond the end of the error bar
        if m >= 0:
            y_label = m + s + 0.005
            va = 'bottom'
        else:
            y_label = m - s - 0.005
            va = 'top'
        ax.text(xi, y_label, sig, ha='center', va=va, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(AREAS_ORDER, fontsize=8)
    ax.set_ylabel('Change index\n(change - cont) / (change + cont)')
    ax.set_title('Scene change selectivity across brain regions', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig4_change_index_by_region.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig4_change_index_by_region')

    # save cache for replotting
    import pickle as _pickle
    cache = dict(
        bin_centers=bin_centers,
        psth_data=psth_data,
        emo_data=emo_data,
        emo_cat_data=emo_cat_data,
        subj_memory=subj_memory,
        subj_neural=subj_neural,
        cuts_df=cuts_df,
    )
    with open(os.path.join(out_dir, 'analysis_cache.pkl'), 'wb') as _f:
        _pickle.dump(cache, _f)
    print(f'Cache saved to {out_dir}/analysis_cache.pkl')


if __name__ == '__main__':
    # change to your path
    main(
        nwb_dir='E:/000623',
        scenecuts_file='E:/bmovie-release-NWB-BIDS/assets/annotations/scenecut_info.csv',
        faceannots_file='E:/bmovie-release-NWB-BIDS/assets/annotations/short_faceannots.pkl',
        out_dir='./event_boundary_figs',
    )
