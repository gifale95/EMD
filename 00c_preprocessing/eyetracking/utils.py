def preprocess_eyetracking(args, session):
    """Preprocess the raw eyetracking data.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    session : str
        EEG recording session.

    Returns
    -------
    eyetracking_data : float
        Preprocessed eyetracking data.
    times : float
        Eyetracking time points.
    stimulus_id : list of int
        Stimulus condition IDs.
    run_number : list of int
        Run number of each trial of a given session (in the range [1, 16]).
    trial_number : list of int
        Number of each trial of a given session (in the range [1, 66]).

    """

    import os
    from glob import glob
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from scipy.signal import butter, filtfilt

    ### Get the eyetracking files ###
    eye_files = glob(os.path.join(args.emd_dir, f'sub-{args.subject:02}',
        f'ses-{session:02}', 'eeg', '*physio.tsv.gz'))
    eye_files.sort()

    ### Get the event files ###
    event_files = glob(os.path.join(args.emd_dir, f'sub-{args.subject:02}',
        f'ses-{session:02}', 'eeg', '*physioevents.tsv.gz'))
    event_files.sort()

    ### Loop across runs ###
    for f in tqdm(range(len(eye_files))):

        ### Import the eyelink samples ###
        sample_cols = ['timestamp', 'x_coordinate', 'y_coordinate', 'pupil_size']
        samples = pd.read_csv(eye_files[f], sep="\t", header=None,
            names=sample_cols, compression='gzip')
        # Round the sample time to 3 digits (ms precision)
        samples['timestamp'] = np.round(samples['timestamp'], 3)
        # Convert to float 32
        samples = samples.astype({'timestamp': np.float32,
            'x_coordinate': np.float32, 'y_coordinate': np.float32,
            'pupil_size': np.float32})
        # Set negative and zero values (blinks & artefacts) to NaNs
        for col in ['x_coordinate', 'y_coordinate', 'pupil_size']:
            samples.loc[samples[col] <= 0, col] = np.nan

        ### Import the events ###
        events = pd.read_table(event_files[f], compression='gzip')
        stim_idx = np.where(events['event_type'] == 'stimulus')[0]
        stim_onset = events['onset'][stim_idx].to_numpy().astype(np.float32)
        stim_id = events['stim_id'][stim_idx].to_numpy().astype(int)
        trial_num = events['trial_num'][stim_idx].to_numpy().astype(int)
        if args.subject == 4 and session == 5 and f+1 == 5:
            n_trials = 39
        elif args.subject == 4 and session == 6 and f+1 == 10:
            n_trials = 23
        elif args.subject == 4 and session == 6 and f+1 == 12:
            n_trials = 33
        elif args.subject == 5 and session == 6 and f+1 == 11:
            n_trials = 50
        else:
            n_trials = 66
        if len(stim_id) != n_trials:
            raise Exception('Incorrect number of eyelink events!')

        ### Pad eye blinks with NaNs ###
        # Import the eyeblinks
        blink_idx = np.where(events['event_type'] == 'blink')[0]
        blink_onset = events['onset'][blink_idx].to_numpy().astype(np.float32)
        blink_offset = (blink_onset + \
            events['duration'][blink_idx]).to_numpy().astype(np.float32)
        # Set to NaN samples 150 ms before and 150 ms after eye blinks
        pad = 0.15 # seconds
        blink_onset = np.round((blink_onset - pad), 3)
        blink_offset = np.round((blink_offset + pad), 3)
        for b in range(len(blink_onset)):
            if blink_onset[b] < samples['timestamp'].iloc[0]:
                idx_start = 0
            else:
                idx_start = np.where(samples['timestamp'] == blink_onset[b])[0][0]
            if blink_offset[b] > samples['timestamp'].iloc[-1]:
                idx_end = int(samples['timestamp'].iloc[-1])
            else:
                idx_end = np.where(samples['timestamp'] == blink_offset[b])[0][0]
            for col in ['x_coordinate', 'y_coordinate', 'pupil_size']:
                samples.loc[idx_start:idx_end, col] = np.nan

        ### Interpolate missing NaN values ###
        # for col in ['x_coordinate', 'y_coordinate', 'pupil_size']:
        #     samples[col] = samples[col].interpolate(
        #         method='linear',
        #         limit_direction='both'
        #     )

        ### Low-pass filter the pupil data ###
        # sfreq = 1000
        # nyq  = sfreq / 2
        # cutoff = 8 # Hz
        # normal_cutoff = cutoff / nyq
        # order = 3
        # b, a = butter(order, normal_cutoff, btype="low", analog=False)
        # samples['pupil_size'] = filtfilt(b, a, np.array(samples['pupil_size']))

        ### Epoch the data ###
        times = np.round(np.arange(args.tmin, args.tmax, 0.001), 3)
        channels = ['x_coordinate', 'y_coordinate', 'pupil_size']
        epochs = np.zeros((n_trials, len(channels), len(times)),
            dtype=np.float32)
        for e, onset in enumerate(stim_onset):
            t_start = np.round(onset+args.tmin, 3)
            t_end = np.round(onset+args.tmax, 3)
            idx_start = np.where(np.array(samples['timestamp']) == t_start)[0][0]
            idx_end = np.where(np.array(samples['timestamp']) == t_end)[0][0]
            for c, chan in enumerate(channels):
                epochs[e,c] = np.array(samples[chan])[idx_start:idx_end]
        del samples

        ### Convert gaze data to visual degrees (wrt screen center) ###
        stim_deg = 5 # stimulus size in degrees
        stim_px = 194 # stimulus size in pixels
        deg_per_px = stim_deg / stim_px # degrees of visual angle for each screen pixel
        screen_width_px = 1680
        screen_height_px = 1050
        screen_center_x = screen_width_px / 2
        screen_center_y = screen_height_px / 2
        epochs[:,0] = (epochs[:,0] - screen_center_x) * deg_per_px
        epochs[:,1] = (epochs[:,1] - screen_center_y) * deg_per_px

        ### Baseline correct the pupil data ###
        idx_start = np.where(times == args.tmin)[0][0]
        idx_end = np.where(times == 0)[0][0]
        if args.baseline_correction != 'none':
            for e in range(len(epochs)):
                if args.baseline_correction == 'baseline':
                    mean = np.nanmean(epochs[e,2,idx_start:idx_end])
                elif args.baseline_correction == 'epoch':
                    mean = np.nanmean(epochs[e,2])
                if np.isnan(mean): # Set to NaN epochs with NaN baseline
                    epochs[e,2] = np.nan
                else:
                    epochs[e,2] = epochs[e,2] - mean

        ### Exclude samples deviating more than 4° from central fixation ###
        # Samples with gaze coordinates deviating more than 4° indicate eye
        # movements beyond the stimulus boundaries.
        # Excise data 20 ms before and 20 ms after each occurrence
        cutoff = 4 # degrees from central fixation
        pad = 0.02 # seconds
        for e in range(len(epochs)):
            euclDist = np.sqrt(np.power(epochs[e,0], 2) + \
                np.power(epochs[e,1], 2))
            idx_bad = np.where(euclDist > cutoff)[0]
            for i in idx_bad:
                t_start = np.round(times[i] - pad, 3)
                if t_start < times[0]:
                    idx_start = 0
                else:
                    idx_start = np.where(times == t_start)[0][0]
                t_end = np.round(times[i] + pad, 3)
                if t_end > times[-1]:
                    idx_end = len(times)
                else:
                    idx_end = np.where(times == np.round(t_end, 3))[0][0]
                epochs[e,:,idx_start:idx_end] = np.nan

        ### Resample ###
        data_sfreq = 1000
        if args.sfreq < data_sfreq:
            step = int(1000 / args.sfreq)
            idx = np.arange(0, len(times), step)
            epochs = epochs[:,:,idx]
            times = times[idx]

        ### Ensure that missing values are equal for gaze and pupil data ###
        for e in range(len(epochs)):
            idx_nan = np.isnan(epochs[e]).any(axis=(0))
            epochs[e,:,idx_nan] = np.nan

        ### Reject epochs with less than 75% valid samples ###
        # Epochs with less than 75% valid samples (i.e., samples with
        # non-NaN values) during the video presentation interval are discarded.
        # This effectively rejects trials with eye blinks, loss of tracking
        # during stimulus presentation, excessive eye movements during stimulus
        # presentation, or trials with no available data points during the
        # baseline period (for baseline correction).
        threshold = 0.75
        t_start = np.where(times == 0)[0][0]
        t_end = np.where(times == 3)[0][0]
        idx_non_nan = ~np.isnan(epochs[:,:,t_start:t_end]).any(axis=(1))
        perc_good = np.sum(idx_non_nan, 1) / epochs[:,:,t_start:t_end].shape[2]
        idx_bad = np.where(perc_good < threshold)[0]
        epochs = np.delete(epochs, idx_bad, axis=0)
        stim_id = np.delete(stim_id, idx_bad)
        trial_run = np.full_like(stim_id, f+1)
        trial_num = np.delete(trial_num, idx_bad)

        ### Store the preprocessed eyetracking data and trial information ###
        if f == 0:
            eyetracking_data = epochs.astype(np.float32)
            stimulus_id = stim_id
            run_number = trial_run
            trial_number = trial_num
        else:
            eyetracking_data = np.append(eyetracking_data,
                epochs.astype(np.float32), 0)
            stimulus_id = np.append(stimulus_id, stim_id)
            run_number = np.append(run_number, trial_run)
            trial_number = np.append(trial_number, trial_num)
        del epochs, stim_id, trial_run, trial_num

    ### Output ###
    return eyetracking_data, times, stimulus_id, run_number, trial_number