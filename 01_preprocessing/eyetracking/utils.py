def preprocess_eyetracking(args, session):
    """Import the eytrack data and perform epoching, baseline correction and
    frequency downsampling.
    https://github.com/cvnlab/nsddatapaper/blob/main/main/nsd_et_preprocessing.m
    https://stackoverflow.com/questions/44779315/detrending-data-with-nan-value-in-scipy-signal

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

    """

    import os
    from glob import glob
    import numpy as np
    import pandas as pd
    from scipy import stats
    from copy import copy
    from tqdm import tqdm

    ### Get the eyetracking files ###
    eye_files = glob(os.path.join(args.project_dir, f'sub-{args.subject:02}',
        f'ses-{session:02}', 'eeg', '*physio.tsv.gz'))
    eye_files.sort()

    ### Get the event files ###
    event_files = glob(os.path.join(args.project_dir, f'sub-{args.subject:02}',
        f'ses-{session:02}', 'eeg', '*physioevents.tsv.gz'))
    event_files.sort()

    ### Loop across runs ###
    for f in tqdm(range(len(eye_files))):

        ### Load the data ###
        # Import the eyelink samples, and set to NaN missing values during eye
        # blinks
        samples = pd.read_csv(eye_files[f], sep="\t", header=None)
        samples[0] = np.round(samples[0], 3) # Round the sample time to 3 digits (ms precision)
        samples[1] = samples[1]
        samples[2] = samples[2]
        samples = samples.to_numpy().astype(np.float32)
        # Import the stimulus IDs
        events = pd.read_table(event_files[f])
        stim_idx = np.where(events['event_type'] == 'stimulus')[0]
        stim_onset = events['onset'][stim_idx].to_numpy().astype(np.float32)
        stim_id = events['stim_id'][stim_idx].to_numpy().astype(int)
        trial_num = events['trial_num'][stim_idx].to_numpy().astype(int)
        trial_run = np.full_like(stim_id, f+1)
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
        if f == 0:
            stimulus_id = stim_id
            run_number = trial_run
            trial_number = trial_num
        else:
            stimulus_id = np.append(stimulus_id, stim_id)
            run_number = np.append(run_number, trial_run)
            trial_number = np.append(trial_number, trial_num)
        del stim_id
        # Import the eyeblinks
        blink_idx = np.where(events['event_type'] == 'blink')[0]
        blink_onset = events['onset'][blink_idx].to_numpy().astype(np.float32)
        blink_offset = (blink_onset + \
            events['duration'][blink_idx]).to_numpy().astype(np.float32)
        # Set negative and zero (artefacts) to NaNs
        samples[samples[:,1]<=0,1] = np.nan
        samples[samples[:,2]<=0,2] = np.nan
        samples[samples[:,3]<=0,3] = np.nan

        ### Remove eye blinks ###
        # Set to NaN samples 100 ms before and 100 ms after eye blinks
        pad = 0.1 # seconds
        blink_onset = np.round((blink_onset - pad), 3)
        blink_offset = np.round((blink_offset + pad), 3)
        for b in range(len(blink_onset)):
            if blink_onset[b] < samples[0,0]:
                idx_start = 0
            else:
                idx_start = np.where(samples[:,0] == blink_onset[b])[0][0]
            if blink_offset[b] > samples[-1,0]:
                idx_end = int(samples[-1,0])
            else:
                idx_end = np.where(samples[:,0] == blink_offset[b])[0][0]
            samples[idx_start:idx_end,1:4] = np.nan
        if any(samples[:,1:4].flatten() == 0):
            raise Exception('Zero values!')

        ### Convert gaze data to visual degrees ###
        stim_deg = 5 # stimulus size in degrees
        stim_px = 194 # stimulus size in pixels
        deg_per_px = stim_deg / stim_px
        samples[:,1:3] = samples[:,1:3] * deg_per_px

        ### Detrend and mean center the X-Y gaze data ###
        # Remove slow signal drift by linear detrending and mean-centering of
        # the gaze position time series (X and Y). This step assumes that the
        # mean gaze position corresponds to central fixation (the presentation
        # screen pixel resolution is 1680×1050). The pupil size is also
        # mean-centered
        # for c in range(1,samples.shape[1]-1):
        #     non_nan_idx = ~np.isnan(samples[:,c])
        #     m, b, _, _, _ = stats.linregress(
        #         np.arange(len(samples))[non_nan_idx],
        #         samples[:,c][non_nan_idx])
        #     samples[:,c] = samples[:,c] - ((m * np.arange(len(samples))) + b)
        # for c in range(1,samples.shape[1]):
        #     samples[:,c] = samples[:,c] - np.nanmean(samples[:,c])

        ### Epoch the data ###
        epoched_data = []
        for e in stim_onset:
            t_start = np.round(e+args.tmin, 3)
            t_end = np.round(e+args.tmax, 3)
            idx_start = np.where(samples[:,0] == t_start)[0][0]
            idx_end = np.where(samples[:,0] == t_end)[0][0]
            epoched_data.append(samples[idx_start:idx_end,1:4])
        epoched_data = np.asarray(epoched_data)
        times = np.round(np.arange(args.tmin, args.tmax, 0.001), 3)

        ### Mean centering ###
        idx_start = np.where(times == args.tmin)[0][0]
        idx_end = np.where(times == 0)[0][0]
        if args.mean_centering != 'none':
            for e in range(len(epoched_data)):
                if args.mean_centering == 'baseline':
                    mean = np.nanmean(epoched_data[e,idx_start:idx_end], 0)
                elif args.mean_centering == 'epoch':
                    mean = np.nanmean(epoched_data[e], 0)
                if any(np.isnan(mean)): # Set to NaN epochs with NaN baseline
                    epoched_data[e] = np.nan
                else:
                    epoched_data[e] = epoched_data[e] - mean

        ### Exclude samples deviating more than 1° from central fixation ###
        # Excise data 250 ms before and 250 ms after each occurrence
        cutoff = 1 # degrees from central fixation
        pad = 0.25 # seconds
        for e in range(len(epoched_data)):
            euclDist = np.sqrt(np.power(epoched_data[e,:,0], 2) + \
                np.power(epoched_data[e,:,1], 2))
            idx_bad = np.where(euclDist > cutoff)[0]
            for i in idx_bad:
                t_start = times[i] - pad
                if t_start < times[0]:
                    idx_start = 0
                else:
                    idx_start = np.where(times == np.round(t_start, 3))[0][0]
                t_end = times[i] + pad
                if t_end > times[-1]:
                    idx_end = len(times)
                else:
                    idx_end = np.where(times == np.round(t_end, 3))[0][0]
                epoched_data[e,idx_start:idx_end,0:3] = np.nan

        ### Temporal smoothing using a 50ms running average ###
        # Smooth the time-series data for gaze position and pupil size using a
        # 50-ms running average
        # smooth_epoched_data = copy(epoched_data)
        # smooth_ms = 50
        # for e in tqdm(range(len(epoched_data))):
        #     for t in range(epoched_data.shape[1]):
        #         if t < (smooth_ms / 2):
        #             idx_start = 0
        #         else:
        #             idx_start = int(t - (smooth_ms/2))
        #         if t > (epoched_data.shape[1] - smooth_ms / 2):
        #             idx_end = epoched_data.shape[1]
        #         else:
        #             idx_end = int(t + (smooth_ms/2))
        #         for c in range(epoched_data.shape[2]):
        #             if np.isnan(epoched_data[e,t,c]) == False:
        #                 smooth_epoched_data[e,t,c] = np.nanmean(
        #                     epoched_data[e,idx_start:idx_end,c])

        ### Downsample the data ###
        if args.sfreq < 1000:
            step = int(1000 / args.sfreq)
            idx = np.arange(0, len(times), step)
            epoched_data = epoched_data[:,idx]
            times = times[idx]

        # Store the epoched data
        if f == 0:
            eyetracking_data = epoched_data.astype(np.float32)
        else:
            eyetracking_data = np.append(eyetracking_data,
                epoched_data.astype(np.float32), 0)

    ### Output ###
    return eyetracking_data, times, stimulus_id, run_number, trial_number