def preprocess_eeg(args, session):
    """Load the raw EEG data and perform filtering, epoching, baseline
    correction and frequency downsampling.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    session : str
        EEG recording session.

    Returns
    -------
    eeg : list of float
        Preprocessed EEG data.
    stimulus_id : list of int
        Stimulus condition IDs.
    run_number : list of int
        Run number of each trial of a given session (in the range [1, 16]).
    trial_number : list of int
        Number of each trial of a given session (in the range [1, 66]).
    ch_names : list of str
        EEG channel names.
    times : float
        EEG time points.

    """

    import os
    import numpy as np
    from glob import glob
    import mne
    import pandas as pd

    ### Get the EEG files ###
    eeg_files = glob(os.path.join(args.project_dir, f'sub-{args.subject:02}',
        f'ses-{session:02}', 'eeg', '*.vhdr'))
    eeg_files.sort()

    ### Get the event files ###
    event_files = glob(os.path.join(args.project_dir, f'sub-{args.subject:02}',
        f'ses-{session:02}', 'eeg', '*events.tsv'))
    event_files.sort()

    ### Loop across runs ###    
    for f in range(len(eeg_files)):

        ### Load and preprocess the EEG data ###
        # Load the raw EEG data
        raw = mne.io.read_raw_brainvision(eeg_files[f], preload=True)
        # Filter the data
        if args.highpass != None and args.lowpass != None:
            raw = raw.copy().filter(l_freq=args.highpass, h_freq=args.lowpass)
        # Get the event samples info
        events, _ = mne.events_from_annotations(raw)
        # Epoch the EEG trials
        if args.baseline_correction == 1:
            epochs = mne.Epochs(raw, events, tmin=args.tmin, tmax=args.tmax,
                baseline=args.baseline, preload=True)
        elif args.baseline_correction == 0:
            epochs = mne.Epochs(raw, events, tmin=args.tmin, tmax=args.tmax,
                baseline=None, preload=True)
        del raw
        # Resample the epoched data
        if args.sfreq < 1000:
            epochs.resample(args.sfreq)
        ch_names = epochs.info['ch_names']
        times = epochs.times
        # Store the epoched data
        events = epochs.events
        if f == 0:
            eeg = epochs.get_data().astype(np.float32)
        else:
            eeg = np.append(eeg, epochs.get_data().astype(np.float32), 0)
        del epochs
        # Match the EEG events with the stimulus IDs
        events_tsv = pd.read_csv(event_files[f], sep="\t")
        stim_id = np.asarray(events_tsv['stim_id'])
        stim_id_short = []
        for s in stim_id:
            stim_id_short.append(int(str(s)[:2]))
        if not all(events[:,2] == np.array(stim_id_short)):
            raise Exception('EEG events do not match with stimulus IDs')
        del events
        # Store the stimulus order
        if f == 0:
            stimulus_id = stim_id
        else:
            stimulus_id = np.append(stimulus_id, stim_id)
        # Store the run number
        if f == 0:
            run_number = np.full_like(stim_id, f+1)
        else:
            run_number = np.append(run_number, np.full_like(stim_id, f+1))
        # Store the trial number
        trial_n = np.asarray(events_tsv['trial_num'])
        if len(trial_n) != len(stim_id):
            raise Exception(f'Trial number {len(trial_n)} does not match with stimulus presentation order {len(stim_id)}!')
        if f == 0:
            trial_number = trial_n
        else:
            trial_number = np.append(trial_number, trial_n)

    ### Output ###
    return eeg, stimulus_id, run_number, trial_number, ch_names, times


def compute_ncsnr(preprocessed_eeg, stimulus_id):
    """Compute the noise ceiling as in the NSD paper, using the test split
    data.

    Parameters
    ----------
    preprocessed_eeg : list of float
        Preprocessed EEG data.
    stimulus_id : list of int
        Stimulus condition IDs.

    Returns
    -------
    ncsnr : float
        Noise ceiling signal-to-noise ratio.
    noise_ceiling : float
        Noise ceiling.

    """

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    ### Standardize the data at each scan session ###
    for s in range(len(preprocessed_eeg)):
        eeg_shape = preprocessed_eeg[s].shape
        eeg_sess = np.reshape(preprocessed_eeg[s], (eeg_shape[0],-1))
        scaler = StandardScaler()
        eeg_sess = scaler.fit_transform(eeg_sess)
        if s == 0:
            zscored_data = np.reshape(eeg_sess, eeg_shape)
            stim_id = stimulus_id[s]
        else:
            zscored_data = np.append(zscored_data, np.reshape(eeg_sess, 
                eeg_shape), 0)
            stim_id = np.append(stim_id, stimulus_id[s], 0)

    ### Select the test split data ###
    test_video_cond = np.arange(1001, 1103)
    test_video_rep = 3 * len(preprocessed_eeg)
    # Test split data array of shape:
    # (Video conditions × EEG repetitions × EEG channels × EEG time points)
    test_data = np.zeros((len(test_video_cond), test_video_rep, eeg_shape[1],
        eeg_shape[2]))
    # Index the test split data
    for v, video in enumerate(test_video_cond):
        idx_videos = np.where(stim_id == video)[0]
        if len(idx_videos) != test_video_rep:
            raise Exception('Wrong test video condition repetition amount!')
        test_data[v] = zscored_data[idx_videos]

    ### Compute the ncsnr ###
    tot_var = np.nanvar(np.reshape(test_data,
        (-1, test_data.shape[2], test_data.shape[3])), axis=0, ddof=1)
    std_noise = np.sqrt(np.mean(np.nanvar(test_data, axis=1, ddof=1), 0))
    std_signal = tot_var - (std_noise ** 2)
    std_signal[std_signal<0] = 0
    std_signal = np.sqrt(std_signal)
    ncsnr = std_signal / std_noise

    ### Compute the noise ceiling ###
    noise_ceiling = 100 * ((ncsnr ** 2) / ((ncsnr ** 2) + (1 / test_video_rep)))

    ### Output ###
    return ncsnr, noise_ceiling