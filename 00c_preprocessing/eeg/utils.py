def preprocess_eeg(args, session):
    """Load the raw EEG data and perform filtering, eyeblink removal, epoching,
    rejection and interpolation of bad epochs/channels, baseline correction,
    and frequency downsampling.

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
    from mne.preprocessing import ICA
    import pandas as pd
    from autoreject import AutoReject

    ### Get the EEG files ###
    eeg_files = glob(os.path.join(args.emd_dir, f'sub-{args.subject:02}',
        f'ses-{session:02}', 'eeg', '*.vhdr'))
    eeg_files.sort()

    ### Get the event files ###
    event_files = glob(os.path.join(args.emd_dir, f'sub-{args.subject:02}',
        f'ses-{session:02}', 'eeg', '*events.tsv'))
    event_files.sort()

    ### Loop across runs ###    
    for f in range(len(eeg_files)):

        ### Load the raw EEG data ###
        # Load the raw EEG data
        raw = mne.io.read_raw_brainvision(eeg_files[f], preload=True)
        # Filter the data
        if args.highpass != None and args.lowpass != None:
            raw = raw.copy().filter(l_freq=args.highpass, h_freq=args.lowpass)

        ### Automatic ICA eyeblink removal ###
        # Fit on highpass-filtered copy
        raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None)
        ica = ICA(n_components=20, random_state=20200220, method='fastica',
            max_iter='auto')
        ica.fit(raw_for_ica)
        del raw_for_ica
        # Find blink components (use a frontal EEG channel as a proxy for
        # blinks)
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='Fp1')
        ica.exclude = eog_indices
        print(f"Excluding ICA components: {eog_indices}")
        # Apply the ICA solution to the raw data
        raw = ica.apply(raw)

        ### Epoch the EEG ###
        # Get the event samples info
        events, _ = mne.events_from_annotations(raw)
        # Epoch the EEG trials
        epochs = mne.Epochs(raw, events, tmin=args.tmin, tmax=args.tmax,
            baseline=None, preload=True)
        del raw

        ### Resample ###
        if args.sfreq < 1000:
            epochs.resample(args.sfreq)

        ### Reject and interpolate bad epochs and channels ###
        ar = AutoReject(
            n_interpolate=[1, 2, 4, 8, 16, 32],
            random_state=20200220,
            verbose=False
        )
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
        bad_epochs = np.where(reject_log.bad_epochs)[0]
        del epochs
        # reject_log.plot()

        ### Baseline correction ###
        if args.baseline_correction == 1:
            epochs_clean.apply_baseline(baseline=(None, 0))

        ### Get the channel names and time points ###
        ch_names = epochs_clean.info['ch_names']
        times = np.round(epochs_clean.times, 3)

        ### Match the EEG events with the stimulus IDs ###
        events = epochs_clean.events
        events_tsv = pd.read_csv(event_files[f], sep="\t")
        stim_id = np.delete(np.asarray(events_tsv['stim_id']), bad_epochs)
        stim_id_short = []
        for s in stim_id:
            stim_id_short.append(int(str(s)[:2]))
        if not all(events[:,2] == np.asarray(stim_id_short)):
            raise Exception('EEG events do not match with stimulus IDs')
        del events

        ### Store the preprocessed EEG and the trial information ###
        trial_n = np.delete(np.asarray(events_tsv['trial_num']), bad_epochs)
        if len(trial_n) != len(stim_id):
            raise Exception(f'Trial number {len(trial_n)} does not match with stimulus presentation order {len(stim_id)}!')
        if f == 0:
            eeg = epochs_clean.get_data().astype(np.float32)
            stimulus_id = stim_id
            run_number = np.full_like(stim_id, f+1)
            trial_number = trial_n
        else:
            eeg = np.append(eeg, epochs_clean.get_data().astype(np.float32), 0)
            stimulus_id = np.append(stimulus_id, stim_id)
            run_number = np.append(run_number, np.full_like(stim_id, f+1))
            trial_number = np.append(trial_number, trial_n)
        del epochs_clean

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
        eeg_shape[2]), dtype=np.float32)
    test_data[:] = np.nan
    # Index the test split data
    for v, video in enumerate(test_video_cond):
        idx_videos = np.where(stim_id == video)[0]
        if len(idx_videos) < 2:
            raise Exception(f'Less than 2 video presentations for condition {video}!')
        test_data[v,:len(idx_videos)] = zscored_data[idx_videos]

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