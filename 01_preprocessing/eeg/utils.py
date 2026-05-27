def epoch_eeg(args, session):
    """Convert the EEG data to MNE raw format and perform filtering, epoching,
    baseline correction and frequency downsampling.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    session : str
        EEG recording session.

    Returns
    -------
    epoched_data : float
        Epoched EEG data.
    stim_order : int
        Stimuli conditions presentation order.
    ch_names : list of str
        EEG channel names.
    times : float
        EEG time points.
    info : Info
        EEG data info.
    beh_ground_truth : int
        Ground truth behavioral responses.
    beh_response : int
        Behavioral responses.

    """

    import os
    import numpy as np
    from glob import glob
    from scipy import io
    import mne

    ### Load the stimuli presentation order ###
    stim_order = io.loadmat(os.path.join(args.project_dir, 'dataset',
        'source_data', f'sub-{args.sub:02}', f'ses-{session+1:02}',
        f'stim_order_sub-{args.sub:02}_sess-{session+1:02}.mat'))
    stim_order = stim_order['stim_order']
    stim_order = np.reshape(stim_order, -1, order='F')
    # Remove missing EEG data:
    # Subject 1, session 3, run 10, trial 1
    if args.sub == 1 and (session+1) == 3:
        stim_order = np.delete(stim_order, 594)

    ### Load the EEG trigger numbers ###
    eeg_triggers = np.zeros(len(stim_order), dtype=int)
    for s, stim in enumerate(stim_order):
        if stim < 100:
            eeg_triggers[s] = stim
        else:
            eeg_triggers[s] = int(str(stim)[:2])

    ### Load the behavioral data ###
    behav_files = glob(os.path.join(args.project_dir, 'dataset', 'source_data',
        f'sub-{args.sub:02}', f'ses-{session+1:02}', 'run-*.mat'))
    behav_files.sort()
    for f, file in enumerate(behav_files):
        behav = io.loadmat(file)['data']
        res = np.asarray(behav[0][0][7]['response'][0], dtype=float)
        correct = np.asarray(behav[0][0][7]['correctness'][0], dtype=float)
        if f == 0:
            response = res
            correctness = correct
        else:
            response = np.append(response, res, 0)
            correctness = np.append(correctness, correct, 0)

    ### Create the behavioral results matrices of each trial ###
    # (0==left_key_press, 1==right_key_press, 2==no_response,
    # 99==no_task_trials)
    beh_response = np.zeros(len(response), dtype=int)
    beh_response[:] = 99
    beh_correctness = np.zeros(len(correctness), dtype=int)
    beh_correctness[:] = 99
    for t in range(len(beh_response)):
        if ~np.isnan(response[t]):
            beh_response[t] = response[t]
        if ~np.isnan(correctness[t]):
            beh_correctness[t] = correctness[t]

    ### Get the EEG files ###
    eeg_files = glob(os.path.join(args.project_dir, 'dataset', 'source_data',
        f'sub-{args.sub:02}', f'ses-{session+1:02}', '*.vhdr'))
    eeg_files.sort()

    ### Load and preprocess the EEG data ###
    for f, file in enumerate(eeg_files):
        # Load the raw EEG data
        raw = mne.io.read_raw_brainvision(file, preload=True)
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
        info = epochs.info
        # Store the epoched data
        if f == 0:
            epoched_data = epochs.get_data()
            all_events_num = events[:,2]
        else:
            epoched_data = np.append(epoched_data, epochs.get_data(), 0)
            all_events_num = np.append(all_events_num, events[:,2], 0)
        del epochs, events

    ### Match the EEG triggers with stimulus presentation order ###
    if not all(all_events_num == eeg_triggers):
        # Missing EEG data:
        # Subject 1, session 3, run 10, trial 1
        if args.sub == 1 and (session+1) == 3:
            pass
        else:
            raise Exception('EEG events do not match with stimulus presentation order!')

    ### Output ###
    return epoched_data, stim_order, ch_names, times, info, beh_response, \
        beh_correctness


def mvnn(args, epoched_data, stim_order, session):
    """For each video condition compute the covariance matrix of the EEG data
    (calculated for every time-point or epoch/repetitions), compute the mean
    covariance matrix across all image conditions, and use its inverse to
    whiten the EEG data.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    epoched_data : float
        Epoched EEG data.
    stim_order : int
        Stimuli conditions presentation order.
    session : str
        EEG recording session.

    Returns
    -------
    whitened_data : float
        Whitened EEG data.

    """

    import numpy as np
    from tqdm import tqdm
    from sklearn.discriminant_analysis import _cov
    import scipy

    ### Get unique video conditions and repetitions ###
    stim_cond = np.unique(stim_order)
    n_rep = 3

    ### Compute the covariance matrix of each video condition ###
    sigma_all = np.zeros((len(stim_cond),epoched_data.shape[1],
        epoched_data.shape[1]))
    whitened_data = epoched_data
    for v in tqdm(range(len(stim_cond))):
        # Index the repetitions of each video condition
        idx_cond = np.where(stim_order == stim_cond[v])[0]
        if len(idx_cond) != n_rep:
            # Missing EEG data:
            # Subject 1, session 3, run 10, trial 1
            if args.sub == 1 and (session+1) == 3:
                pass
            else:
                raise Exception('Not all data conditions have 3 repetitions!')
        cond_data = epoched_data[idx_cond]
        # Compute the covariace matrices at each time point, and then average
        # across time points
        if args.mvnn == "time":
            sigma = np.mean([_cov(cond_data[:,:,t],
                shrinkage='auto') for t in range(cond_data.shape[2])], axis=0)
        # Compute the covariace matrices at each epoch (EEG repetition), and
        # then average across epochs/repetitions
        elif args.mvnn == "epochs":
            sigma = np.mean([_cov(np.transpose(cond_data[e]),
                shrinkage='auto') for e in range(cond_data.shape[0])], axis=0)
        if args.mvnn != "none":
            sigma_all[v] = sigma

    ### Apply MVNN ###
    if args.mvnn != "none":
        # Compute the inverse of the covariance matrix
        sigma = np.mean(sigma_all, 0)
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
        # Whiten the data
        whitened_data = (whitened_data.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)

    ### Convert the data to float32 ###
    whitened_data = whitened_data.astype(np.float32)

    ### Output ###
    return whitened_data


def compute_ncsnr(preprocessed_eeg, stimulus_presentation_order):
    """Compute the noise ceiling as in the NSD paper, using the test split
    data.

    Parameters
    ----------
    preprocessed_eeg : list of float
        Preprocessed EEG data.
    stimulus_presentation_order : list of int
        Stimulus condition presentation order.

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
            stim_order = stimulus_presentation_order[s]
        else:
            zscored_data = np.append(zscored_data, np.reshape(
                eeg_sess, eeg_shape), 0)
            stim_order = np.append(stim_order, stimulus_presentation_order[s],
                0)

    ### Select the test split data ###
    test_video_cond = np.arange(1001, 1103)
    test_video_rep = 3 * len(preprocessed_eeg)
    # Test split data array of shape:
    # (Video conditions × EEG repetitions × EEG channels × EEG time points)
    test_data = np.zeros((len(test_video_cond), test_video_rep, eeg_shape[1],
        eeg_shape[2]))
    # Index the test split data
    for v, video in enumerate(test_video_cond):
        idx_videos = np.where(stim_order == video)[0]
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