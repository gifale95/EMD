"""Convert datasets from source data to raw BIDS format.

Parameters
----------
subject : int
    Subject number.
n_sessions : int
    Number of EEG recording sessions per subject.
n_runs : int
    Number of runs per session.
dataset_dir : str
    Directory of the dataset folder.

"""

import argparse
import os
import numpy as np
from scipy import io
import shutil
import csv
import json
import mne
from eyelinkio import read_edf
import gzip
import pandas as pd


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--n_sessions', default=8, type=int)
parser.add_argument('--n_runs', default=16, type=int)
parser.add_argument('--dataset_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset/', type=str)
args, unknown = parser.parse_known_args()

print('\n\n\n>>> Dataset source to raw BIDS <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the dataset_description.json file
# =============================================================================
dataset_description_json = {
    "Name": "EEG Moments Dataset (EMD)",
    "BIDSVersion": "1.9.0",
    "DatasetType": "raw",
    "License": "CC NC 4.0",
    "Authors": [
        "Alessandro T. Gifford",
        "Pablo Oyarzo",
        "Anne W. Zonneveld",
        "Christina Sartzetaki",
        "Iris I.A. Groen",
        "Radoslaw M. Cichy"
    ],
    "HowToAcknowledge": "??????????????????????????????", # !!!
    "EthicsApprovals": [
        "Procedures were approved by the ethical committee of the Department of Education and Psychology at Freie Universität Berlin and were in accordance with the Declaration of Helsinki."
    ],
    "ReferencesAndLinks": [
        "Paper: ???????????????????????????????", # !!!
        "Code: https://github.com/gifale95/EMD"
    ]
}

file_name_json = os.path.join(args.dataset_dir, "dataset_description.json")

with open(file_name_json, "w") as f:
    json.dump(dataset_description_json, f, indent=4)
del dataset_description_json


# =============================================================================
# Create the participants.tsv file
# =============================================================================
participants = [
    ["participant_id", "age", "sex"],
    ["sub-01", 28, "m"],
    ["sub-02", 32, "m"],
    ["sub-03", 24, "f"],
    ["sub-04", 23, "f"],
    ["sub-05", 21, "f"],
    ["sub-06", 21, "m"]
]

file_name_tsv = os.path.join(args.dataset_dir, "participants.tsv")

with open(file_name_tsv, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(participants)
del participants


# =============================================================================
# Create the *_eeg.json file (EEG)
# =============================================================================
eeg_json = {
    "EEGReference":"FCz",
    "EEGGround":"Fpz",
    "SamplingFrequency":1000,
    "PowerLineFrequency":50,
    "SoftwareFilters":{
        "LowpassFilter": {
            "MaximumFrequency": 280
        }
    },
    "EEGChannelCount":128,
    "ECGChannelCount":0,
    "EMGChannelCount":0,
    "EOGChannelCount":0,
    "RecordingType":"continuous",
    "EEGPlacementScheme":"10-10",
    "CapManufacturer":"EasyCap",
    "CapManufacturersModelName":"antiCAP snap",
    "Manufacturer":"Brain Vision",
    "ManufacturersModelName":"actiCHamp Plus",
    "TaskName":"video",
    "InstitutionName":"Freie Universität Berlin"
}

file_name_json = os.path.join(args.dataset_dir, "task-video_eeg.json")

with open(file_name_json, "w") as f:
    json.dump(eeg_json, f, indent=4)
del eeg_json


# =============================================================================
# Create the *_events.json file (EEG)
# =============================================================================
events_json = {
    "onset": {
        "Description": "Onset of the video measured from the beginning of the EEG recording.",
            "Units": "seconds"
    },
    "duration": {
        "Description": "Duration of the video (measured from onset) in seconds.",
        "Units": "seconds"
    },
    "trial_num": {
        "LongName": "Trial Number",
        "Description": "Number of each trial within a run, in the range [1, 66]."
    },
    "stim_id": {
        "LongName": "Stimulus ID",
        "Description": "Index indicating the number ID of the presented video. Stimuli with stim_id in the range [1, 1000] belong to the training split, and stimuli with stim_id in the range [1001, 1102] belong to the test split."
    },
    "stim_file": {
        "LongName": "Stimulus File",
        "Description": "Represents the location of the stimulus video file presented at the given onset time."
    },
    "is_task": {
        "Description": "Whether the trial required an active behavioral response",
        "Levels": {
            "0": "Passive/non-task trial",
            "1": "Task trial"
        }
    },
    "correct": {
        "Description": "Behavioral response accuracy",
        "Levels": {
            "0": "Incorrect response",
            "1": "Correct response"
        }
    }
}

file_name_json = os.path.join(args.dataset_dir, "task-video_events.json")

with open(file_name_json, "w") as f:
    json.dump(events_json, f, indent=4)
del events_json


# =============================================================================
# Create the *_physio.json file (eyetracking)
# =============================================================================
physio_json = {
    "PhysioType": "eyetrack",
    "RecordedEye": "right",
    "Manufacturer": "SR-Research",
    "Columns": [
        "timestamp",
        "x_coordinate",
        "y_coordinate",
        "pupil_size"
    ],
    "timestamp": {
        "Description": "a continuously increasing identifier of the sampling time registered by the device",
        "Units": "seconds",
        "Origin": "System startup"
    },
    "x_coordinate": {
        "LongName": "Gaze position (x)",
        "Description": "Gaze position x-coordinate of the recorded eye, in the coordinate units specified in the corresponding metadata sidecar.",
        "Units": "pixel"
    },
    "y_coordinate": {
        "LongName": "Gaze position (y)",
        "Description": "Gaze position y-coordinate of the recorded eye, in the coordinate units specified in the corresponding metadata sidecar.",
        "Units": "pixel"
    },
    "pupil_size": {
        "Description": "Pupil area of the recorded eye as calculated by the eye-tracker in arbitrary units (see EyeLink's documentation for conversion).",
        "Units": "a.u."
    },
    "SampleCoordinateSystem": "gaze-on-screen",
    "ManufacturersModelName": "EYELINK II 1",
    "DeviceSerialNumber": "CL1-82113",
    "EyeTrackingMethod": "P-CR",
    "SamplingFrequency": 1000,
    "StimulusPresentation": {
        "ScreenDistance": 0.6,
        "ScreenOrigin": ["top", "left"],
        "ScreenRefreshRate": 60,
        "ScreenResolution": [1680, 1050],
        "ScreenSize": [0.475, 0.297]
    }
}

file_name_json = os.path.join(args.dataset_dir, "task-video_recording-eye1_physio.json")

with open(file_name_json, "w") as f:
    json.dump(physio_json, f, indent=4)
del physio_json


# =============================================================================
# Create the *_physioevents.json file (eyetracking)
# =============================================================================
events_json = {
    "onset": {
        "Description": "Onset of the event measured from the beginning of the eyetracking recording.",
            "Units": "seconds"
    },
    "duration": {
        "Description": "Duration of the video (measured from onset) in seconds.",
        "Units": "seconds"
    },
    "event_type": {
        "Description": "Type of the event",
        "Levels": {
            "blink": "Blink event",
            "stimulus": "Stimulus (video) presentation event"
        }
    },
    "trial_num": {
        "LongName": "Trial Number",
        "Description": "Number of each trial within a run, in the range [1, 66]."
    },
    "stim_id": {
        "LongName": "Stimulus ID",
        "Description": "Index indicating the number ID of the presented video. Stimuli with stim_id in the range [1, 1000] belong to the training split, and stimuli with stim_id in the range [1001, 1102] belong to the test split."
    },
    "stim_file": {
        "LongName": "Stimulus File",
        "Description": "Represents the location of the stimulus video file presented at the given onset time."
    },
    "is_task": {
        "Description": "Whether the trial required an active behavioral response",
        "Levels": {
            "0": "Passive/non-task trial",
            "1": "Task trial"
        }
    },
    "correct": {
        "Description": "Behavioral response accuracy",
        "Levels": {
            "0": "Incorrect response",
            "1": "Correct response"
        }
    }
}

file_name_json = os.path.join(args.dataset_dir, "task-video_recording-eye1_physioevents.json")

with open(file_name_json, "w") as f:
    json.dump(events_json, f, indent=4)
del events_json


# =============================================================================
# Get the stimulus video file names
# =============================================================================
videos = os.listdir(os.path.join(args.dataset_dir, 'stimuli'))
videos.sort()


# =============================================================================
# Format the EEG files into BIDS (EEG)
# =============================================================================
ftypes = ['eeg', 'vhdr', 'vmrk']
sub = args.subject

# Loop across sessions
for ses in range(1, args.n_sessions+1):

    # Create the EEG data folder
    eeg_dir_old = os.path.join(args.dataset_dir, 'sourcedata',
        f'sub-{sub:02d}', f'ses-{ses:02d}')
    eeg_dir_new = os.path.join(args.dataset_dir, f'sub-{sub:02d}',
        f'ses-{ses:02d}', 'eeg')
    os.makedirs(eeg_dir_new, exist_ok=True)

    # Load the stimulus presentation order
    stim_order = io.loadmat(os.path.join(eeg_dir_old,
        f'stim_order_sub-{sub:02}_sess-{ses:02}.mat'))
    stim_order = stim_order['stim_order']

    # Loop across runs and file types
    for r, run in enumerate(range(1, args.n_runs+1)):

        # Get the stimulus presentation order for the current run
        stim_order_run = stim_order[:,r]
        # Remove missing EEG data:
        # Subject 1, session 3, run 10, trial 1
        if sub == 1 and ses == 3 and run == 10:
            stim_order_run = np.delete(stim_order_run, 0)
        # Subject 5, session 7, run 16, trials 65-66
        elif sub == 5 and ses == 7 and run == 16:
            stim_order_run = np.delete(stim_order_run, np.arange(64, 66))

        # Loop across EEG file types
        for ftype in ['eeg', 'vhdr', 'vmrk']:

            # Rename and move the EEG files to the new folder
            old_filename = f"subj-{sub:02d}_sess-{ses:02d}_{run:04d}.{ftype}"
            new_filename = f"sub-{sub:02d}_ses-{ses:02d}_task-video_run-{run:02d}_eeg.{ftype}"
            if os.path.exists(os.path.join(eeg_dir_old, old_filename)):
                shutil.move(
                    os.path.join(eeg_dir_old, old_filename),
                    os.path.join(eeg_dir_new, new_filename)
                    )
                print(f"Moved {os.path.join(eeg_dir_old, old_filename)} to {os.path.join(eeg_dir_new, new_filename)}")
            else:
                print(f"File {os.path.join(eeg_dir_old, old_filename)} does not exist.")

            # Add the new EEG file names to the '.vhdr' and '.vmrk' files
            if ftype in ['vhdr', 'vmrk']:
                with open(os.path.join(eeg_dir_new, new_filename), "r") as f:
                    lines = f.readlines()
                lines = [
                    f"DataFile=sub-{sub:02d}_ses-{ses:02d}_task-video_run-{run:02d}_eeg.eeg\n"
                    if line.startswith("DataFile=")
                    else f"MarkerFile=sub-{sub:02d}_ses-{ses:02d}_task-video_run-{run:02d}_eeg.vmrk\n"
                    if line.startswith("MarkerFile=")
                    else line
                    for line in lines
                ]
                with open(os.path.join(eeg_dir_new, new_filename), "w") as f:
                    f.writelines(lines)


# =============================================================================
# Create the *_events.tsv file (EEG)
# =============================================================================
        # Get the events from the EEG data
        filename = f"sub-{sub:02d}_ses-{ses:02d}_task-video_run-{run:02d}_eeg.vhdr"
        raw = mne.io.read_raw_brainvision(os.path.join(eeg_dir_new, filename), preload=False)
        events = mne.events_from_annotations(raw)[0][1:]

        # Check that the event numbers match the number of stimulus
        # presentations in the stim_order_run
        if len(events) != len(stim_order_run):
            raise ValueError(f"The number of stimulus lines in the .vmrk file ({len(events)}) does not match the number of stimulus presentations in the stim_order_run ({len(stim_order_run)}).")

        # Check that the events match thes stimulus presentation order in
        # the stim_order_run
        for i, event in enumerate(events):
            event_number = int(str(event[2])[:2])
            stim_num = int(str(stim_order_run[i])[:2])
            if event_number != stim_num:
                raise ValueError(f"The event number in the .vmrk file ({event_number}) does not match the stimulus presentation order in the stim_order_run ({stim_num[i]}).")

        # Get the trial numbers
        trial_nums = np.arange(1, len(stim_order_run)+1)
        if sub == 1 and ses == 3 and run == 10:
            trial_nums += 1

        # Load the behavioral results
        beh_file = os.path.join(eeg_dir_old, f'run-{run:03d}.mat')
        beh = io.loadmat(beh_file)['data']
        task_trial = np.array([elem.item() for elem in beh[0][0][7]['task_trial'][0]], dtype=int)
        correctness = np.array([elem.item() for elem in beh[0][0][7]['correctness'][0]], dtype=float)
        # Remove missing EEG data:
        # Subject 1, session 3, run 10, trial 1
        if sub == 1 and ses == 3 and run == 10:
            task_trial = np.delete(task_trial, 0)
            correctness = np.delete(correctness, 0)
        # Subject 5, session 7, run 16, trials 65-66
        elif sub == 5 and ses == 7 and run == 16:
            task_trial = np.delete(task_trial, np.arange(64, 66))
            correctness = np.delete(correctness, np.arange(64, 66))
        if len(task_trial) != len(stim_order_run):
            raise ValueError(f"The length of task_trial ({len(task_trial)}) does not match the expected number of stimulus presentations ({len(stim_order_run)}).")
        if len(correctness) != len(stim_order_run):
            raise ValueError(f"The length of correctness ({len(correctness)}) does not match the expected number of stimulus presentations ({len(stim_order_run)}).")

        # Add the events to the *_events.tsv file
        events_tsv = [["onset", "duration", "trial_num", "stim_id", "stim_file", "is_task", "correct"]]
        for i, stim in enumerate(stim_order_run):
            correct =  "n/a" if np.isnan(correctness[i]) else correctness[i].astype(int)
            events_tsv.append(
                [events[i][0]/1000,     # onset
                3,                      # duration
                trial_nums[i],          # trial_num
                int(stim),              # stim_id
                videos[stim-1],         # stim_file
                task_trial[i],          # is_task
                correct                 # correct
            ])

        # Save the event files
        file_name_tsv = os.path.join(eeg_dir_new,
            f"sub-{sub:02d}_ses-{ses:02d}_task-video_run-{run:02d}_events.tsv")
        with open(file_name_tsv, "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(events_tsv)
        del events_tsv


# =============================================================================
# Format the eyetracking files into BIDS (eyetracking)
# =============================================================================
        # Extract and save the eyetracking samples from the .edf files
        edf_dir = os.path.join(eeg_dir_old, f'r{run:02d}.edf')
        edf = read_edf(edf_dir)

        # Extract the time-continuous samples
        times = np.round(np.reshape(edf['times'], (-1, 1)), 3) # Round to 3 decimals
        samples = np.transpose(edf['samples'])
        eye_samples = np.hstack([times, samples])

        # Save the samples
        eye_samples_filename = os.path.join(eeg_dir_new,
            f"sub-{sub:02d}_ses-{ses:02d}_task-video_run-{run:02d}_recording-eye1_physio.tsv.gz")
        np.savetxt(
            gzip.open(eye_samples_filename, "wt"),
            eye_samples,
            delimiter="\t"
        )
        del eye_samples


# =============================================================================
# Create the *_recording-eye1_physioevents.tsv.gz file (eyetracking)
# =============================================================================
        # Extract the blink times
        blinks = edf['discrete']['blinks']
        onset_blink = []
        offset_blink = []
        for blink in blinks:
            onset_blink.append(blink[1])
            offset_blink.append(blink[2])
        onset_blink = np.array(onset_blink)
        offset_blink = np.array(offset_blink)
        duration_blink = np.round(offset_blink - onset_blink, 3)

        # Get the stimulus presentation order for the current run
        stim_order_run = stim_order[:,r]

        # Extract the video onset times
        messages = edf['discrete']['messages']
        onset_stim = []
        for m, (onset, message) in enumerate(messages):
            if "TRIAL_START" in message.astype(str):
                onset_stim.append(onset)
        onset_stim = np.array(onset_stim)
        if len(onset_stim) != len(stim_order_run):
            raise ValueError(f"The number of video onsets in the .edf file ({len(onset_stim)}) does not match the expected number of stimulus presentations ({len(stim_order_run)}).")

        # Load the behavioral results
        beh_file = os.path.join(eeg_dir_old, f'run-{run:03d}.mat')
        beh = io.loadmat(beh_file)['data']
        task_trial = np.array([elem.item() for elem in beh[0][0][7]['task_trial'][0]], dtype=int)
        correctness = np.array([elem.item() for elem in beh[0][0][7]['correctness'][0]], dtype=float)
        if len(task_trial) != len(stim_order_run):
            raise ValueError(f"The length of task_trial ({len(task_trial)}) does not match the expected number of stimulus presentations ({len(stim_order_run)}).")
        if len(correctness) != len(stim_order_run):
            raise ValueError(f"The length of correctness ({len(correctness)}) does not match the expected number of stimulus presentations ({len(stim_order_run)}).")

        # Remove trials with missing eyetracking data
        if sub == 4 and ses == 5 and run == 5:
            tot_trials = 39
            onset_stim = onset_stim[:tot_trials]
            stim_order_run = stim_order_run[:tot_trials]
            task_trial = task_trial[:tot_trials]
            correctness = correctness[:tot_trials]
        if sub == 4 and ses == 6 and run == 10:
            tot_trials = 23
            onset_stim = onset_stim[:tot_trials]
            stim_order_run = stim_order_run[:tot_trials]
            task_trial = task_trial[:tot_trials]
            correctness = correctness[:tot_trials]
        if sub == 4 and ses == 6 and run == 12:
            tot_trials = 33
            onset_stim = onset_stim[:tot_trials]
            stim_order_run = stim_order_run[:tot_trials]
            task_trial = task_trial[:tot_trials]
            correctness = correctness[:tot_trials]
        if sub == 5 and ses == 6 and run == 11:
            tot_trials = 50
            onset_stim = onset_stim[:tot_trials]
            stim_order_run = stim_order_run[:tot_trials]
            task_trial = task_trial[:tot_trials]
            correctness = correctness[:tot_trials]

        # Create the blink events
        onset = []
        duration = []
        event_type = []
        trial_nums = []
        stim_id = []
        stim_file = []
        is_task = []
        correct = []
        for i in range(len(onset_blink)):
            onset.append(onset_blink[i])
            duration.append(duration_blink[i])
            event_type.append("blink")
            trial_nums.append("n/a")
            stim_id.append("n/a")
            stim_file.append("n/a")
            is_task.append("n/a")
            correct.append("n/a")
        # Create the stimulus presentation events
        for i in range(len(onset_stim)):
            onset.append(onset_stim[i])
            duration.append(3)
            event_type.append("stimulus")
            trial_nums.append(i+1)
            stim_id.append(stim_order_run[i])
            stim_file.append(videos[stim_order_run[i]-1])
            is_task.append(task_trial[i])
            correct.append("n/a" if np.isnan(correctness[i]) else correctness[i].astype(int))
        # Sort the events by onset time
        idx = np.argsort(onset)

        # Add the events to the *_recording-eye1_physioevents.tsv.gz file
        events_tsv = [["onset", "duration", "event_type", "trial_num", "stim_id", "stim_file", "is_task", "correct"]]
        for i in idx:
            events_tsv.append([
                onset[i],       # onset
                duration[i],    # duration
                event_type[i],  # event_type
                trial_nums[i],  # trial_num
                stim_id[i],     # stim_id
                stim_file[i],   # stim_file
                is_task[i],     # is_task
                correct[i]      # correct
            ])
        if len(events_tsv)-1 != len(onset_blink)+len(onset_stim):
            raise ValueError(f"The number of events in the events_tsv ({len(events_tsv)-1}) does not match the expected number of blinks and stimulus presentations ({len(onset_blink)+len(onset_stim)}).")

        # Save the event files
        file_name_tsv = os.path.join(eeg_dir_new,
            f"sub-{sub:02d}_ses-{ses:02d}_task-video_run-{run:02d}_recording-eye1_physioevents.tsv.gz")
        events_tsv = pd.DataFrame(events_tsv)
        events_tsv.to_csv(
            file_name_tsv,
            sep="\t",
            index=False,
            compression="gzip",
            header=False
        )
        del events_tsv


# =============================================================================
# Create the *_recording-eye1_physio.json file (eyetracking)
# =============================================================================
        # Compute the StartTime (Start time in seconds in relation to the
        # start of acquisition of the first data sample in the
        # corresponding (neural) dataset (negative values are allowed).)
        # In case of missing EEG data:
        # Subject 1, session 3, run 10, trial 1
        if sub == 1 and ses == 3 and run == 10:
            StartTime = (events[0,0] / 1000) - onset_stim[1]
        else:
            StartTime = (events[0,0] / 1000) - onset_stim[0]

        physio_json = {
            "StartTime": StartTime,
            "RecordedEye": "right",
            "PupilFitMethod": "CENTROID",
            "CalibrationCount": len(edf['info']['calibrations']),
            "CalibrationType": edf['info']['calibrations'][0]['model'],
            "AverageCalibrationError": np.mean(edf['info']['calibrations'][0]['validation']['offset']),
            "MaximalCalibrationError": max(edf['info']['calibrations'][0]['validation']['offset'])
        }

        file_name_json = os.path.join(eeg_dir_new,
            f"sub-{sub:02d}_ses-{ses:02d}_task-video_run-{run:02d}_recording-eye1_physio.json")

        with open(file_name_json, "w") as f:
            json.dump(physio_json, f, indent=4)