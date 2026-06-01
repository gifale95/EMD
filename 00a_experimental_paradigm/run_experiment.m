% 1) Install GStreamer 1.18.6 Windows
				% (https://psychtoolbox.discourse.group/t/no-audio-when-movies-are-played-in-windows/4404/7)
% 2) Sync check failing
				% https://github.com/kleinerm/Psychtoolbox-3/blob/master/Psychtoolbox/PsychDocumentation/SyncTrouble.m

% https://psychtoolbox.discourse.group/t/invalid-mex-file-screen-problem/202/4
% AssertOpenGL

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PTB checks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc

rng('shuffle'); % randomization of trials according to Matlab internal time
KbName('UnifyKeyNames') % (following MacOS-X naming scheme)
Priority(2); %%% 2 (setting high-priority for Psychtoolbox timing)

% Get rid of warning messages
Screen('Preference', 'SkipSyncTests', 0); %%% 0
Screen('Preference', 'VisualDebugLevel', 2); %%% 2
Screen('Preference', 'SuppressAllWarnings', 0); %%% 0



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To edit
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set to '1' to send EEG triggers
eeg_mode = 0;
% Set to '0' to enable eye-tracking recordings
eyelink_dummy = 1;

% Directories
slash = '/';
video_dir = '../eeg_moments_dataset/stimuli';
metadata_dir = '../eeg_moments_dataset/derivatives/stimuli_metadata/annotations.json';
utils_dir = '../EMD/00a_experimental_paradigm/utils';
stim_order_dir = '../EMD/00a_experimental_paradigm/stimulus_order';
save_dir = '../EMD/00a_experimental_paradigm/behavioral_results';
addpath(utils_dir);

% Screen info
dist_from_screen = 600; % millimeters
screen_width = 475; % millimeters

% Screen window number for PTB 'OpenWindow' function
screenWin = 0; % 0

% Input device number for PTB keyboard functions
deviceNum = -1;

% Name of response keys (following MacOS-X naming scheme)
right_button = 'RightArrow'; % press when the caption is correct
left_button = 'LeftArrow'; % press when the caption is wrong
esc_button = 'ESCAPE'; % press (during the response period) to exit the experiment
continue_button = 'space'; % press (during pauses) to start new run

if eeg_mode == 1 
	trigger_delay_time = 0.01367;
	address=hex2dec('3FE0');
	%addpath('.\iosetup\');
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experimental parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Timing
data.paradigm.video_duration = 3; % 3s of video onscreen duration
data.paradigm.pre_baseline = 1; % 1s of pre-video baseline
data.paradigm.post_baseline = 0.25; % 0.25s of post-video baseline
data.paradigm.blink_time = 1.5; % 1.5s to blink
data.paradigm.response_time = 5; % up to 5s to respond

% Presentation structure
data.paradigm.trials_per_run = 66; % ~7m per run
data.paradigm.runs = 16; % ~2h per experiment
data.paradigm.questions_per_run = 6;

% Stimuli
data.paradigm.tot_test_videos = 102;
data.paradigm.tot_train_videos = 1000;
data.paradigm.presented_test_videos = 102;
data.paradigm.presented_train_videos = 250;
data.paradigm.video_repeats = 3;

% Horizontal (x) and vertical (y) visual angle
data.paradigm.vis_angle_x = 5;
data.paradigm.vis_angle_y = 5;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subject's info
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data.subject.id = input('Subject''s Number: --> ');
data.subject.age = input('Subject''s Age: --> ');
data.subject.sex = input('Subject''s Sex: --> ','s');
data.subject.session = input('Session: --> ');
start_run = input('Starting run: --> ');

% Create saving directory if not existing
output_save_dir = [save_dir, slash, 'sub-', sprintf('%02d', data.subject.id), ...
	slash, 'sess-', sprintf('%02d', data.subject.session)];
if ~exist(output_save_dir)
	mkdir(output_save_dir)
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the stimuli into a structure and define the presentation order
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create a structure with the train and test stimuli videos
data.stimuli = getStimuli(video_dir, metadata_dir);

% This function generates the pseudo-randomized stimuli order: in every
% sessions, each test video is repeated 3 times, and each quartile of the
% training videos is repeated 3 times (session 1 = first quartile; session 2 =
% second quartile; session 3 = third quartile; session 4 = fourth quartile).
% Furthermore, after 6 trials per run there is a task question.
%defineStimulusOrder(data, stim_order_dir);

% Load the (already defined) stimuli and task order
load([stim_order_dir, slash, 'stim_order_sub-', ...
	sprintf('%02d', data.subject.id), ...
	'_sess-', sprintf('%02d', data.subject.session)])
load([stim_order_dir, slash, 'task_trials_sub-', ...
	sprintf('%02d', data.subject.id), ...
	'_sess-', sprintf('%02d', data.subject.session)])

data.presentation_order = stim_order;
data.task_trials = task_trials;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Open Psychtoolbox and display instructions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[win, screenRect] = Screen('OpenWindow', screenWin, [159 162 166], [0 0 800 800]);
Screen('BlendFunction', win, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
%VBLSyncTest() %%%

HideCursor(0)
Screen('TextSize', win, floor(screenRect(3) / 50));
Screen('TextStyle', win, 0);

% Instructions
ListenChar(2);
instructions = strcat('Always gaze at the central fixation dot.\n\n', ...
	'Only blink after the blinking cue appears.\n\n', ...
	'end the blink while the blinking cue is still on the screen.\n\n', ...
	'Press the RIGHT ARROW key if the object/scene/action label is correct.\n\n', ...
	'Press the LEFT ARROW key if the object/scene/action label is wrong.\n\n\n\n', ...
	'Press any key to continue');
DrawFormattedText(win, instructions, 'center', 'center', [0 0 0]);

% Fixation cross
fixationCross(win, screenRect, dist_from_screen, screen_width)

% Flip screen and wait for keyboard input to continue
Screen('Flip', win);
KbPressWait(deviceNum);

% Refresh rate and flip interval
monitorFrameRate = FrameRate(win);
monitorFlipInterval = Screen('GetFlipInterval', win);
if round(monitorFrameRate) ~= 60
	Screen('CloseAll')
	ListenChar(0);
	error('The refresh rate is not 60 Hz. Please change it.');
end

% Milliseconds to be subtracted from the flipping time so to not miss the
% visual frame (and having to wait an additional 16ms (or more, or less,
% depending on the display device refresh rate) for the flip).
% If not needed, set to 0
fix_flip_time = monitorFlipInterval / 2;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define stimuli size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Stimuli size
[stimSizeX, stimSizeY] = visangle2stimsize(data.paradigm.vis_angle_x, ...
	data.paradigm.vis_angle_y, dist_from_screen, screen_width, screenRect(3));

% Coordinates of the destination rectangle
destRect = [(screenRect(3) / 2) - floor(stimSizeX / 2), ...
	(screenRect(4) / 2) - floor(stimSizeY / 2), ...
	(screenRect(3) / 2) + floor(stimSizeX / 2), ...
    (screenRect(4) / 2) + floor(stimSizeY / 2)];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize eyelink
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

el=EyelinkInitDefaults(win);

% Initialization of the connection with the Eyelink Gazetracker.
if ~EyelinkInit(eyelink_dummy)
    fprintf('Eyelink Init aborted.\n');
    cleanup; % cleanup function
    return;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Video presentation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for run = start_run:data.paradigm.runs

	data.run = run;
	data.trials = [];
	data.task = [];

	edfFile = ['r', sprintf('%02d', run),'.edf'];
	cd(output_save_dir);
	Eyelink('OpenFile',edfFile); % open file to write eye tracking data

	% Do eye tracker calibration
	if eyelink_dummy == 0
		EyelinkDoTrackerSetup(el);
		EyelinkDoDriftCorrection(el);
	end

	% Eyelink start recording
	Eyelink('StartRecording');
	WaitSecs(0.1);   

	for trial = 1:data.paradigm.trials_per_run+1 % Add catch trial at beginning since the first EEG trigger is usually not precise 

		% If first trial, play a random video (without saving it or sending triggers)
		if trial == 1
			random_video = randi(1000);
		end

		% Blank screen with fixation cross
		fixationCross(win, screenRect, dist_from_screen, screen_width)
		if trial == 1
			t1 = Screen('Flip', win);
		else
			t1 = Screen('Flip', win, t3+data.paradigm.blink_time);
		end

		% Open movie file
		if trial == 1
			movie_file = [data.stimuli(random_video).folder, slash, ...
				data.stimuli(random_video).name];   
		else
			movie_file = [data.stimuli(stim_order(trial-1, run)).folder, slash, ...
				data.stimuli(stim_order(trial-1, run)).name];
		end
		[movie, duration, fps, ~, ~, count]  = ...
			Screen('OpenMovie', win, movie_file, 0);

		% Pre-video baseline
		fixationCross(win, screenRect, dist_from_screen, screen_width)
		Screen('Flip', win, t1+data.paradigm.pre_baseline);

		% Start playback of movie. This will start the realtime playback clock and
		% playback of audio tracks, if any
		Screen('PlayMovie', movie, 1, 0, 1.0);
		onset_audio = GetSecs;
		f = 0; % frame counter
		frames_timing = zeros(1,count);
		% Display movie frames
		while 1
			% Return next frame in movie, in sync with current playback
			% time and sound
			% tex is either the positive texture handle or zero if no
			% new frame is ready yet in non-blocking mode (blocking == 0).
			% It is -1 if something went wrong and playback needs to be stopped
			[tex, disp_time] = Screen('GetMovieImage', win, movie);
			% Valid texture returned?
			if tex < 0
				% No, and there won't be any in the future, due to some
				% error. Abort playback loop
				break;
			end
			if tex == 0
				% No new frame in polling wait (blocking == 0). Just sleep
				% a bit and then retry
				WaitSecs(0.001);
				continue;
			end
			% Draw the new texture immediately to screen
			Screen('DrawTexture', win, tex, [], destRect);
			fixationCross(win, screenRect, dist_from_screen, screen_width)
			% Update display
			if f  == 0
				%onset_visuals = Screen('Flip', win);
				onset_visuals = Screen('Flip', win, disp_time-fix_flip_time);
				frames_timing(f+1) = onset_visuals;
				% Send EEG & eye-tracker triggers
				if eeg_mode == 1
					if trial ~= 1
						WaitSecs(trigger_delay_time);
						send_triggerIO64(address, data.stimuli(stim_order(trial-1, run)).eeg_trigger);
						Eyelink('Message', 'TRIAL_START %d', trial-1);
					end
				end
			else
				%frames_timing(f+1) = Screen('Flip', win);
				frames_timing(f+1) = Screen('Flip', win, disp_time-fix_flip_time);
			end
			% Release texture
			Screen('Close', tex);
			% Framecounter
			f = f + 1;
		end

		% Stop playback
		Screen('PlayMovie', movie, 0);
		offset_audio = GetSecs;
		% Blank screen with fixation cross
		fixationCross(win, screenRect, dist_from_screen, screen_width)
		offset_visuals = Screen('Flip', win);
		% Close movie object
		Screen('CloseMovie', movie);
		% Eyelink mark the end of the trial
		if trial ~= 1
			Eyelink('Message', 'TRIAL_END %d', trial-1);
		end

		% Save the trial info into the results structure
		if trial ~= 1
			data.trials(trial-1).run = run;
			data.trials(trial-1).trial = trial-1;
			data.trials(trial-1).video_number = ...
				data.stimuli(stim_order(trial-1, run)).video_number;
			data.trials(trial-1).eeg_trigger = ...
				data.stimuli(stim_order(trial-1, run)).eeg_trigger;
			data.trials(trial-1).partition = ...
				data.stimuli(stim_order(trial-1, run)).partition;
			data.trials(trial-1).name = ...
				data.stimuli(stim_order(trial-1, run)).name;
			data.trials(trial-1).duration = duration;
			data.trials(trial-1).fps = fps;
			data.trials(trial-1).count = count;
			data.trials(trial-1).onset_audio =  onset_audio;
			data.trials(trial-1).onset_visuals =  onset_visuals;
			data.trials(trial-1).offset_audio =  offset_audio;
			data.trials(trial-1).offset_visuals =  offset_visuals;
			data.trials(trial-1).audio_duration =  offset_audio - onset_audio;
			data.trials(trial-1).visuals_duration =  offset_visuals - onset_visuals;
			data.trials(trial-1).played_frames = f;
			data.trials(trial-1).frames_timing = frames_timing;
		end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Blink & task
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		if trial ~= 1

			% Task trials
			if task_trials(trial-1,run) == 1

				% Select the correct/incorrect object/scene/action label to present
				label_type  = rand(1);
				disp_right_label  = randi([0 1], 1);
				
				% Correct label
				if label_type <= (1/3)
					correct_label = ...
						data.stimuli(stim_order(trial-1, run)).object_correct;
				elseif label_type > (1/3) && label_type <= (2/3)
					correct_label = ...
						data.stimuli(stim_order(trial-1, run)).scene_correct;
				elseif label_type > (2/3)
					correct_label = ...
						data.stimuli(stim_order(trial-1, run)).action_correct;
				end

				% Incorrect label
				if label_type <= (1/3)
					wrong_labels = ...
						data.stimuli(stim_order(trial-1, run)).object_wrong;
				elseif label_type > (1/3) && label_type <= (2/3)
					wrong_labels = ...
						data.stimuli(stim_order(trial-1, run)).scene_wrong;
				elseif label_type > (2/3)
					wrong_labels = ...
						data.stimuli(stim_order(trial-1, run)).action_wrong;
				end
				idx_wrong = randi([1 numel(wrong_labels)], 1);
				incorrect_label = char(wrong_labels(idx_wrong));

				% Display the object/scene/action label
				if disp_right_label == 1
					displayed_label = correct_label;
				elseif disp_right_label == 0
					displayed_label = incorrect_label;
				end
				DrawFormattedText(win, displayed_label, 'center', ...
					screenRect(4) / 2.2, [0 0 0]);
				fixationCross(win, screenRect, dist_from_screen, screen_width)
				t2 = Screen('Flip', win, offset_audio+data.paradigm.post_baseline);

				% Collect the response
				while true
					[secs, keyCode] = KbPressWait(deviceNum, ...
						t2+data.paradigm.response_time);
					thisResp = KbName(keyCode);
					if strcmp(thisResp, left_button) || strcmp(thisResp, right_button) || ...
						strcmp(thisResp, esc_button) || isempty(thisResp)
						break
					end
				end
				if strcmp(thisResp, left_button) % wrong label
					response = 0;
				elseif strcmp(thisResp, right_button) % correct label
					response = 1;
				elseif isempty(thisResp) % no response
					response = 2;
				elseif strcmp(thisResp, esc_button)
					Eyelink('StopRecording');
					Eyelink('CloseFile');
					Screen('CloseAll')
					ListenChar(0);
				end

				% Evaluate the response correctness
				if disp_right_label == response
					correctness = 1;
				else
					correctness = 0;
				end

				% Put the behavioral results into a structure
				data.task(trial-1).run = run;
				data.task(trial-1).trial = trial-1;
				data.task(trial-1).task_trial = task_trials(trial-1,run);
				data.task(trial-1).presented_label = displayed_label;
				data.task(trial-1).correct_label = correct_label;
				data.task(trial-1).incorrect_label = incorrect_label;
				data.task(trial-1).response = response;
				data.task(trial-1).correctness = correctness;

				% Blinking period
				blink = sprintf('Blink');
				DrawFormattedText(win, blink, 'center', screenRect(4) / 2.2, [0 0 0]);
				fixationCross(win, screenRect, dist_from_screen, screen_width)
				t3 = Screen('Flip', win, secs+data.paradigm.post_baseline);

			% Non-task trials
			else % if task_trials(trial-1,run) == 1

				% Blinking period
				blink = sprintf('Blink');
				DrawFormattedText(win, blink, 'center', screenRect(4) / 2.2, [0 0 0]);
				fixationCross(win, screenRect, dist_from_screen, screen_width)
				t3 = Screen('Flip', win, offset_audio+data.paradigm.post_baseline);

                % Exit the experiment
                [secs, keyCode] = KbPressWait(deviceNum, t3+1);
                thisResp = KbName(keyCode);
                if strcmp(thisResp, esc_button)
                    Eyelink('StopRecording');
                    Eyelink('CloseFile');
                    Screen('CloseAll')
                    ListenChar(0);
                end

				% Put the behavioral results into a structure
				data.task(trial-1).run = run;
				data.task(trial-1).trial = trial-1;
				data.task(trial-1).task_trial = task_trials(trial-1,run);
				data.task(trial-1).presented_label = nan;
				data.task(trial-1).correct_label = nan;
				data.task(trial-1).incorrect_label = nan;
				data.task(trial-1).response = nan;
				data.task(trial-1).correctness = nan;

			end % if task_trials(trial-1,run) == 1

		else % if trial ~= 1

			% Blinking period
			blink = sprintf('Blink');
			DrawFormattedText(win, blink, 'center', screenRect(4) / 2.2, [0 0 0]);
			fixationCross(win, screenRect, dist_from_screen, screen_width)
			t3 = Screen('Flip', win, offset_audio+data.paradigm.post_baseline);

            % Exit the experiment
            [secs, keyCode] = KbPressWait(deviceNum, t3+1);
            thisResp = KbName(keyCode);
            if strcmp(thisResp, esc_button)
                Eyelink('StopRecording');
                Eyelink('CloseFile');
                Screen('CloseAll')
                ListenChar(0);
            end

		end % if trial ~= 1

	end % trial



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the data structure after each run
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	save([output_save_dir, slash, 'run-', sprintf('%03d', run), '.mat'], 'data')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Break
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% Eyelink wait a bit before stop recording 
	WaitSecs(3);
	Eyelink('StopRecording');
	Eyelink('CloseFile');
	Eyelink('ReceiveFile');

	if run < data.paradigm.runs
		Screen('Flip', win);
		text = strcat('END OF RUN %d/%d\n\n\nYou can now take a break.', ...
			'\n\n\n\nPress the space bar when you wish to continue.');
		end_of_run_text = sprintf(text, run, data.paradigm.runs);
		DrawFormattedText(win, end_of_run_text, 'center', 'center', [0 0 0]);
		fixationCross(win, screenRect, dist_from_screen, screen_width)
		Screen('Flip', win);
		while true
			[secs, keyCode] = KbPressWait(deviceNum);
			thisResp = KbName(keyCode);
			if strcmp(thisResp, continue_button)
				break
			end
		end
	end

end % run



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exit experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Eyelink('Shutdown');
Screen('CloseAll')
ListenChar(0);
