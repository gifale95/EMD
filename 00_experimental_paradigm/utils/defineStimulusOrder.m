function defineStimulusOrder(data, stim_order_dir, slash)
% This function generates the pseudo-randomized  stimulus order: in every
% sessions, each test video is repeated 3 times, and each quartile of the
% training videos is repeated 3 times (session 1 = first quartile; session 2 =
% second quartile; session 3 = third quartile; session 4 = fourth quartile).
% Furthermore, in 6 trials per run there is a task question.

% There are a total of 1102 videos: the first 1000 are the training videos and
% the last 102 the test videos. In each session, only 250 training videos are
% used
remainder = rem(data.subject.session-1, 4);
train_quartile_start = (data.paradigm.presented_train_videos * remainder) ...
	+ 1;
train_quartile_end = train_quartile_start + ...
	data.paradigm.presented_train_videos - 1;
train_videos = train_quartile_start:train_quartile_end;
test_quartile_start = data.paradigm.tot_train_videos + 1;
test_quartile_end = test_quartile_start + data.paradigm.tot_test_videos - 1;
test_videos = test_quartile_start:test_quartile_end;

% Define the random video order, making sure that no video is repeated
% in two consecutive trials
all_videos = [train_videos, test_videos];
video_order = [];
for r = 1:data.paradigm.video_repeats
	ordered_trials = Shuffle(all_videos);
	if r ~= 1
		while any(ismember(video_order(end-5:end), ordered_trials(1:5)))
			ordered_trials = Shuffle(all_videos);
		end
	end
	video_order = [video_order, ordered_trials];
end

% Randomly choose the 6 trials in each run after which the task question is
% asked
task_trials = zeros(data.paradigm.trials_per_run, data.paradigm.runs);
for r = 1:data.paradigm.runs
	idx_task = 1:data.paradigm.trials_per_run;
	idx_task = Shuffle(idx_task);
	idx_task = idx_task(1:data.paradigm.questions_per_run);
	task_trials(idx_task,r) = 1;
end

% Create the 2-D stimuli presentation array of shape: (Videos per run × Runs)
% Each entry indicates the corresponding video number to present
stim_order = zeros([data.paradigm.runs, data.paradigm.trials_per_run]);
counter = 1;
for r = 1:data.paradigm.runs
	for t = 1:data.paradigm.trials_per_run
			stim_order(r,t) = video_order(counter);
			counter = counter + 1;
	end
end
stim_order = transpose(stim_order);

% Save the stimuli and task order
save([stim_order_dir, slash, 'stim_order_sub-', ...
	sprintf('%02d', data.subject.id), ...
	'_sess-', sprintf('%02d', data.subject.session)], 'stim_order')
save([stim_order_dir, slash, 'task_trials_sub-', ...
	sprintf('%02d', data.subject.id), ...
	'_sess-', sprintf('%02d', data.subject.session)], 'task_trials')

end
