function stimuli = getStimuli(video_dir, metadata_dir)
% This function generates a structure with the training and test stimuli videos.

	% Get videos metadata
	fid = fopen(metadata_dir); 
	raw = fread(fid,inf); 
	str = char(raw'); 
	fclose(fid); 
	val = jsondecode(str);
	fn = fieldnames(val);

	% List all unique object labels
	all_object_labels = {};
	counter = 1;
	for v = 1:numel(fn)
		for s1=1:length(val.(fn{v}).objects)
				for  s2=1:length(val.(fn{v}).objects{s1})
					if ~any(strcmp(all_object_labels, val.(fn{v}).objects{s1}(s2))) &&  ~strcmp(val.(fn{v}).objects{s1}(s2), '--')
						all_object_labels(counter) = val.(fn{v}).objects{s1}(s2);
						counter = counter + 1;
					end
				end
		end
	end
	
	% List all unique scene labels
	all_scene_labels = {};
	counter = 1;
	for v = 1:numel(fn)
		for s1=1:length(val.(fn{v}).scenes)
				if ~any(strcmp(all_scene_labels, val.(fn{v}).scenes(s1)))
					all_scene_labels(counter) = val.(fn{v}).scenes(s1);
					counter = counter + 1;
				end
		end
	end

	% List all unique action labels
	all_action_labels = {};
	counter = 1;
	for v = 1:numel(fn)
		for s1 = 1:numel(val.(fn{v}).actions)
			if ~any(strcmp(all_action_labels, val.(fn{v}).actions(s1)))
				all_action_labels(counter) = val.(fn{v}).actions(s1);
				counter = counter + 1;
			end
		end
	end

	% Get video file names
	video_files = dir(video_dir);
	video_files = video_files(3:end);

	for v = 1:length(video_files)

		% Video number
		stimuli(v).video_number = v;

		% Only keep the first two video condition digits for the EEG trigger
		if v < 100
			stimuli(v).eeg_trigger = v;
		else
			eeg_trigger = string(v);
			eeg_trigger = char(eeg_trigger);
			stimuli(v).eeg_trigger = str2double(eeg_trigger(1:2));
		end

		% The first 1000 videos belong to the training partition
		if v <= 1000
			stimuli(v).partition = 'train';
		elseif v > 1000
			stimuli(v).partition = 'test';
		end

		% Video file name
		stimuli(v).name = video_files(v).name;
		% Video folder
		stimuli(v).folder = video_files(v).folder;

		% Video metadata: select the object label most repeated
		objects = {};
		counter = 1;
			for s1=1:length(val.(fn{v}).objects)
					for  s2=1:length(val.(fn{v}).objects{s1})
						if ~strcmp(val.(fn{v}).objects{s1}(s2), '--')
							objects(counter) = val.(fn{v}).objects{s1}(s2);
							counter = counter + 1;
						end
					end
			end
		unique_objects = unique(objects, 'stable');
		freq = cellfun(@(x) sum(ismember(objects, x)), unique_objects, 'un', 0);
		[~, idx] = max(cell2mat(freq));
		stimuli(v).object_correct = char(unique_objects(idx));
		
		% Wrong action labels (for the task)
		wrong_object_labels = {};
		counter = 1;
		for l = 1:numel(all_object_labels)
			if ~any(strcmp(objects, all_object_labels(l)))
				wrong_object_labels(counter) = all_object_labels(l);
				counter = counter + 1;
			end
		end
		stimuli(v).object_wrong = wrong_object_labels;
		
		% Video metadata: select the scene label most repeated
		scenes = val.(fn{v}).scenes;
		unique_scenes = unique(scenes, 'stable');
		freq = cellfun(@(x) sum(ismember(scenes, x)), unique_scenes, 'un', 0);
		[~, idx] = max(cell2mat(freq));
		stimuli(v).scene_correct = char(unique_scenes(idx));
		
		% Wrong scene labels (for the task)
		wrong_scene_labels = {};
		counter = 1;
		for l = 1:numel(all_scene_labels)
			if ~any(strcmp(scenes, all_scene_labels(l)))
				wrong_scene_labels(counter) = all_scene_labels(l);
				counter = counter + 1;
			end
		end
		stimuli(v).scene_wrong = wrong_scene_labels;
		
		% Video metadata: select the action label most repeated (among the 5
		% labels)
		actions = val.(fn{v}).actions;
		unique_actions = unique(actions, 'stable');
		freq = cellfun(@(x) sum(ismember(actions, x)), unique_actions, 'un', 0);
		[~, idx] = max(cell2mat(freq));
		stimuli(v).action_correct = char(unique_actions(idx));
		
		% Wrong action labels (for the task)
		wrong_action_labels = {};
		counter = 1;
		for l = 1:numel(all_action_labels)
			if ~any(strcmp(actions, all_action_labels(l)))
				wrong_action_labels(counter) = all_action_labels(l);
				counter = counter + 1;
			end
		end
		stimuli(v).action_wrong = wrong_action_labels;

	end

end