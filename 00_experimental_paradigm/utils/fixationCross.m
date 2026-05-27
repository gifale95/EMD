function fixationCross(win, screenRect, dist_from_screen, screen_width)
% Red fixation cross centered and had a diameter of 28 pixels, a
% line-width of 8 pixels, and a white circle of diameter 8 pixels at the center,
% for a total visual angle of 0.52 degrees.

% Fixation cross size
stimSizeX = visangle2stimsize(0.52, 0.52, dist_from_screen, ...
	screen_width, screenRect(3));

% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(screenRect);

% Here we set the size of the arms of our fixation cross
fixCrossDimPix = stimSizeX / 2;

% Now we set the coordinates (these are all relative to zero we will let
% the drawing routine center the cross in the center of our monitor for us)
xCoords = [-fixCrossDimPix fixCrossDimPix 0 0];
yCoords = [0 0 -fixCrossDimPix fixCrossDimPix];
allCoords = [xCoords; yCoords];

% Set the line width for our fixation cross
lineWidthPix = 7;

% Draw the fixation cross in red, set it to the center of our screen and
% set good quality antialiasing
Screen('DrawLines', win, allCoords, lineWidthPix, [255 0 0], [xCenter, yCenter], 2);
Screen('FillOval', win, [255 255 255], [xCenter-lineWidthPix/2, ...
	yCenter-lineWidthPix/2, xCenter+lineWidthPix/2, yCenter+lineWidthPix/2]);

end