faceFrame = [];
% Load video
videoFile = 'C:\Users\user\Desktop\_\video2.mp4';
videoReader = vision.VideoFileReader(videoFile);

% Create face detector object
faceDetector = vision.CascadeObjectDetector();

% Create video player object
videoPlayer = vision.VideoPlayer();

% Initialize variables
pTime = 0;
pauseDuration = 0.01; % Adjust this value to change the playback speed

while ~isDone(videoReader)
    % Read frame from video
    frame = step(videoReader);
    
    % Convert frame to grayscale
    grayFrame = rgb2gray(frame);
    
    % Detect faces
    bbox = step(faceDetector, grayFrame);
    
    if ~isempty(bbox)
        % Draw bounding box around faces
        faceFrame = insertShape(frame, 'Rectangle', bbox, 'Color', 'red', 'LineWidth', 2);
        
        for i = 1:size(bbox, 1)
            % Convert bbox to roi format
            roi = [bbox(i, 1), bbox(i, 2), bbox(i, 3), bbox(i, 4)];
            
            % Check if the ROI is within image boundaries
            if roi(1) > 0 && roi(2) > 0 && roi(1) + roi(3) <= size(grayFrame, 2) && roi(2) + roi(4) <= size(grayFrame, 1)
                % Detect face landmarks
                facePoints = detectMinEigenFeatures(grayFrame, 'ROI', roi);
                
                % Convert facePoints to an M-by-2 matrix
                faceLandmarks = facePoints.Location;
                
                % Draw face landmarks on the frame
                faceFrame = insertMarker(faceFrame, faceLandmarks, 'Color', 'blue', 'Size', 5);
            end
        end
    end
    
    % Calculate and display FPS
    cTime = tic();
    fps = 1 / (cTime - pTime);
    pTime = cTime;
    faceFrame = insertText(faceFrame, [10 65], ['FPS: ' num2str(int32(fps))], 'FontSize', 16, 'BoxColor', 'blue', 'BoxOpacity', 0.8);
    
    % Display the frame
    step(videoPlayer, faceFrame);
    pause(pauseDuration); % Adjust the pause duration to change the playback speed
end

% Release resources
release(videoReader);
release(videoPlayer);
release(faceDetector);
