% Input Parameter Definitions
turnTypes = {'Gentle turn', 'Intersection', 'Sharp turn', 'Straight'};
roadTypes = {'Wide', 'Medium', 'Narrow'};
envTypes = {'City', 'Highway', 'Residential'};
visibilityConditions = {'Low', 'High'};
pedestrianLevels = {'Low', 'High', 'Moderate'};
trafficLevels = {'Low', 'High', 'Moderate'};
dayNights = {'Day', 'Night'};
weatherConditions = {'Clear', 'Fogg', 'Rain'};

% Generate all possible combinations
[turnGrid, roadGrid, envGrid, visibilityGrid, pedestrianGrid, trafficGrid, dayNightGrid, weatherGrid] = ndgrid(...
    1:length(turnTypes), 1:length(roadTypes), 1:length(envTypes), 1:length(visibilityConditions), ...
    1:length(pedestrianLevels), 1:length(trafficLevels), 1:length(dayNights), 1:length(weatherConditions));

% Reshape grids to form all combinations
inputCombinations = [turnGrid(:), roadGrid(:), envGrid(:), visibilityGrid(:), ...
                     pedestrianGrid(:), trafficGrid(:), dayNightGrid(:), weatherGrid(:)];

% Total number of combinations
totalCombinations = size(inputCombinations, 1);

% Preallocate numeric lookup table
numericLookupTable = zeros(totalCombinations, 9);

% Start time tracking
startTime = tic;

% Progress bar setup
progressInterval = round(totalCombinations * 0.05);  % 5% intervals
fprintf('Starting speed prediction lookup table generation...\n');

% Loop through combinations with progress tracking
for i = 1:totalCombinations
    % Extract the combination for this row
    combination = inputCombinations(i, :);
    
    % Map indices to actual values
    turnType = turnTypes{combination(1)};
    roadType = roadTypes{combination(2)};
    envType = envTypes{combination(3)};
    visibility = visibilityConditions{combination(4)};
    pedestrian = pedestrianLevels{combination(5)};
    traffic = trafficLevels{combination(6)};
    dayNight = dayNights{combination(7)};
    weather = weatherConditions{combination(8)};
    
    % Make the prediction
    predicted_speed = regressionSpeedPrediction(turnType, roadType, ...
                                                envType, visibility, ...
                                                pedestrian, traffic, ...
                                                dayNight, weather);
    
    % Store numerical results
    numericLookupTable(i, :) = [combination(1)-1, combination(2)-1, combination(3)-1, ...
                                combination(4)-1, combination(5)-1, combination(6)-1, ...
                                combination(7)-1, combination(8)-1, predicted_speed];
    
    % Progress tracking
    if mod(i, progressInterval) == 0
        progress = (i / totalCombinations) * 100;
        elapsedTime = toc(startTime);
        estimatedTotalTime = (elapsedTime / progress) * 100;
        estimatedRemainingTime = estimatedTotalTime - elapsedTime;
        
        fprintf('Progress: %.2f%% (Elapsed: %.2f s, Estimated Remaining: %.2f s)\n', ...
                progress, elapsedTime, estimatedRemainingTime);
    end
end

% Save numeric lookup table
csvwrite('numeric_lookup_table.csv', numericLookupTable);

% Final completion message
fprintf('Lookup table generation complete. Total time: %.2f seconds\n', toc(startTime));