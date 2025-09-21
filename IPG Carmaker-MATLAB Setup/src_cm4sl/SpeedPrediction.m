classdef SpeedPrediction < matlab.System
    % SpeedPrediction: System object for predicting safe speed using a Python model.
    % This class handles both simulation (with Python interaction) and code generation.

    % Public, tunable properties
    properties
        % Add tunable properties here if needed
    end

    % Discrete state properties
    properties(DiscreteState)
        % Add discrete state properties if needed
    end

    % Pre-computed constants (used for simulation only)
    properties(Access = private, Nontunable)
        model % Python model loaded during simulation
    end

    methods(Access = protected)
        function setupImpl(obj)
            if isSimulating()
                % Load Python model for simulation
                coder.extrinsic('py.importlib.import_module');
                coder.extrinsic('py.joblib.load');
                py.importlib.import_module('joblib');
                obj.model = py.joblib.load('speed_limit_regression_model_without_accident_sim_coeff_for_reallife.pkl');
            end
        end

        function predicted_speed = stepImpl(obj, turnType, roadType, envType, visibility, pedestrian, traffic, dayNight, weather)
            if isSimulating()
                % Use Python model for prediction during simulation
                coder.extrinsic('py.dict', 'py.pandas.DataFrame');

                % Map input indices to strings
                turnTypeStr = mapInput(turnType, {'Gentle turn', 'Intersection', 'Sharp turn', 'Straight'});
                roadTypeStr = mapInput(roadType, {'Wide', 'Medium', 'Narrow'});
                envTypeStr = mapInput(envType, {'City', 'Highway', 'Residential'});
                visibilityStr = mapInput(visibility, {'Low', 'High'});
                pedestrianStr = mapInput(pedestrian, {'Low', 'High', 'Moderate'});
                trafficStr = mapInput(traffic, {'Low', 'High', 'Moderate'});
                dayNightStr = mapInput(dayNight, {'Day', 'Night'});
                weatherStr = mapInput(weather, {'Clear', 'Fog', 'Rain'});

                % Prepare input dictionary
                input_dict = py.dict(pyargs(...
                    'TurnType', py.list({turnTypeStr}), ...
                    'RoadType', py.list({roadTypeStr}), ...
                    'EnvType', py.list({envTypeStr}), ...
                    'VisibilityConditions', py.list({visibilityStr}), ...
                    'PedestrianLevel', py.list({pedestrianStr}), ...
                    'TrafficLevel', py.list({trafficStr}), ...
                    'DayNight', py.list({dayNightStr}), ...
                    'WeatherConditions', py.list({weatherStr}) ...
                ));

                % Convert to pandas DataFrame
                input_data = py.pandas.DataFrame(input_dict);

                % Predict using the Python model
                predicted_speed_limit = obj.model.predict(input_data);
                predicted_speed_list = predicted_speed_limit.tolist();
                predicted_speed = double(predicted_speed_list{1});
            else
                % Placeholder value for code generation
                predicted_speed = 0; % Replace with a meaningful default value if needed
            end
        end

        function resetImpl(obj)
            % Reset discrete-state properties if needed
        end
    end

    methods(Access = private)
        function str = mapInput(~, idx, strList)
            % Helper function to map numeric input indices to strings
            if idx >= 0 && idx < numel(strList)
                str = strList{idx + 1}; % MATLAB is 1-based indexed
            else
                str = 'Unknown'; % Default value for invalid indices
            end
        end
    end

    methods(Access = protected)
        function num = getNumInputsImpl(~)
            % Specify the number of input ports
            num = 8;
        end

        function num = getNumOutputsImpl(~)
            % Specify the number of output ports
            num = 1;
        end

        function flag = isInputSizeMutableImpl(~, ~)
            % Inputs are not size-mutable
            flag = false;
        end
    end
end
