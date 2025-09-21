function predicted_speed = regressionSpeedPrediction(turnType, roadType, envType, visibility, pedestrian, traffic, dayNight, weather)
    % MATLAB function to predict speed limit based on input parameters using a pre-trained model

    % Ensure Python modules are imported
    py.importlib.import_module('joblib');
    py.importlib.import_module('pandas');
    py.importlib.import_module('numpy');

    % Load the pre-trained model
    model_path = 'speed_limit_regression_model_without_accident_sim_coeff_for_reallife.pkl';
    loaded_model = py.joblib.load(model_path);

    % Prepare input data using function parameters
    input_dict = py.dict(...
        pyargs(...
            'TurnType', py.list({turnType}), ...
            'RoadType', py.list({roadType}), ...
            'EnvType', py.list({envType}), ...
            'VisibilityConditions', py.list({visibility}), ...
            'PedestrianLevel', py.list({pedestrian}), ...
            'TrafficLevel', py.list({traffic}), ...
            'DayNight', py.list({dayNight}), ...
            'WeatherConditions', py.list({weather}) ...
        ) ...
    );

    % Create pandas DataFrame
    input_data = py.pandas.DataFrame(input_dict);

    % Make prediction
    predicted_speed_limit = loaded_model.predict(input_data);

    % Convert prediction to MATLAB format
    predicted_speed_list = predicted_speed_limit.tolist();
    predicted_speed = cellfun(@double, cell(predicted_speed_list));

    % Return the predicted speed limit
    predicted_speed = predicted_speed(1);
end