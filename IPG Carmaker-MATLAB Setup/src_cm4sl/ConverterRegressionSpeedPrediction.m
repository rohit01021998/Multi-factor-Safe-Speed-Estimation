function predicted_speed = ConverterRegressionSpeedPrediction(turnType, roadType, envType, visibility, pedestrian, traffic, dayNight, weather)
    % Ensure Python modules are imported
    py.importlib.import_module('joblib');
    py.importlib.import_module('pandas');
    py.importlib.import_module('numpy');
    
    % Load the pre-trained model
    model_path = 'speed_limit_regression_model_without_accident_sim_coeff_for_reallife.pkl';
    loaded_model = py.joblib.load(model_path);
    
    % For RandomForestRegressor, extract feature importances
    if isa(loaded_model, 'py.sklearn.ensemble._forest.RandomForestRegressor')
        feature_importances = loaded_model.feature_importances_;
        model_struct.feature_importances = double(py.array(feature_importances));
    elseif isa(loaded_model, 'py.sklearn.pipeline.Pipeline')
        % If it's a pipeline, access the final estimator
        final_estimator = loaded_model.named_steps{'regressor'}; 
        
        if isa(final_estimator, 'py.sklearn.ensemble._forest.RandomForestRegressor')
            feature_importances = final_estimator.feature_importances_;
            model_struct.feature_importances = double(py.array(feature_importances));
        end
    end
    
    % Save the model structure
    save('speed_limit_regression_model.mat', '-struct', 'model_struct');
    
    % Prepare input data
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