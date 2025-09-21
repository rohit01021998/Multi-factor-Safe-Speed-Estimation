function outputValueSpeedBasedData = lookupValueFunction(input1, input2, input3, input4, input5, input6, input7, input8)
    % Access numericlookuptable from the base workspace
        numericlookuptable1 = evalin('base', 'numericlookuptable1');
    
    % Check if it is a table
    if ~istable(numericlookuptable1)
        error('numericlookuptable must be a table.');
    end
    
    % Combine the 8 input values into a single row vector
    inputValues = [input1, input2, input3, input4, input5, input6, input7, input8];
    
    % Extract the first 8 columns of the table as a matrix
    tableData = numericlookuptable1{:, 1:8}; % Convert table columns to a matrix
    
    % Find the row where the first 8 columns match the input values
    matchingRow = all(tableData == inputValues, 2);
    
    % Check if a matching row was found
    if any(matchingRow)
        % Return the value in the 9th column of the matching row
        outputValueSpeedBasedData = numericlookuptable1{matchingRow, 9}; % Access the 9th column
    else
        % If no match is found, return NaN or an appropriate error message
        outputValueSpeedBasedData = NaN;
        warning('No matching row found for the given input values.');
    end
end