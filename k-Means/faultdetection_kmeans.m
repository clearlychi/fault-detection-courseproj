close all;
clc;
%% Data

faultydata = readtable('fault.csv');
nofaultdata = readtable('nofault.csv');

% Concatenate faulty and non-faulty datasets
Data = [faultydata; nofaultdata];

% Extract features
faultfree = Data(Data.faultNumber == 0, 4:end);

% Convert table to array
ffarray = table2array(faultfree);
%% k-means clustering

%once again, confusion matrices have been omitted so I don't have to sift
%through 20 extra graphs

% Mean-center features
ffscaled = zscore(ffarray);

% Define the number of clusters
k = 1;

% Manually implement K-means clustering

% Randomly initialize centroids
rng(42); % for reproducibility
centroid_indices = randperm(size(ffscaled, 1), k);
C = ffscaled(centroid_indices, :);

% Initialize cluster indices
idx = zeros(size(ffscaled, 1), 1);

% Define maximum number of iterations
maxiters = 100;

for iter = 1:maxiters
    % Assign each data point to the nearest centroid
    for i = 1:size(ffscaled, 1)
        distances = sum((ffscaled(i, :) - C).^2, 2);
        [~, idx(i)] = min(distances);
    end
    
    % Update centroids
    newcent = zeros(size(C));
    for j = 1:k
        clusterpoints = ffscaled(idx == j, :);
        if ~isempty(clusterpoints)
            newcent(j, :) = mean(clusterpoints, 1);
        else
            % If cluster is empty, keep the centroid unchanged
            newcent(j, :) = C(j, :);
        end
    end
    
    % Check for convergence
    if isequal(C, newcent)
        break;
    end
    
    % Update centroids
    C = newcent;
end
% idx now contains the cluster indices assigned to each observation
% C contains the final centroids of the clusters

% Use 'distanceFromCenter' function
% Obtain distances for fault-free instances
ffdist = distanceFromCenter(C, ffscaled);

% Plot histogram of distances
figure
histogram(ffdist, 100, 'Normalization', 'probability');
xlabel('Distance');
ylabel('Probability');
title('Fault-Free Distance Distribution');

% Calculate threshold
mu = mean(ffdist);
stdev = std(ffdist);
threshold = mu + 2 * stdev;

% Loop over different fault numbers
f1all = [];
accall = [];
for fnum = 0:20
    % Filter data for the current fault number and simulation run
    temp = Data(Data.faultNumber == fnum & Data.simulationRun == 1, 4:end);

    % Convert table to array and mean center features
    temparray = table2array(temp);
    xfaulty = zscore(temparray);

    % Obtain distances from the cluster center
    faultydist = distanceFromCenter(C, xfaulty);

    % Plot distances and threshold
    figure;
    plot(faultydist);
    hold on;
    yline(threshold, '--', 'Threshold');
    title(['Fault Number = ', num2str(fnum)]);
    xlabel('Sample');
    ylabel('Distance');

    % Combine results based on threshold
    result = faultydist > threshold;

    % Set true values
    if fnum == 0
        ytrue = zeros(size(result));
    else
        ytrue = ones(size(result));
        ytrue(1:21) = 0; % Fault introduced after 20th sample
    end

    % Ensure result and y_true have the same size
    minlen = min(length(result), length(ytrue));
    result = result(1:minlen);
    ytrue = ytrue(1:minlen);

    % Convert result to the same type as y_true
    result = double(result);

    % Calculate F1-score and accuracy
    truepos = sum(result & ytrue);
    falsepos = sum(result & ~ytrue);
    falseneg = sum(~result & ytrue);

    % Calculate precision and recall
    precision = truepos / (truepos + falsepos);
    recall = truepos / (truepos + falseneg);

    % Check if precision and recall are both zero
    if precision == 0 && recall == 0
        F1 = 0;
    else
        F1 = 2 * (precision * recall) / (precision + recall);
    end

    accuracy = sum(result == ytrue) / length(result);

    f1all = [f1all, F1];
    accall = [accall, accuracy];

    disp(['The F1-Score for Fault_num = ', num2str(fnum), ' is ', num2str(F1)]);
    disp(['The Accuracy-Score for Fault_num = ', num2str(fnum), ' is ', num2str(accuracy)]);

    % % Plot confusion matrix
    % figure;
    % cm = confusionmat(ytrue, result);
    % confusionchart(cm);
end

% Loop over different fault numbers for 'combine_result'
for Fnum = setdiff(0:20, [3, 9])
    % Filter data for the current fault number and random simulation run
    tempdf = Data(Data.faultNumber == Fnum & Data.simulationRun == randi([1, 500]), 4:end);

    % Convert table to array and mean center features
    temparray = table2array(tempdf);
    xfaulty = zscore(temparray);

    % Obtain distances from the cluster center
    faultydist = distanceFromCenter(C, xfaulty);

    % Combine results based on threshold
    result = faultydist > threshold;

    % Set true values
    if Fnum == 0
        ytrue = zeros(size(result));
    else
        ytrue = ones(size(result));
        ytrue(1:21) = 0; % Fault introduced after 20th sample
    end

    % Ensure result and y_true have the same size
    minlen = min(length(result), length(ytrue));
    result = result(1:minlen);
    ytrue = ytrue(1:minlen);

    % Convert result to the same type as y_true
    result = double(result);

    % Calculate F1-score and accuracy
    truepos = sum(result & ytrue);
    falsepos = sum(result & ~ytrue);
    falseneg = sum(~result & ytrue);

    % Calculate precision and recall
    precision = truepos / (truepos + falsepos);
    recall = truepos / (truepos + falseneg);

    % Check if precision and recall are both zero
    if precision == 0 && recall == 0
        F1 = 0;
    else
        F1 = 2 * (precision * recall) / (precision + recall);
    end

    accuracy = sum(result == ytrue) / length(result);

    f1all = [f1all, F1];
    accall = [accall, accuracy];
end

% Calculate average F1 score and average accuracy score
average_F1 = mean(f1all, 'omitnan');
average_accuracy = mean(accall, 'omitnan');

% Display average results
disp(['Average F1 Score: ', num2str(average_F1)]);
disp(['Average Accuracy Score: ', num2str(average_accuracy)]);