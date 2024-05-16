clc;
clear all;
close all;

%% Data

faultydata = readtable('fault.csv');
nofaultdata = readtable('nofault.csv');

% Concatenating faulty and non-faulty datasets
Data = [faultydata; nofaultdata];

faultfree = Data(Data.faultNumber == 0, 4:end);
faultfreearray = table2array(faultfree);
%% PCA

% Mean-center features using zscore
faultfreescaled = zscore(faultfreearray);

%taking economy SVD, reporting principal components
[U, Sigma, V] = svd(faultfreescaled,"econ");
% Compute the cumulative explained variance ratio
explainedvarratio = cumsum(diag(Sigma).^2) / sum(diag(Sigma).^2);

% Determine the number of principal components capturing 90% of the variance
ncomponents = find(explainedvarratio >= 0.90, 1);

% Retain principal components capturing 90% of variance
components = V(:, 1:ncomponents);
%calculate reconstruction error of the reconstructed dataset
reconerror = reconloss(components,faultfreescaled);
%plotting histogram of reconstruction error distribution
figure;
histogram(reconerror,100);
xlabel('Reconstruction Error');
ylabel('# of data points');
title('Histogram of Reconstruction Errors');

% Calculate threshold (using 2 standard deviations in this case, gave the
% most reasonable results when tested)
mu = mean(reconerror);
stddev = std(reconerror);
threshold = mu + 2 * stddev;

% Now looping over fault numbers to plot each fault
F1faults = [];
Accfaults = [];
for Fnum = 0:20
    % Current fault number
    temp = Data(Data.faultNumber == Fnum & Data.simulationRun == 1, 4:end);

    % Convert table to array and mean center
    temparray = table2array(temp);
    Xfaulty = zscore(temparray);

    % PCA reconstruction loss
    faultyreconloss = reconloss(components, Xfaulty);

    % Plotting the reconstruction losses along the threshold and the point
    % where the fault was introduced
    figure;
    plot(faultyreconloss);
    hold on;
    yline(threshold, 'r'); % Threshold value
    xline(20, 'g'); % Time of fault introduction
    title(['Fault Number = ', num2str(Fnum)]);
    xlabel('Time');
    ylabel('Reconstruction Loss');
    legend('Reconstruction Loss', 'Threshold', 'Time of Fault Introduction');
    hold off;

    % Combine results based on threshold
    result = faultyreconloss > threshold;

    % Set true values
    if Fnum == 0
        ytrue = zeros(size(result));
    else
        ytrue = ones(size(result));
        ytrue(1:21) = 0; % Fault introduced after 20th sample
    end

    % Making the sizes match so the arrays can multiply
    minlen = min(length(result), length(ytrue));
    result = result(1:minlen);
    ytrue = ytrue(1:minlen);
    result = double(result);

    % Calculate F1-score and accuracy
    tp = sum(result & ytrue);
    fp = sum(result & ~ytrue);
    fn = sum(~result & ytrue);

    % Calculate precision and recall
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);

    % Check if precision and recall are both zero
    if precision == 0 && recall == 0
        F1 = 0;
    else
        F1 = 2 * (precision * recall) / (precision + recall);
    end

    accuracy = sum(result == ytrue) / length(result);

    F1faults = [F1faults, F1];
    Accfaults = [Accfaults, accuracy];

    disp(['The F1-Score for Fault_num = ', num2str(Fnum), ' is ', num2str(F1)]);
    disp(['The Accuracy-Score for Fault_num = ', num2str(Fnum), ' is ', num2str(accuracy)]);
    %Space left for confusion matrix bc i didn't do that part yet
    % figure;
    % cm = confusionmat(ytrue, result);
    % confusionchart(cm);
end

% Loop over different fault numbers for 'combine_result'
for Fnum = setdiff(0:20, [3, 9])
    % Using random simulation run for each fault number
    tempdata = Data(Data.faultNumber == Fnum & Data.simulationRun == randi([1, 500]), 4:end);

    % Convert table to array and mean center features
    temparray = table2array(tempdata);
    Xfaulty = zscore(temparray);

    % Reconstruction loss from PCA 
    faultyreconloss = reconloss(components, Xfaulty);

    % Combine results based on threshold
    result = faultyreconloss > threshold;

    % Set true values
    if Fnum == 0
        ytrue = zeros(size(result));
    else
        ytrue = ones(size(result));
        ytrue(1:21) = 0; % Fault introduced after 20th sample
    end

    % Size compatibility for array multiplication
    minlen = min(length(result), length(ytrue));
    result = result(1:minlen);
    ytrue = ytrue(1:minlen);
    result = double(result);

    % Calculate F1-score and accuracy
    tp = sum(result & ytrue);
    fp = sum(result & ~ytrue);
    fn = sum(~result & ytrue);

    % Calculate precision and recall
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);

    % Check if precision and recall are both zero
    if precision == 0 && recall == 0
        F1 = 0;
    else
        F1 = 2 * (precision * recall) / (precision + recall);
    end

    accuracy = sum(result == ytrue) / length(result);

    F1faults = [F1faults, F1];
    Accfaults = [Accfaults, accuracy];
end

% Calculate average F1 score and average accuracy score
average_F1 = mean(F1faults, 'omitnan');
average_accuracy = mean(Accfaults, 'omitnan');

% Display average results
disp(['Average F1 Score: ', num2str(average_F1)]);
disp(['Average Accuracy Score: ', num2str(average_accuracy)]);
