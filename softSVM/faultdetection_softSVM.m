clear figures
close all
clear variables
clc;
%% Importing data
%Importing fault free and faulty data;
nofaultdata = importdata("nofault.csv");
faultdata = importdata("fault.csv");

faultydatapoints = length(faultdata.data);
normaldatapoints = length(nofaultdata.data);

data = [faultdata.data;nofaultdata.data];

X = data(:,4:end);

%% SVM Portion

%Confusion matrices and histograms are commented out for ease of working,
%preventing the need to click through 20 more graphs
[mdl,tf,scores] = ocsvm(X(faultydatapoints+1:end,:),ContaminationFraction=0,KernelScale='auto',StandardizeData=true, NumExpansionDimensions="auto");

% Loop over different fault numbers
f1all = [];
accall = [];
trueposall = [];
truenegall = [];
falseposall = [];
falsenegall = [];
counter = 0;
for fnum = 0:20

    % Filter data for the current fault number and simulation run
    temptestdata = data(data(:,1)==fnum & data(:,2) == 1, 4:end);%== 

    [tf_test,s_test] = isanomaly(mdl,temptestdata);
    

    %Plot distances and threshold
    figure;
    plot(s_test);
    hold on;
    yline(mdl.ScoreThreshold, '--', 'Threshold');
    title(['Fault Number = ', num2str(fnum)]);
    xlabel('Sample');
    ylabel('Distance');

    figure;

    hold on
    histogram(s_test)
    histogram(scores)
    xline(mdl.ScoreThreshold,"r-",join(["Threshold" mdl.ScoreThreshold]))
    hold off
    legend("Training Fault Data","Training Fault Free Data",Location="north")

    % Combine results based on threshold
    result = s_test > mdl.ScoreThreshold;

    % Set true values
    if fnum == 0
        ytrue = false(size(result));
    else
        ytrue = true(size(result));
        ytrue(1:21) = false; % Fault introduced after 20th sample
    end

    % Ensure result and y_true have the same size
    minlen = min(length(result), length(ytrue));
    result = result(1:minlen);
    ytrue = ytrue(1:minlen);

    % Calculate F1-score and accuracy
    truepos = sum(result & ytrue);
    falsepos = sum(result & ~ytrue);
    falseneg = sum(~result & ytrue);
    trueneg = sum(~result & ~ytrue);

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
    trueposall = [trueposall, truepos];
    truenegall = [truenegall, trueneg];
    falseposall = [falseposall, falsepos];
    falsenegall = [falsenegall, falseneg];

    disp(['The F1-Score for Fault_num = ', num2str(fnum), ' is ', num2str(F1)]);
    disp(['The Accuracy-Score for Fault_num = ', num2str(fnum), ' is ', num2str(accuracy)]);

    % % Plot confusion matrix
    % figure;
    % cm = confusionmat(ytrue, result);
    % confusionchart(cm);
end

% Loop over different fault numbers for 'combine_result'

for Fnum = setdiff(0:20, [3, 9])
    faultnumbers(counter+1)=fnum;
    counter = counter +1;
    SimulationNumber = randi([1,500],1);
    % Filter data for the current fault number and simulation run
    temptestdata = data(data(:,1) == fnum & data(:,2) == SimulationNumber, 4:end);

    [tf_test,s_test] = isanomaly(mdl,temptestdata);


    % Combine results based on threshold
    result = s_test > mdl.ScoreThreshold;


    % Set true values
    if fnum == 0
        ytrue = false(size(result));
    else
        ytrue = true(size(result));
        ytrue(1:21) = false; % Fault introduced after 20th sample
    end

    % Ensure result and y_true have the same size
    minlen = min(length(result), length(ytrue));
    result = result(1:minlen);
    ytrue = ytrue(1:minlen);

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
avgf1 = mean(f1all, 'omitnan');
avgacc = mean(accall, 'omitnan');

% Display average results
disp(['Average F1 Score: ', num2str(avgf1)]);
disp(['Average Accuracy Score: ', num2str(avgacc)]);

% %h1 = histogram(scores,NumBins=100);
% trueposall = trueposall';
% truenegall = truenegall';
% falseposall = falseposall';
% falsenegall = falsenegall';
