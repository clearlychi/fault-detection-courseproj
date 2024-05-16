% Function to combine result based on threshold
function result = combineresult(x, threshold)
    % If reconstruction loss is greater than threshold, then it's a fault
    result = x > threshold;
end