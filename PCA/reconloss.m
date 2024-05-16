% Function to calculate the reconstruction loss when projecting data
function [faultyreconloss] = reconloss(principalcomponents,X)
% 'principalcomponents' is the matrix of principal components obtained from PCA
    % 'X' is the data matrix
    
    % Project data onto principal components
    Xpca = X * principalcomponents;
    
    % Reconstruct data
    Xrecon = Xpca * principalcomponents';
    
    % Calculate reconstruction loss
    faultyreconloss = mean((X - Xrecon).^2, 2);
end