function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%possible_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%J = zeros(length(possible_values)^2, 4);
%
%for i = 1:length(possible_values)
%    C = possible_values(i);
%    for j = 1:length(possible_values)
%        combination_number = (i - 1) * length(possible_values) + j;
%        sigma = possible_values(j);
%        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%        predictions = svmPredict(model, Xval);
%        J(combination_number, :) = [combination_number, C, sigma, mean(double(predictions ~= yval))];
%    end
%end
%
%J
%
%[min_cost, ind] = min(J(:, 4))
%min_row = J(ind, :)
%C = min_row(2)
%sigma = min_row(3)

% Using the above code, the minimum values were found to be:
C = 1;
sigma = 0.1;

% =========================================================================

end
