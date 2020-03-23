function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

ref_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
max_error = Inf;

for ref_sig = ref_val
   for  ref_c = ref_val
    model = svmTrain(X,y,ref_c,@(x1, x2) gaussianKernel(x1, x2, ref_sig)); % Training Model
    predictions = svmPredict(model, Xval); % Predicting our answer
    error = mean(double(predictions ~= yval)); % Getting our error for the prediction
    if error < max_error % if we get a smaller error then udpate C & Sigma
        max_error = error;
        C = ref_c;
        sigma = ref_sig;
    end
   end
end

% =========================================================================

end
