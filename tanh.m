function g = tanh(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

euler = e(size(z));
allones = ones(size(z));
g = ((euler.^(z)).-(euler.^(z.*-1)))./((euler.^(z.*-1)).+(euler.^(z)));
