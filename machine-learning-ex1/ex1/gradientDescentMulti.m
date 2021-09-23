function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y);
n = length(theta); % number of training examples
J_history = zeros(num_iters, 1);
numtheta = zeros(n, 1);
for i = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
h = (X* theta - y)';
for j= 1:n
  numtheta(j)=theta(j)-(alpha/m)*h*X(:,j);
 end

for j = 1:n
  theta(j)= numtheta(j);
 end
J_history(i)=computeCostMulti(X, y, theta);
end

end










    % ============================================================

    

