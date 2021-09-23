% 400 pixels
input_layer_size  = 400;
% numbers 1-10
num_labels = 10; 

load('ex3data1.mat');
m = size(X, 1);

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;

%[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

m = length(y);
J = 0;
grad = zeros(size(theta));
h_theta = sigmoid(X*theta);
J = (-1/m)*(y'*log(h_theta) + (1- y')*log(1-h_theta)) + lambda/(2*m)*theta(2:end)'*theta(2:end);

grad = (X'*(h_theta- y) +lambda*theta)/m;
grad(1,:) -= lambda*theta(1,:)/m;

grad = grad(:);

lambda = 0.1;
%[all_theta] = oneVsAll(X, y, num_labels, lambda);

m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n + 1);

X = [ones(m, 1) X];

initial_theta = zeros(n+1,1);

options = optimset('GradObj', 'on', 'MaxIter', 50);
  for i= 1:num_labels;
  all_theta(i,:) = fmincg(@(t)(lrCostFunction(t, X, (y==i), lambda)), initial_theta, options);
endfor

num_labels = size(all_theta, 1);
p = zeros(size(X, 1), 1);

value = sigmoid(X*all_theta');

for (i= 1:m)
  [dummy, p(i)] = max(value(i,:));
end
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

