X = [1.0000    0.1000    0.6000    1.1000;
    1.0000    0.2000    0.7000    1.2000;
    1.0000    0.3000    0.8000    1.3000;
    1.0000    0.4000    0.9000    1.4000;
    1.0000    0.5000    1.0000    1.5000];
y = [1;0;1;0;1];
m = 5;
theta = [-2;-1;1;2];
lambda = 3;
J = (sum((-y.*log(sigmoid(X*theta)))-((1-y).*log(1-sigmoid(X*theta))))/m) + ((lambda/(2*m))*sum(theta(2:length(theta)).^2));