{% include "../math.md" %}  

  
## ex5  
---  
  
### Part 2: Regularized Linear Regression Cost   
  
```matlab  
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)  
  
H = X * theta; 		% X(12x2), theta(2x1)  
Reg = (lambda / (2*m)) * sum(theta(2:end, 1) .^ 2);	% theta(2:end, 1) -> theta(1) 은 더하면 안됨.  
  
J = (1/(2*m)) * sum((H - y).^2) + Reg;  
  
end  
```  
  
### Part 3: Regularized Linear Regression Gradient  
  
Gradient 계산 하기   
  
```matlab  
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)  
  
H = X * theta; 				% X(12x2), theta(2x1)  
theta_0 = [0; theta(2:end, :)];		% H계산 이후에 theta는 2부터 필요함.  
  
Reg = (lambda / (2*m)) * (theta_0' * theta_0); 		% theta(2:end, 1) -> theta(1) 은 더하면 안됨.  
                                        		% sum(theta_0 .^ 2) 을 theta_0' * theta_0 으로 표현할 수 있음!  
J = (1/(2*m)) * sum((H - y).^2) + Reg;  
  
grad    = (1/m) * (X' * (H - y)) + ((lambda / m) * theta_0);  
  
end  
```  
  
  
### Part 4: Train Linear Regression   
  
Caller   
  
```matlab  
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);  
```  
  
```matlab  
function [theta] = trainLinearReg(X, y, lambda)   
  
   
% Initialize Theta   
initial_theta = zeros(size(X, 2), 1);    
   
% Create "short hand" for the cost function to be minimized   
costFunction = @(t) linearRegCostFunction(X, y, t, lambda);   
   
% Now, costFunction is a function that takes in only one argument   
options = optimset('MaxIter', 200, 'GradObj', 'on');   
  
% Minimize using fmincg  
theta = fmincg(costFunction, initial_theta, options);  
```  
  
  
### Part 5: Learning Curve for Linear Regression  
  
Learning algorithm을 디버깅(검증) 하기 위해서 Learning Curve 를 그린다.  Learning Curve 는 training set 갯수에 따라서 training example과 cross validation set의 cost를 각각 구해서 그려봄 으로써, high bias vs variaence 인지 확인하는 방법론이다.   
기존의 training example에서 크기m을 1부터 늘려가며 $$\theta$$를 학습하여 모델을 만든다 -> 그뒤 training example과  C.V 에서 각각 cost를 구해본다.   
  
```matlab  
function [error_train, error_val] = ...  
    learningCurve(X, y, Xval, yval, lambda)  
  
for i = 1:m   
        % 먼저 1 ~ i 까지 Training set 에서 theta를 학습  
        theta = trainLinearReg(X(1:i, :), y(1:i), lambda);  
  
        % 구해진 theta로 1 ~ i 까지 Training set 에서 cost 계산  
        error_train(i) = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);   
  
        % 위의 Training example에서 학습한 theta로 CV에서 cost 계산  
        error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);   
end  
```  
  
- Caller  
  
```matlab  
[error_train, error_val] = ...  
    learningCurve([ones(m, 1) X], y, ...  
                  [ones(size(Xval, 1), 1) Xval], yval, ...  
                  lambda);  
```  
  
- 그 뒤 그래프를 그려본다.   
  
```  
plot(1:m, error_train, 1:m, error_val);  
  
```  
  
### Part 6: Feature Mapping for Polynomial Regression  
  
결과 matrix X_poly는 (m x p) 크기다. 각 열은 x, x^2, x^3, .. x^p 가 계산된 값이 들어있어야 한다.    
  
```matlab  
function [X_poly] = polyFeatures(X, p)  
  
X_poly = zeros(numel(X), p);  
  
for j = 1:p  
        X_poly(:,j) = X .^ j;  
end  
```  
  
### Part 7: Learning Curve for Polynomial Regression   
  
### Part 8: Validation for Selecting Lambda   
  
Caller  
  
```matlab  
[lambda_vec, error_train, error_val] = ...  
    validationCurve(X_poly, y, X_poly_val, yval);  
```  
  
```matlab  
function [lambda_vec, error_train, error_val] = ...  
    validationCurve(X, y, Xval, yval)  
  
for i = 1:length(lambda_vec)  
    lambda = lambda_vec(i);  
    theta = trainLinearReg(X,y,lambda);  
    error_train(i) = linearRegCostFunction(X   , y   , theta, 0);  
    error_val(i)   = linearRegCostFunction(Xval, yval, theta, 0);  
end  
```  
  
