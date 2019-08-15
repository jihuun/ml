{% include "../math.md" %}  

  
  
# ex2-2: Regularized Logistic Regression  
---  
  
<!-- toc -->  

  
Training data set이 Non-linear한 decision boundary를 가지고 있을때.  
  
## 1. mapFeature.m  
`중요` 기존 데이터 X를 가지고 polynomial feature를 생성한다. 이후에 X는 열은 28개가 된다. $$m \times 28$$   
training set을 더 잘 표현할수 있게 되었지만 Overfitting 위험성이 생겼다. 따라서 Regularized해야한다.   
  
```matlab  
  
function out = mapFeature(X1, X2)  
% MAPFEATURE Feature mapping function to polynomial features  
%  
%   MAPFEATURE(X1, X2) maps the two input features  
%   to quadratic features used in the regularization exercise.  
%  
%   Returns a new feature array with more features, comprising of   
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..  
%  
%   Inputs X1, X2 must be the same size  
%  
  
degree = 6;  
out = ones(size(X1(:,1)));  
for i = 1:degree  
    for j = 0:i  
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);  
    end  
end  
  
end  
```  
  
## 1. costFunctionReg.m  
  
gradient descent를 할때 for 문을 사용해 반복 계산할 필요가 없다. 나중에 fminunc를 사용할때 알아서한다. 여기서는 한번의공식만 구하면된다.   
~~Regularziation logistic regression 의 gradient descent 공식이 내가 알던거랑 다른것 같음, 알아보기.~~확인 및 수정완료.  
  
```matlab  
  
function [J, grad] = costFunctionReg(theta, X, y, lambda)  
m = length(y); % number of training examples  
  
J = 0;  
grad = zeros(size(theta));  
  
% ====================== YOUR CODE HERE ======================  
  
hx = X * theta;  
gx = sigmoid(hx);  
  
% Cost function J  
pos = -y' * log(gx);  
neg = (1 - y)' * log(1 - gx);  
regular = lambda / (2 * m) * sum(theta(2:end) .^ 2);  
  
J = (1 / m) * (pos - neg) + regular;  
  
% Gradient descent  
  
grad = (1/m) * X' * (gx - y) + (lambda/m) .* theta;  
grad(1) = (1/m) * X(:,1)' * (gx - y);  
  
% =============================================================  
  
end  
```  
  
