{% include "../math.md" %}  

  
  
# ex8: Recommander System  
---  
  
<!-- toc -->  

  
- ex8_cofi.m  
  
```matlab  
function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...  
                                  num_features, lambda)  
  
% 1. Cost fuction J()  
% from test: num_users = 4; num_movies = 5; num_features = 3;  
% X: 5x3, Theta:4x3, R:Y: 5x4  
M = (X * Theta') .* R;  % 5x4  
T = (M - Y) .^ 2; % 5x4  
J = (1/2) * sum(sum(T)); % scalar  
  
% 2. Gradient   
X_grad = (M - Y) * Theta; % (5x4) * (4x3) = (5x3)  
Theta_grad = (M - Y)' * X; % (5x4)' * (5x3) = (4x3)  
  
% 3. Cost fuction J() with Regularization  
Reg_J = (lambda/2) * sum(sum(Theta .^ 2)) + (lambda/2) * sum(sum(X .^ 2));  
J = J + Reg_J;  
  
% 4. Gradient with Regularization  
Reg_x_grad = lambda * X;  
Reg_Theta_grad = lambda * Theta;  
  
X_grad = X_grad + Reg_x_grad;  
Theta_grad = Theta_grad + Reg_Theta_grad;  
```  
  
