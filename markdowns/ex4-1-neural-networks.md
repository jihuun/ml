{% include "../math.md" %}  

  
# ex4-1 : neural networks  
---  
  
<!-- toc -->  

  
## 1. feedforward and cost function (without regularization)  
> nnCostFunction.m  
  
```matlab  
  
function [J grad] = nnCostFunction(nn_params, ...  
                                   input_layer_size, ...  
                                   hidden_layer_size, ...  
                                   num_labels, ...  
                                   X, y, lambda)  
  
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...  
                 hidden_layer_size, (input_layer_size + 1));  
  
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...  
                 num_labels, (hidden_layer_size + 1));  
  
% ====================== YOUR CODE HERE ======================  
% Instructions: You should complete the code by working through the  
%               following parts.  
%  
% Part 1: Feedforward the neural network and return the cost in the  
%         variable J. After implementing Part 1, you can verify that your  
%         cost function computation is correct by verifying the cost  
%         computed in ex4.m  
%  
  
% Calculate h(x) through the forward propagation   
X = [ones(m, 1) X];		% X: 5000x(400+1)  
a2 = sigmoid(X * Theta1');	% theta1: 25x401  
  
a2 = [ones(m, 1) a2];		% a2: 5000x(25+1)  
a3 = sigmoid(a2 * Theta2');	% theta2: 10x26  
				% a3: 5000x10  
				% h(x) == a3   
				% a3의 하나의 row에서 1열 ~ 10(K)열은 각각이 class를 나타냄. 1 ~ 10까지  
				% 따라서 Y(5000x10) 의 각 열은 1부터 10으로 채워져야함 -> 아님.  
% Generate Y: 오답  
%Y = zeros(m, num_labels);  
%for i=1:m  
%	for j=1:num_labels  
%		Y(:,j) = j;  
%	end  
%end  
  
% Generate Y: 정답  
% 하나의 row는 하나의 y 벡터여야함. 결과값 y(10x1) 벡터는 하나만 1이고 나머지는 0임.  
% Y는 y'한것을 m(row갯수) 만큼 이어붙임 '  
I = eye(num_labels);  
Y = zeros(m, num_labels);  
for i=1:m  
	Y(i, :)= I(y(i), :);  
end  
  
% Cost function J1  
  
pos = -Y .* log(a3);		% Y: 5000x10, a3: 5000x10  
				% Y .* a3: 5000x10  
neg = (1 - Y) .* log(1 - a3);  
  
J = (1/m) * sum(sum(pos - neg,2),1);  
```  
  
## 2. regularized cost function  
  
```matlab  
  
% Cost function J + regularization  
% theta1: 25x401  
% theta2: 10x26  
  
Theta1Reg = Theta1(:,2:size(Theta1,2));  
Theta2Reg = Theta2(:,2:size(Theta2,2));  
  
Reg = sum(sum((Theta1Reg).^2, 2)) + sum(sum((Theta2Reg).^2, 2));  
J = J + (lambda/(2*m))*Reg;  
```  
  
