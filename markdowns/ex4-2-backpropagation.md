{% include "../math.md" %}  

  
# ex4-2 Backpropagation  
---  
  
<!-- toc -->  

  
## 1. sigmoid gradient  
  
```matlab  
gz = sigmoid(z);  
g = gz .* (1 - gz);  
```  
  
## 2. Random initialization  
  
```matlab  
epsilon_init = 0.12;  
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;  
```  
  
## 3. Backpropagation  
  
```matlab  
  
% -------------------------------------------------------------  
% Part 2: Implement the backpropagation algorithm to compute the gradients  
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of  
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and  
%         Theta2_grad, respectively. After implementing Part 2, you can check  
%         that your implementation is correct by running checkNNGradients  
%  
%         Note: The vector y passed into the function is a vector of labels  
%               containing values from 1..K. You need to map this vector into a   
%               binary vector of 1's and 0's to be used with the neural network  
%               cost function.  
%  
%         Hint: We recommend implementing backpropagation using a for-loop  
%               over the training examples if you are implementing it for the   
%               first time.  
  
for t=1:m  
	% 1. feedforward   
	a_1 = X(t,:)';		% t-th training set x^t  
				% a_1: (401)x1  
	z_2 = (Theta1 * a_1);	% theta1: 25x401  
	a_2 = sigmoid(z_2);	% z_2: 25x1   
  
	a_2 = [1 ; a_2];	% a_2: (25+1)x1  
	z_3 = (Theta2 * a_2);	% theta2: 10x26   
	a_3 = sigmoid(z_3);	% a_3: 10x1  
  
	% 2. delta of output unit  
	d_3 = a_3 - Y(t,:)';	% d_3: 10x1  
	% 3. delta of hidden layer l=2  
	z_2 = [1 ; z_2];				% z_2: (25+1)x1  
	d_2 = (Theta2' * d_3) .* sigmoidGradient(z_2);	% d_2: 26x1  
	% 4. accumulate gradient   
	d_2 = d_2(2:end);				% d_2: (26-1)x1  
	Theta2_grad = Theta2_grad + d_3 * a_2';		% 10x1 * 1x26  
	Theta1_grad = Theta1_grad + d_2 * a_1';		% 25x1 * 1x401  
end  
  
% 5. obtain final gredient  
Theta2_grad = Theta2_grad ./ m;  
Theta1_grad = Theta1_grad ./ m;  
  
```  
  
## 4. Regularizaed Neural networks  
  
```matlab  
  
% -------------------------------------------------------------  
% Part 3: Implement regularization with the cost function and gradients.  
%  
%         Hint: You can implement this around the code for  
%               backpropagation. That is, you can compute the gradients for  
%               the regularization separately and then add them to Theta1_grad  
%               and Theta2_grad from Part 2.  
%  
  
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);  
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);  
  
Theta2_grad(:,1) = Theta2_grad(:,1);  
Theta1_grad(:,1) = Theta1_grad(:,1);  
  
```  
  
# ex4.m  
---  
  
위에서 구현한 costFunction 을 가지고 fmincg 돌리면 $$\Theta$$ 구해짐.   
  
```matlab  
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);  
```  
  
