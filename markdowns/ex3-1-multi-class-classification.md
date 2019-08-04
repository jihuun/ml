{% include "../math.md" %}  

  
  
## ex3-1 Multi-class classification  
---  
  
### lrCostFunction.m  
  
```matlab  
function [J, grad] = lrCostFunction(theta, X, y, lambda)  
...(중략)  
  
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
```  
  
### oneVsAll.m  
  
```matlab  
  
function [all_theta] = oneVsAll(X, y, num_labels, lambda)  
...(중략)  
  
for c = 1:num_labels  
	initial_theta = zeros(n + 1, 1);  
  
	% Set options for fminunc  
	options = optimset('GradObj', 'on', 'MaxIter', 50);  
  
	% Run fmincg to obtain the optimal theta  
	% This function will return theta and the cost   
	[theta] = ...  
		fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...  
			initial_theta, options);  
  
	all_theta(c,:) = theta';  
end  
```  
> ' 위에서 만든 lrCostFunction 을 가지고 num_labels 갯수만큼 반복한다. fmincg는 최적의 parameter $$\Theta$$를 찾아줄것이고 1x401 크기의 결과($$\Theta$$)를 줄 것이다. feature가 401개 였기 때문에 그에 상응하는 parameter(weight)의 갯수도 401개. 결과는 단지 all_theta에 반복 횟수만큼 row가 추가되는것. 이 결과 all_theta를 가지고 predict 할것이다.    
  
### predictOneVsAll.m  
  
  
```matlab  
function p = predictOneVsAll(all_theta, X)  
...(중략)  
  
gx = sigmoid(X * all_theta');  
[max_pobability_one_example, class_in_one_example] = max(gx, [], 2);  
p = class_in_one_example;  
```  
X : 5000 X 401  
all_theta : 10 X 401  
gx : 5000 X 10  
matrix gx의 각 row의 의미: 한 training set의 가설함수 $$h_\theta(x)$$ 의 결과가 0~9 순서로 10개 저장되어있다. 해당 training set이  0~9 숫자일확률이 각각 들어있다.   
max(gx, [], 2); 를 하면 첫번째 결과물은 해당 row의 max 값이고 두번째는 그 max값의 index를 리턴한다. 그게 최종 우리의 예측 값.   
  
