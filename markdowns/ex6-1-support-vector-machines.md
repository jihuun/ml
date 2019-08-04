{% include "../math.md" %}  

  
## ex6-1. support vector machines  
---  
  
### 1. Try different value of C  
---  
  
```matlab  
% You should try to change the C value below and see how the decision  
% boundary varies (e.g., try C = 1000)  
C = 1;  
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);  
visualizeBoundaryLinear(X, y, model);  
```  
> model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);    
> SVM 라이브러리 사용으로 코드 구현할 필요없음.    
  
### 2. Gaussian Kernel  
---  
  
```matlab  
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;  
sim = gaussianKernel(x1, x2, sigma);  
```  
> caller  
  
```matlab  
function sim = gaussianKernel(x1, x2, sigma)  
% ====================== YOUR CODE HERE ======================  
sim = exp(-( (sum((x1 - x2) .^ 2)) / (2*(sigma^2)) ));  
```  
> 공식에 맞게 계산만함.  
  
### 3. Gaussian Kernel for non-linear  
---  
  
gaussianKernel()을 구현해서 다음 library 함수를 사용하면 decision boundary가 나옴.  
  
```matlab  
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));   
visualizeBoundary(X, y, model);  
```  
  
### 4. Find best C and Sigma  
---  
  
```matlab  
  
% Try different SVM Parameters here  
[C, sigma] = dataset3Params(X, y, Xval, yval);  
  
% Train the SVM  
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));  
visualizeBoundary(X, y, model);  
```  
> caller  
  
  
```matlab  
function [C, sigma] = dataset3Params(X, y, Xval, yval)  
  
% ====================== YOUR CODE HERE ======================  
% Instructions: Fill in this function to return the optimal C and sigma  
%               learning parameters found using the cross validation set.  
%               You can use svmPredict to predict the labels on the cross  
%               validation set. For example,   
%                   predictions = svmPredict(model, Xval);  
%               will return the predictions on the cross validation set.  
%  
%  Note: You can compute the prediction error using   
%        mean(double(predictions ~= yval))  
%  
  
x1 = [1 2 1]; x2 = [0 4 -1];   
max_min = 99999999;  
  
for tmp_c =[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]  
	for tmp_sigma =	[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]  
		model= svmTrain(X, y, tmp_c, @(x1, x2) gaussianKernel(x1, x2, tmp_sigma));   
		predictions = svmPredict(model, Xval);  
		minimun = mean(double(predictions ~= yval));  
		%fprintf('C = %f, sigma = %f minimun=%f\n', tmp_c, tmp_sigma, minimun);  
		if (minimun < max_min)  
			max_min = minimun;  
			res_c = tmp_c;  
			res_sigma = tmp_sigma;  
		end  
	end  
end  
  
C = res_c;  
sigma = res_sigma;  
  
```  
  
