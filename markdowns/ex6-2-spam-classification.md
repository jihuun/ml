{% include "../math.md" %}  

  
  
# ex6-2 Spam classification  
---  
  
<!-- toc -->  

  
  
## Part 1: Email Preprocessing   
---  
  
1. normialize : 각 단어들을 집계하기 쉽게 변경, 불필요 charactor제거  
2. 각 단어 mapping: email을 vocabList 에 있는 index로 변경하기 -> email이 번호가 나열된 벡터가됨.    
  
결과적으로 이메일이 mapping된 숫자가 들어있는 벡터 word_indices로 변경됨.  
  
caller  
  
```matlab  
% Extract Features  
file_contents = readFile('emailSample1.txt');  
word_indices  = processEmail(file_contents);  
```  
  
구현내용: 2(mapping)  
  
```matlab  
function word_indices = processEmail(email_contents)  
...(중략)  
while ~isempty(email_contents)  
	...(중략)  
	for i=1:size(vocabList,1)  
		if strcmp(vocabList(i), str)  
			word_indices = [word_indices ; i];  
		end  
	end  
end  
```  
  
## Part 2: Feature Extraction  
---  
  
mapping된 email에서 feature vector x를 추출해야한다. 결과적으로 하나의 email에서 다음과 같은 형태의 x벡터가 추출될 것이다. (vocabList 갯수가 1899개)    
  
$$  
x =   
\begin{bmatrix}  
0 \\  
\vdots \\  
1 \\  
0 \\  
\vdots \\  
0 \\  
1 \\  
\end{bmatrix}  
\in \mathbb{R}^{1899}  
$$  
  
Caller    
  
```matlab  
% Extract Features  
file_contents = readFile('emailSample1.txt');  
word_indices  = processEmail(file_contents);  
features      = emailFeatures(word_indices);  
```  
  
```matlab  
function x = emailFeatures(word_indices)  
...(중략)  
for i=1:size(word_indices,1)  
	x(word_indices) = 1;  
end  
```  
  
  
## Part 3: Train Linear SVM for Spam Classification   
---  
  
spamTrain.mat은 우리가 학습할 trainig examples 이다.   
spamTrain.mat 에는 위처럼 email을 feature vector x로 mapping한 1899크기의 x벡터가 4000개 포함되어있다.    
  
```matlab  
% Load the Spam Email dataset  
% You will have X, y in your environment  
load('spamTrain.mat');  
```  
  
load 하면 feature 벡터 X와 그에 상응하는 결과값 y가 load 된다. y는 4000개의 각 email이 spam인지 아닌지 0 or 1로 되어있다.    
만약 유사한 프로젝트를 하려면 이런 training data를 사전에 수집해야한다.    
  
```  
        X               4000x1899                60768000  double    
        features        1899x1                      15192  double    
        y               4000x1                      32000  double    
```  
  
그 뒤 해당 데이터를 SVM으로 train시킨다. (C=0.1)  
  
```matlab  
C = 0.1;  
model = svmTrain(X, y, C, @linearKernel);  
```  
> 이렇게 생성한 model을 가지고 앞으로 모두 예측할 것이다.    
  
gaussianKernel 커널을 사용한다면 아래와 같이 변경  
```matlab  
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;  
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));   
```  
  
(참고) model에 저장되어있는 값은 무엇일까?    
who   
> model              1x1                   11801400  struct  
model.  
> model.X               model.b               model.w                 
> model.alphas          model.kernelFunction  model.y    
model은 이렇게 6개의 matrix로 이루어져 있음.    
  
  
그 뒤, predict한다.   
training sets  X에 대한 결과를 예측한 결과가 실제 y와 얼마나 가까운지 확인하는 과정이다.    
  
```matlab  
p = svmPredict(model, X);  
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);  
```  
  
svmPredict() 은 각 모델에 맞게 예측한결과 p를 return 한다.   
  
```matlab  
function pred = svmPredict(model, X)  
%SVMPREDICT returns a vector of predictions using a trained SVM model  
%(svmTrain).   
%   pred = SVMPREDICT(model, X) returns a vector of predictions using a   
%   trained SVM model (svmTrain). X is a mxn matrix where there each   
%   example is a row. model is a svm model returned from svmTrain.  
%   predictions pred is a m x 1 column of predictions of {0, 1} values.  
%  
  
% Dataset   
m = size(X, 1);  
p = zeros(m, 1);  
pred = zeros(m, 1);  
  
if strcmp(func2str(model.kernelFunction), 'linearKernel')  
    % We can use the weights and bias directly if working with the   
    % linear kernel  
    p = X * model.w + model.b;  
elseif strfind(func2str(model.kernelFunction), 'gaussianKernel')  
    % Vectorized RBF Kernel  
    % This is equivalent to computing the kernel on every pair of examples  
    X1 = sum(X.^2, 2);  
    X2 = sum(model.X.^2, 2)';  
    K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));  
    K = model.kernelFunction(1, 0) .^ K;  
    K = bsxfun(@times, model.y', K);  
    K = bsxfun(@times, model.alphas', K);  
    p = sum(K, 2);  
else  
    % Other Non-linear kernel  
    for i = 1:m  
        prediction = 0;  
        for j = 1:size(model.X, 1)  
            prediction = prediction + ...  
                model.alphas(j) * model.y(j) * ...  
                model.kernelFunction(X(i,:)', model.X(j,:)');  
        end  
        p(i) = prediction + model.b;  
    end  
end  
  
% Convert predictions into 0 / 1  
pred(p >= 0) =  1;  
pred(p <  0) =  0;  
  
end  
```  
  
  
## Part 4: Test Spam Classification   
---  
  
training example에서 학습한 모델을 가지고 test set에서도 적중하는지 맞춰본다.   
  
```matlab  
p = svmPredict(model, Xtest);  
fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);  
```  
  
  
## Part 5: Top Predictors of Spam   
---  
  
Spam으로 분류되는데 가장 많은 weight를 갖는 단어 10개를 순서대로 보여준다.   
  
```matlab  
% Sort the weights and obtin the vocabulary list  
[weight, idx] = sort(model.w, 'descend');  
vocabList = getVocabList();  
  
fprintf('\nTop predictors of spam: \n');  
for i = 1:15  
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));  
end  
```  
  
  
## Part 6: Try Your Own Emails   
---  
  
기존에 train한 model을 가지고, 신규 메일 emailSample4.txt 이 spam 인지 아닌지 예측하기!  
  
```matlab  
filename = 'emailSample4.txt';  
  
% Read and predict  
file_contents = readFile(filename);  
word_indices  = processEmail(file_contents);  
x             = emailFeatures(word_indices);  
p = svmPredict(model, x);  
  
fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);  
```  
  
  
