{% include "../math.md" %}  

  
## ex3-2 Neural Networks  
---  
  
### predict.m  
  
```matlab  
  
% Add ones to the X data matrix  
X = [ones(m, 1) X];  
a2 = sigmoid(X * Theta1');  
  
% Add ones to the a2 data matrix  
a2 = [ones(m, 1) a2];  
a3 = sigmoid(a2 * Theta2');  
  
[max_pobability_one_example, class_in_one_example] = max(a3, [], 2);  
p = class_in_one_example;  
```  
Layer가 3개인 NN은 Logistic regression을 단계별로 두번 하는것과 동일하다.     
그렇게 해서 구해진 a3 벡터는  $$h_\Theta(x)$$ 벡터이다.     
X : 5000 x (400+1) (1 열 추가함)    
Theta1 : 25 x 401    
a2 : 5000 x (25+1) (1 열 추가함)    
Theta2 : 10 x 26    
a3 : 5000 x 10     
즉 predictOneVsAll.m 경우처럼 한 row의 크기가 10이고 각각에는 0~9 에 해당하는 확률값이 들어있다.     
  
  
