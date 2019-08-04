{% include "../math.md" %}  

  
# 15. Vectorized implementations (in Octave)  
---  
  
최대한 loop사용을 줄이고, array요소의 개별적 연산을 줄이는 행렬연산에 익숙해질(훈련할) 필요가 있다.    
The goal in vectorization is to write code that avoids loops and uses whole-array operations.  
  
## 15.1. Vector A가 있을때, $$sum(A.^2)$$ ?  
---  
  
$$  
A * A^T  
$$  
  
  
## 15.2. loop 제거방법  
---  
> 출처: https://octave.org/doc/v4.0.1/Basic-Vectorization.html#Basic-Vectorization  
  
Loop 사용   
```matlab  
for i = 1:n  
  for j = 1:m  
    c(i,j) = a(i,j) + b(i,j);  
  endfor  
endfor  
```  
  
Vectorization  
```matlab  
c = a + b;  
```  
  
  
Loop 사용   
```matlab  
for i = 1:n-1  
  a(i) = b(i+1) - b(i);  
endfor  
```  
  
Vectorization  
```matlab  
a = b(2:n) - b(1:n-1);  
```  
  
Loop 사용   
```matlab  
for i = 1:n  
  if (a(i) > 5)  
    a(i) -= 20  
  endif  
endfor  
```  
  
Vectorization  
```matlab  
a(a>5) -= 20;  
```  
  
## 15.3. Built-in Function: vectorize (fun)  
---  
  
Create a vectorized version of the inline function fun by replacing all occurrences of *, /, etc., with .*, ./, etc.  
  
This may be useful, for example, when using inline functions with numerical integration or optimization where a vector-valued function is expected.  
  
```matlab  
fcn = vectorize (inline ("x^2 - 1"))  
   ⇒ fcn = f(x) = x.^2 - 1  
quadv (fcn, 0, 3)  
   ⇒ 6  
```  
  
## 15.4. Vectorization built-in 함수 목록 in Octave  
  
- Index manipulation  
	- find  
	- sub2ind  
	- ind2sub  
	- sort  
	- unique  
	- lookup  
	- ifelse / merge  
- Repetition  
	- repmat  
	- repelems  
- Vectorized arithmetic  
	- sum  
	- prod  
	- cumsum  
	- cumprod  
	- sumsq  
	- diff  
	- dot  
	- cummax  
	- cummin  
- Shape of higher dimensional arrays  
	- reshape  
	- resize  
	- permute  
	- squeeze  
	- deal  
