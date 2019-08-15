{% include "../math.md" %}  


  
# ex7: K-means clustering and PCA  
---  
  
<!-- toc -->  

  
## Part 1: Find Closest Centroids   
---  
각 X(i) 를 순회하면서   
주어진 초기 centroids 에 가까운 index k 를 idx(i) 에 저장  
  
Caller    
  
```matlab  
idx = findClosestCentroids(X, initial_centroids);  
```  
  
```matlab  
function idx = findClosestCentroids(X, centroids)  
% Set K  
K = size(centroids, 1);  
  
% You need to return the following variables correctly.  
idx = zeros(size(X,1), 1);  
  
for i = 1:size(X,1)  
  
        max_dis = 9999999;  
        for k = 1:K  
		dis = sum((centroids(k,:) - X(i,:)).^2);  
                if (dis < max_dis)  
                        idx(i,1) = k;  
                        max_dis = dis;  
                end  
        end  
end  
  
```  
  
  
## Part 2: Compute Means   
---  
  
각 data의 평균 좌표 구하기  
  
```matlab  
%  Compute means based on the closest centroids found in the previous part.  
centroids = computeCentroids(X, idx, K);  
```  
  
```matlab  
function centroids = computeCentroids(X, idx, K)  
  
% Useful variables  
[m n] = size(X);  
  
% You need to return the following variables correctly.  
centroids = zeros(K, n);  
  
cnt = zeros(3,1);  
  
for i=1:m  
        for j=1:n  
                centroids(idx(i), j) = centroids(idx(i), j) + X(i,j);  
        end  
        cnt(idx(i),1)++;  
end  
  
centroids = centroids ./ cnt;  
```  
  
  
  
  
