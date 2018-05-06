N = 5000;
K = 2;
% numTree = 1;

x = rand(N, 10);
disp('brute-force');
tic
D1 = zeros(N, N);
for i = 1:N
    for j=1:N
        D1(i,j)=0;
        for k = 1:10
            D1(i,j) = D1(i,j) + (x(i,k)-x(j, k))^2;
        end
    end
end
[B1, I1] = sort(D1);
% disp(I1(2, :));
sec1 = toc;
% disp('matrix');
% tic
% square = x .* x;
% sumS = sum(square,2);
% D2 = repmat((sumS), [1, N]) - 2 * x * x' + repmat((sumS)', [N, 1]);
% D2 = D2 - diag(diag(D2));
% 
%     
% [B2, I2] = sort(D2);
% % disp(I2(2,:))
% toc

disp('KD-Tree');
newx = x';
kdtree = vl_kdtreebuild(newx, 'NumTrees', numTree);
I3 = zeros(K, N);
B3 = zeros(K, N);
tic
for i = 1:N
    [I3(:,i), B3(:,i)] = vl_kdtreequery(kdtree, newx, newx(:,i), 'NumNeighbors', K, 'MaxNumComparisons', 6000);
end
sec2 = toc;
error_rate = sum(I1(2,:) ~= I3(2,:)) / N;
arr1 = zeros(2, 10);
arr2 = zeros(2, 10);
iter = 0;
for numComparison = 1:10:101
    iter = iter + 1;
    kdtree = vl_kdtreebuild(newx, 'NumTrees', 1);
    tic
    I3 = zeros(K, N);
    B3 = zeros(K, N);
    for i = 1:N
        [I3(:,i), B3(:,i)] = vl_kdtreequery(kdtree, newx, newx(:,i), 'NumNeighbors', K, 'MaxNumComparisons', numComparison);
    end
    arr1(1, iter) = toc;
    arr1(2, iter) = sum(I1(2,:) ~= I3(2,:)) / N;
end

for numTree = 1:10
    numTree
    kdtree = vl_kdtreebuild(newx, 'NumTrees', numTree);
    tic
    I3 = zeros(K, N);
    B3 = zeros(K, N);
    for i = 1:N
        [I3(:,i), B3(:,i)] = vl_kdtreequery(kdtree, newx, newx(:,i), 'NumNeighbors', K, 'MaxNumComparisons', 100);
    end
    arr2(1, numTree) =  toc;
    arr2(2, numTree) = sum(I1(2,:) ~= I3(2,:)) / N;
end

subplot(2,2,1);
plot([1:10:101], arr1(1,:));
xlabel('MaxNumComparisons');
ylabel('runnung time');
set(gca,'xtick',1:10:101)

subplot(2,2,2);
plot([1:10:101], arr1(2,:));
xlabel('MaxNumComparisons');
ylabel('error rate');
set(gca,'xtick',1:10:101)

subplot(2,2,3);
plot([1:10], arr2(1,:));
xlabel('NumTrees');
ylabel('runnung time');
set(gca,'xtick',1:10)

subplot(2,2,4);
plot([1:10], arr2(2,:));
xlabel('NumTrees');
ylabel('error rate');
set(gca,'xtick',1:10)
