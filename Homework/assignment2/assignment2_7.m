x = randn(2000,2)*[2,1;1,2];
[~ , S , V ] = svd ( x );
m = mean(x);
x = x - repmat(m, 2000, 1);
x = x * V;
x = x * diag(diag(S.^(-1)));
scatter(x(:,1), x(:,2));