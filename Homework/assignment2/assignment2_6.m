% set the random number seed to 0 for r e p o r d u c i b i l i t y
rand ( 'seed' ,0);
avg = [1 2 3 4 5 6 7 8 9 10];
scale = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001];
% g e n e r a t e 5000 examples , each 10 dim
corr1 = zeros(9,1);
corr2 = zeros(9,1);
new_e1 = zeros(10, 9);
e1 = zeros(10, 9);
randData = randn (5000 ,10);
for i = 1:9
    data = randData + repmat ( avg * scale(i) ,5000 ,1);
    m = mean ( data ); % average
    m1 = m / norm ( m ); % n o r m a l i z e d avearge
    % do PCA , but without c e n t e r i n g
    [~ , S , V ] = svd ( data );
    S = diag ( S );
    e1(:,i) = V (: ,1); % first eigenvector , not minus mean vector
    % do correct PCA with c e n t e r i n g
    newdata = data - repmat (m ,5000 ,1);
    [U , S , V ] = svd ( newdata );
    S = diag ( S );
    new_e1(:,i) = V (: ,1); % first eigenvector , minus mean vector
    % c o r r e l a t i o n between first e i g e n v e c t o r ( new & old ) and mean
    avg = avg - mean ( avg );
    avg = avg / norm ( avg );
    e1(:,i) = e1(:,i) - mean ( e1(:,i) );
    e1(:,i) = e1(:,i) / norm ( e1(:,i) );
    new_e1(:,i) = new_e1(:,i) - mean ( new_e1(:,i) );
    new_e1(:,i) = new_e1(:,i) / norm ( new_e1(:,i));
    corr1(i) = avg * e1(:,i);
    corr2(i) = e1(:,i)' * new_e1(:,i);
end