load('prostateStnd.mat')
%istrain, X, Y, xtrain, ytrain, xtest, ytest, names
%(a)
Xtrain_mean = mean(Xtrain,1);
Xtrain_var = var(Xtrain,1);
%normalize the training dataset
%xnormalized = (x - xminimum)/range of x
Xmin = min(Xtrain,[],1);
Xrange = range(Xtrain,1);

Xtrain_normalized = (Xtrain - min(Xtrain,[],1))./range(Xtrain,1)

ytrain_mean = mean(ytrain,1);
ytrain_var = var(ytrain,1);
%normalize the training dataset
%xnormalized = (x - xminimum)/range of x
ymin = min(ytrain,[],1);
yrange = range(ytrain,1);

ytrain_normalized = (ytrain - min(ytrain,[],1))./range(ytrain,1)

for i = 1:8
    fprintf('Feature %d''s mean is %f and variane is %d\n', i, Xtrain_mean(i), Xtrain_var(i));
end
fprintf('Label''s mean is %f and variane is %d\n', ytrain_mean, ytrain_var);

%(b)
lambda = zeros(length(-5:1:10),length(-5:1:10));
lambdas = zeros(1,length(-5:1:10));
for i = -5:1:10
    lambda(i+6,i+6) = exp(i);
    lambdas(i+6) = exp(i);
end

[n_train d] = size(Xtrain_normalized);
mux_train = (1/n_train).*sum(Xtrain_normalized,1);
muy_train = (1/n_train).*sum(ytrain_normalized,1);
sum_train = zeros(d,d);
for i = 1:n_train
    sum_train = sum_train + (Xtrain_normalized(i,:)'-mux_train') * (Xtrain_normalized(i,:)'-mux_train')';
end
Sx = (1/n_train) .* sum_train;

sum_train = zeros(d,1);
for i = 1:n_train
    sum_train = sum_train + (Xtrain_normalized(i,:)'-mux_train') * (ytrain_normalized(i,:)'-muy_train');
end
Sxy = (1/n_train) .* sum_train;

Id = eye(d);
wridge = zeros(d,length(lambdas));
bridge = zeros(1,length(lambdas));
for i = 1:length(lambdas)
    wridge(:,i) = (inv(((lambdas(i)/n_train)*Id) + Sx)) * Sxy;
    bridge(1,i) = muy_train - wridge(:,i)' * mux_train';
end

for i = 1:d
    hold on
    plot(log(lambdas), wridge(i,:))
end