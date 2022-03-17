%(a)
x = [-1 -1/2 0 1/2 1];
y = [-1 -1/8 0 1/8 1];

n = length(y);

mux = (sum(x,2)) * 1/n;
muy = (sum(y,2)) * 1/n;

X = x - mux * ones(n,1)';
Y = y - muy * ones(n,1)';

Sx = 1/n * X * X';
Sxy = 1/n * X * Y';

wOLS = inv(Sx)*Sxy
bOLS = muy - wOLS'*mux

%(b)
