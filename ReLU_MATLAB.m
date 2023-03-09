function y = ReLU(x)

T = length(x);
y = zeros(T,1);

f = find(x>0);
y(f) = x(f);