%this code compares the Euler method vs. 
%analytical solution for:
%(1) the supervised rule alone
%(2) the predictive rule alone
%(3) the combined supervised+predictive rule

clear all
clc;

%SUPERVISED RULE ALONE
%%%%%%%%%%%%%%%%%%%%%%%%%%
num_iter = 100;
g = 0.5;
tau = 0.1;
w0 = 0;
y = 1;

%Euler method
save_w_sup = zeros(num_iter,1);
w = w0;
for i = 1:num_iter
    dw = g * (y-w);
    w = w + tau*dw;
    save_w_sup(i) = w;
end

%analytical solution
w_est_sup = zeros(num_iter,1);
for i = 1:num_iter
    w_est_sup(i) = exp(-(g*i*tau)) * (y * (exp(g*i*tau) - 1 + w0));
end

%PREDICTIVE RULE ALONE
%%%%%%%%%%%%%%%%%%%%%%%%%%
s = 0.11;
bhat = 0.8;
w = 0.2;
tau = 0.01;

c1 = bhat;
c2 = w;

%Euler method
save_w_pred = zeros(num_iter,1);
for i = 1:num_iter
    dw = -s * (w-bhat);
    dbhat = -bhat+w;
    w = w + tau*dw;
    bhat = bhat + tau*dbhat;
    save_w_pred(i) = w;
end

%analytical solution
w_est_pred = zeros(num_iter,1);
for i = 1:num_iter
    w_est_pred(i) = (c2*(s*exp(-(s+1)*i*tau)+1))./(s+1) - (c1*s*(exp(-(s+1)*i*tau)-1))./(s+1);
end

%COMBINED SUPERVISED+PREDICTIVE RULE ALONE
%%%%%%%%%%%%%%%%%%%%%%%%%%
y = 1;
s = 0.1;
g = 0.1;
b = 0;
bhat = 0;
w = 0;
tau = 0.9;

c1 = bhat;
c2 = w;

%Euler method
save_w_combined = zeros(num_iter,1);
for i = 1:num_iter
    dw = g*(y-w) - s * (w-bhat);
    dbhat = -bhat+w;
    w = w + tau*dw;
    bhat = bhat + tau*dbhat;    
    save_w_combined(i) = w;
end

%analytical solution
w_est_combined = zeros(num_iter,1);
for i = 1:num_iter
    w_est_sup1 = exp(-(g*i)) * (exp(g*i) - 1 + c2); 
    w_est_pred1 = (c2*(s*exp(-(s+1)*i*tau)+1))./(s+1) - (c1*s*(exp(-(s+1)*i*tau)-1))./(s+1);
    w_est_combined(i) = w_est_sup1 + w_est_pred1;
end

%plot
figure;
subplot(1,3,1)
hold on;
plot(save_w_sup);
plot(w_est_sup,'k.','markersize',5);
title('supervised')
xlabel('time-steps');
ylabel('w');

subplot(1,3,2)
hold on;
plot(save_w_pred);
plot(w_est_pred,'k.','markersize',5);
title('predictive')
xlabel('time-steps');
ylabel('w');

subplot(1,3,3)
hold on;
plot(save_w_combined);
plot(w_est_combined,'k.','markersize',5);
title('combined')
xlabel('time-steps');
ylabel('w');
return;
