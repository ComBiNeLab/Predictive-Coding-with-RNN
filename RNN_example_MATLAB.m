%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%example of RNN learning 
%a random target pattern
%using a combination of supervised and
%predictive learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
clc;

%RNN parameters
%%%%%%%%%%%%%%%%%%%%%%%%
N = 1; %number of inputs
alpha = 1; %leak term
M = 200; %number of recurrent units
dt=1; %RNN time-step

%training data
%%%%%%%%%%%%%%%%%%%%%%%%
T = 200; %number of time-steps
R = randn(N,T); %random sequence
X = R; %network input
Y = X; %target

%fixed synaptic weights
%%%%%%%%%%%%%%%%%%%%%%%%
Z = randn(M,N); %fixed random weights from input to RNN
W_in = randn(M); %fixed random weights within RNN

%initial conditions
%%%%%%%%%%%%%%%%%%%%%%%%
B = randn(N,T); %RNN output
Bhat = randn(N,T); %top-down output prediction
W = randn(M,N); %RNN output weights

%time constants
%%%%%%%%%%%%%%%%%%%%%%%%
tau = 0.1; %RNN time constant
tau_w = 0.1; %supervised plasticity time constant
tau_b = 0.1; %predictive plasticity time constant

%plasticity parameters
%%%%%%%%%%%%%%%%%%%%%%%%
gamma = 0.5; %supervised learning strength
sigma = 0.05; %predictive learning strength
num_iter = 100; %number of training iterations

%generate RNN activity
%%%%%%%%%%%%%%%%%%%%%%%%
I = Z*X;
K = zeros(M,T);
K(:,1) = I(:,1);
for t = 2:T/dt
    dK = -alpha.*K(:,t-1)+ReLU((K(:,t-1)'*W_in)')+I(:,t)+1;    
    K(:,t) = K(:,t)+tau.*dK;
end

%scaling (optional - helps with plotting)
Ks = scaledata(K,0,1/(M^2));

%main loop for synaptic plasticity
sup_err = zeros(num_iter,1);
pred_err = zeros(num_iter,1);
for i = 1:num_iter
    
    %combined supervised and predictive learning rule
    dW = (pinv(Ks'))*(gamma.*(Y-B))' + (pinv(Ks'))*(-sigma.*(B-Bhat))';
    W = W + tau_w.*dW;
        
    %update top-down prediction
    dBhat = -Bhat + B;
    Bhat = Bhat + tau_b*dBhat;
    
    %network output
    B = (Ks'*W)';
    
    %supervised and predictive error    
    sup_err(i) = mean((B(:)-Y(:)).^2);
    pred_err(i) = mean((Bhat(:)-B(:)).^2);
end

%plot results
Bplot = scaledata(B,0,1);
Xplot = scaledata(Y,0,1);

figure;
subplot(2,2,[1 2]);
hold on;
plot(Bplot(1,10:T),'linewidth',12);
plot(Xplot(1,10:T),'Color',[217 83 25]./255,'linewidth',6);
axis off
set(gca,'PlotBoxAspectRatio',[5 1 1])
legend output target;

subplot(2,2,3);
hold on;
plot(sup_err,'k','linewidth',5);
set(gca,'FontSize',18);
set(gca,'Linewidth',3);
xlabel('time-steps');
ylabel('MSE');
set(gca,'PlotBoxAspectRatio',[3 1 1])
title('supervised error');

subplot(2,2,4)
hold on;
plot(pred_err,'k','linewidth',5);
set(gca,'FontSize',18);
set(gca,'Linewidth',3);
xlabel('time-steps');
ylabel('MSE');
set(gca,'PlotBoxAspectRatio',[3 1 1])
title('predictive error');