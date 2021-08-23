%% ============ Part 1: Load Features ============ %%
data = load('covid_test.txt');
X = data([1:5000],[1:12]);
y = data([1:5000], 13);

%% ============ Part 2: Compute Features ============ %%
m = size(X,1);
X = [ones(m,1) X];
E1 = 0.49;


init_theta1 = rand(120,13)*2*E1 - E1;
init_theta2 = rand(1,121)*2*E1 - E1;
init_thetaVec = [init_theta1(:);init_theta2(:)];

%% ============ Part 3: Compute Cost ============ %%



%% options = optimset('GradObj','on','MaxIter',1000);
%% [optimthetaVec, cost] = fminunc(@(t)(costFunction(t,X,y)),init_thetaVec,options);



options = optimset('MaxIter',3000);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costFunction(p, init_thetaVec);

[optim_theta, cost] = fmincg(costFunction, init_thetaVec, options);


iter = 6000;
alpha = 0.038;
thetaVec = init_thetaVec;
costmatrix = ones(iter,1);
[cost,gradient] = costFunction(thetaVec);

for i = 1:iter,
    thetaVec = thetaVec - alpha*gradient;
    [cost,gradient] = costFunction(thetaVec);
    costmatrix(i)=cost;
    end;

final_theta = thetaVec;