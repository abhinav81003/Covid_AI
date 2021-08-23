function [J, DVec] = costFunction(thetaVec)

data = load('covid_test.txt');
X = data([1:5000],[1:12]);
y = data([1:5000], 13);
m = length(y);
X = [ones(m,1) X];


%% =========== Initialize properties ======= %%

lambda = 0.001;

hidden_layer_size = 120;
input_layer_size = 12;
num_labels = 1;


Theta1 = reshape(thetaVec(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(thetaVec((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


Delta = 0;

%% ============ Need to compute ============ %%
J = 0;
DVec = zeros(size(thetaVec),1);
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%% ================= Cost ================== %%



        a1 = X';
        z2 = Theta1*a1;
        a2 = tanh(z2);
        a2 = [ones(1,m); a2];
        z3 = Theta2*a2;
        a3 = sigmoid(z3);
        J = J + -y'*log(a3)'-((ones(1,m)-y')*log(ones(1,m)-a3)');
    
J = (1/m)*J;

regularcost = 0;

for i= 1:size(Theta1,1),
    regularcost = regularcost + (Theta1(i,[2:size(Theta1,2)]).^2)*ones(size(Theta1,2)-1,1);
    end;

for i= 1:size(Theta2,1),
    regularcost = regularcost + (Theta2(i,[2:size(Theta2,2)]).^2)*ones(size(Theta2,2)-1,1);
    end;

regularcost = (lambda/(2*m))*regularcost;

J = J + regularcost;

%% ================= BackProp ================== %%


D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

    a1 = X';
    z2 = Theta1*a1;
    a2 = tanh(z2);
    a2 = [ones(1,m); a2];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    delta3 = a3 - y';
    delta2 = Theta2(:,[2:end])'*delta3.*sigmoidGradient(z2);
    D1 = D1 + delta2*a1';
    D2 = D2 + delta3*a2';
    
% -------------------------------------------------------------

Theta1_grad = (1/m)*D1;
Theta2_grad = (1/m)*D2;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);

% =========================================================================

% Unroll gradients
DVec = [Theta1_grad(:) ; Theta2_grad(:)];
    


