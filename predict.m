function p = predict(X,thetaVec)

X = [1;X];

Theta1 = reshape(thetaVec(1:1040),80,13);
Theta2 = reshape(thetaVec(1041:7520),80,81);
Theta3 = reshape(thetaVec(7521:14000),80,81);
Theta4 = reshape(thetaVec(14001:end),1,81);

	a1 = X;
        z2 = Theta1*a1;
        a2 = sigmoid(z2);
        a2 = [1; a2];
        z3 = Theta2*a2;
        a3 = sigmoid(z3);
        a3 = [1; a3];
        z4 = Theta3*a3;
        a4 = sigmoid(z4);
	a4 = [1; a4];
        z5 = Theta4*a4;
        p = sigmoid(z5);