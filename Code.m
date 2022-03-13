close all
clear
clc
load('data.txt');
alpha = data(:,1);
cl = data(:,2);
cd = data(:,3);
% CL Vs Alpha
coeff_cl = linear(alpha,cl);
X_cl = alpha;
Y_cl = coeff_cl(1)*X_cl+coeff_cl(2);
% CD Vs CL
coeff_cd = quadratic(cl,cd);
X_cd = cl;
Y_cd = coeff_cd(1)*(X_cd.^2)+coeff_cd(2)*(X_cd)+coeff_cd(3);

tiledlayout(1,2)

nexttile;
plot(alpha,Y_cl,'--k',alpha,cl,'.r')
title('Linear Least Square Fit Curve','$C_{L}$ Vs $\alpha$',...
    'interpreter','latex','FontSize',22)
xlabel('$\alpha$','interpreter','latex','FontSize',22)
ylabel('$C_{L}$','interpreter','latex','FontSize',22)
xlim([1.1*min(X_cl) 1.1*max(X_cl)]);
ylim([1.1*min(Y_cl) 1.1*max(Y_cl)]);
legend('Fitted curve','Given data','Location','best')
box on
grid on
grid minor

nexttile;
plot(X_cd,Y_cd,'--k',X_cd,cd,'.r');
title('Quadratic Least Square Fit Curve','$C_{D}$ Vs $C_{L}$',...
    'interpreter','latex','FontSize',22)
xlabel('$C_{L}$','interpreter','latex','FontSize',22)
ylabel('$C_{D}$','interpreter','latex','FontSize',22)
xlim([1.1*min(X_cd) 1.1*max(X_cd)]);
ylim([0 1.1*max(Y_cd)]);
legend('Fitted curve','Given data','Location','best')
box on 
grid on
grid minor


function [mtx] = linear(x,y)
a22 = length(x);
a12 = sum(x);
a21 = sum(x);
a11 = sum(x.^2);
b21 = sum(y);
b11 = sum(x.*y);
A = [a11,a12;a21,a22];
B = [b11;b21];
mtx = (inv(A))*B;
end

function[mtx] = quadratic(x,y)
a11 = sum(x.^4);
a12 = sum(x.^3);
a13 = sum(x.^2);
a21 = sum(x.^3);
a22 = sum(x.^2);
a23 = sum(x);
a31 = sum(x.^2);
a32 = sum(x);
a33 = length(x);
b11 = sum((x.^2).*y);
b21 = sum(x.*y);
b31 = sum(y);
A = [a11,a12,a13;a21,a22,a23;a31,a32,a33];
B = [b11;b21;b31];
mtx = (inv(A))*B;
end


