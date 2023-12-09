%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%          Matrix method for Poisson Equation         %%%%
%%%     u_{xx} + u_{yy} = f(x, y), 0 < x < 1, 0 < y < 1  %%%%
%%%              u(x, y) = 0 on boundary,                %%%%  
%%%  Exact soln: u(x, y) = sin(pi*x)*sin(pi*y)           %%%%
%%%        Here f(x, y) = -2*pi^2*sin(pi*x)*sin(pi*y);   %%%%
%%%        Course:    Xi'an Li on 12.08 2023             %%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc
close all

x_l = 0.0;
x_r = 1.0;
y_b = 0.0;
y_t = 1.0;

q = 8;
Num = 2^q+1;
NNN = Num*Num;

point_num2x = Num;
point_num2y = Num;
fside = @(x, y) -2*pi^2*sin(pi*x).*sin(pi*y);
utrue = @(x, y) sin(pi*x).*sin(pi*y);

hx = (x_r-x_l)/point_num2x; 
X = zeros(point_num2x-1,1);
for i=1:point_num2x-1
  X(i) = x_l+i*hx;
end

hy = (y_t-y_b)/point_num2y; 
Y=zeros(point_num2y-1,1);
for i=1:point_num2y-1
  Y(i) = y_b+i*hy;
end
[meshX, meshY] = meshgrid(X, Y);

tic;
Unumberi = FDM2Possion_Dirchlet_Zero(point_num2x, point_num2y, x_l, x_r, ...
                                     y_b, y_t, fside);
fprintf('%s%s%s\n','运行时间:',toc,'s')
U_exact = utrue(meshX, meshY);
meshErr = abs(U_exact - Unumberi);

rel_err = sum(sum(meshErr))/sum(sum(abs(U_exact)));
fprintf('%s%s\n','相对误差:',rel_err)

figure('name','Absolute Error')
axis tight;
h = surf(meshX, meshY, meshErr','Edgecolor', 'none');
hold on
% xlabel('$x$', 'Fontsize', 20, 'Interpreter', 'latex')
% ylabel('$y$', 'Fontsize', 20, 'Interpreter', 'latex')
% zlabel('$Error$', 'Fontsize', 20, 'Interpreter', 'latex')
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 18);
set(gcf, 'Renderer', 'zbuffer');
hold on
colorbar;
% caxis([0 0.00012])
hold on