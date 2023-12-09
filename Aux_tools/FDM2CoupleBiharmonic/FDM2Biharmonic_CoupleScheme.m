%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%          Matrix method for Biharmonic Equation      %%%%
%%%     u_{xxxx} + u_{yyyy} + 2*u_{xx}*u_{yy} = f(x, y)  %%%%
%%%           Omega = 0 < x < 1, 0 < y < 1               %%%%
%%%              u(x, y) = 0 on boundary,                %%%%  
%%%  Exact soln: u(x, y) = sin(pi*x)*sin(pi*y)           %%%%
%%%        Here f(x, y) = 4*pi^4*sin(pi*x)*sin(pi*y);   %%%%
%%%        Course:    Xi'an Li on 12.08 2023             %%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc
close all

ftsz = 20;

x_l = -1.0;
x_r = 1.0;
y_b = -1.0;
y_t = 1.0;

q = 6;
Num = 2^q+1;
NNN = Num*Num;

point_num2x = Num;
point_num2y = Num;

fside = @(x, y) 4*pi^4*sin(pi*x).*sin(pi*y);
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
Unumberi = FDM2Biharmonic_Couple2Dirichlet_Zero(point_num2x, point_num2y,...
                                                x_l, x_r, y_b, y_t, fside);
fprintf('%s%s%s\n','运行时间:',toc,'s')
U_exact = utrue(meshX, meshY);
meshErr = abs(U_exact - Unumberi);

rel_err = sum(sum(meshErr))/sum(sum(abs(U_exact)));
fprintf('%s%s\n','相对误差:',rel_err)

figure('name','Exact')
axis tight;
h = surf(meshX, meshY, U_exact','Edgecolor', 'none');
hold on
title('Exact Solu')
% xlabel('$x$', 'Fontsize', 20, 'Interpreter', 'latex')
% ylabel('$y$', 'Fontsize', 20, 'Interpreter', 'latex')
% zlabel('$Error$', 'Fontsize', 20, 'Interpreter', 'latex')
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
hold on
% colorbar;
% caxis([0 0.00012])
hold on

figure('name','Absolute Error')
axis tight;
h = surf(meshX, meshY, meshErr','Edgecolor', 'none');
hold on
title('Absolute Error')
% xlabel('$x$', 'Fontsize', 20, 'Interpreter', 'latex')
% ylabel('$y$', 'Fontsize', 20, 'Interpreter', 'latex')
% zlabel('$Error$', 'Fontsize', 20, 'Interpreter', 'latex')
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
hold on
% colorbar;
% caxis([0 0.00012])
hold on

if q==6
    txt2result = 'result2fdm_mesh6.txt';
elseif q==7
    txt2result = 'result2fdm_mesh7.txt';
elseif q==8
    txt2result = 'result2fdm_mesh8.txt';
elseif q==9
    txt2result = 'result2fdm_mesh9.txt';
end

fop = fopen(txt2result, 'wt');

fprintf(fop,'%s%s%s\n','运行时间:',toc,'s');
fprintf(fop,'%s%d\n','内部网格点数目:',Num-1);
fprintf(fop,'%s%s\n','相对误差:',rel_err);


