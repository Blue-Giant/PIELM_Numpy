%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%       div(a，grad U) = f(x,y) (x,y)in[0,1]X[0,1]
%%%             u = 0 on boundary.
%%%          a = 2+sin(pi*x/eps)*sin(pi*y/eps)
%%%          u_exact = sin(pi*x)*sin(pi*y)
%%%
%%%   dv(a，grad u) = a_x，u_x + a_y，u_y + a，delta u 
%%% u_x = 0.5*[u(i+1,j) - u(i-1,j)]; u_y = 0.5*[u(i,j+1) - u(i,j-1)]
clc
clear all
close all

eps = 0.1;
Aeps = @(XX,YY)  2.0+sin(pi*XX/eps)*sin(pi*YY/eps);
dAeps2dx = @(XX,YY)  (pi/eps)*cos(pi*XX/eps)*sin(pi*YY/eps);
dAeps2dy = @(XX,YY)  (pi/eps)*sin(pi*XX/eps)*cos(pi*YY/eps);
f = @(XX,YY)  -10.0;

N = 100;
a = 0; b = 1;
c = 0; d = 1;
h = (b-a)/N;

x = a:h:b;
y = c:h:d;

u0 = zeros(N+1);
for j=1:N+1
    u0(j,1) = 0;
    u0(j,N+1) = 0;
    u0(1,j) = 0;
    u0(N+1,j) = 0;
end
u = u0;

error = 100;
tol = 10^-8;

while error > tol
    for i=2:N
        for j=2:N
            Ax = dAeps2dx(x(i),y(j))/Aeps(x(i),y(j));
            Ay = dAeps2dy(x(i),y(j))/Aeps(x(i),y(j));
            fa = f(x(i),y(j))/Aeps(x(i),y(j));
            temp =Ax*0.5*h*(u0(i+1,j)-u0(i-1,j))+Ay*0.5*h*(u0(i,j+1)-u0(i,j-1))+u0(i-1,j)+u0(i+1,j)+u0(i,j-1)+u0(i,j+1) - h^2*fa;
            u(i,j) = 0.25*temp;
        end
    end
    error = max(max(abs(u-u0)));
    u0 = u;
end

figure(1)
surf(y,x,u);
title('Approx');