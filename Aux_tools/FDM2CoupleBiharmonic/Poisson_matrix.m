%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%          Matrix method for Poisson Equation         %%%%
%%%     u_{xx} + u_{yy} = f(x, y), 0 < x < 1, 0 < y < 1  %%%%
%%%              u(x, y) = 0 on boundary,                %%%%  
%%%  Exact soln: u(x, y) = sin(pi*x)*sin(pi*y)           %%%%
%%%        Here f(x, y) = -2*pi^2*sin(pi*x)*sin(pi*y);   %%%%
%%%        Course: MATH F422, Dr. P. Dhanumjaya          %%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;

fpoisson = @(x, y) -2*pi^2*sin(pi*x).*sin(pi*y);
uexact = @(x, y) sin(pi*x).*sin(pi*y);

xleft = 0; 
xright = 1; 
ybottom = 0; 
ytop = 1;

Nx = 100; 
Ny = 100;

format long;

hx = (xright-xleft)/Nx; 
x = zeros(Nx-1,1);
for i=1:Nx-1
  x(i) = xleft+i*hx;
end

hy = (ytop-ybottom)/Ny; 
y=zeros(Ny-1,1);
for i=1:Ny-1
  y(i) = ybottom+i*hy;
end


N = (Ny-1)*(Nx-1);
A = sparse(N,N); 
F = zeros(N,1);

hx1 = hx*hx; 
hy1 = hy*hy; 
for j = 1:Ny-1
  for i=1:Nx-1
    k = i + (j-1)*(Nx-1);
    F(k) = fpoisson(x(i),y(j));
    A(k,k) = -2/hx1 -2/hy1;
    if i == 1
        A(k,k+1) = 1/hx1;
    else
       if i==Nx-1
         A(k,k-1) = 1/hx1;
       else
        A(k,k-1) = 1/hx1;
        A(k,k+1) = 1/hx1;
       end
    end
    %-- y direction --------------
    if j == 1
        A(k,k+Nx-1) = 1/hy1;
    else
       if j==Ny-1
         A(k,k-(Nx-1)) = 1/hy1;
       else
          A(k,k-(Nx-1)) = 1/hy1;
          A(k,k+Nx-1) = 1/hy1;
       end
    end
  end
end

U = A\F;

%--- Transform back to (i,j) form to plot the solution ---
j = 1;
for k=1:N
  i = k - (j-1)*(Nx-1) ;
  Uapp(i,j) = U(k);
  Uex(i,j) = uexact(x(i),y(j));
  j = fix(k/(Nx-1)) + 1;
end

% Analyze abd Visualize the result.

e = max( max( abs(Uapp-Uex)))        % The maximum error

surf(x,y,Uapp); 
title('The Approximate solution plot'); 
xlabel('x'); 
ylabel('y');

figure(2); 
surf(x,y,Uex); 
title('The Exact solution plot'); 
xlabel('x');
ylabel('y');
