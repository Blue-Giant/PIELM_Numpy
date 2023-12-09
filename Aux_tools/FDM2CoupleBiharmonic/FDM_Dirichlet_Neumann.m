function [U, U_exact] = FDM_Dirichlet_Neumann(Nx, Ny, xleft, xright, ybottom, ytop, fside)
    % Define the step sizes and create the grid
    dx = 1/(Nx-1);
    dy = 1/(Ny-1);
    x = linspace(xleft, xright, Nx);
    y = linspace(ybottom, ytop, Ny);
    [X, Y] = meshgrid(x, y);

    % Initialize the coefficient matrix A and the right-hand side vector F
    A = zeros(Nx*Ny, Nx*Ny);
    F = zeros(Nx*Ny, 1);

    % Define the source term and the exact solution
%     source_term = @(x, y) 400*(x^4-y^4)*sin(20*x*y);
    source_term = fside;
    U_exact = (X.^2 - Y.^2) .* sin(20.*X.*Y);

    % Loop through each grid point
    for j = 1:Ny
        for i = 1:Nx
            n = (j-1)*Nx + i;

            % Apply finite difference scheme (central differences)
            if i > 1 && i < Nx && j > 1 && j < Ny
                A(n, n) = 2/dx^2 + 2/dy^2;
                A(n, n-1) = -1/dx^2;
                A(n, n+1) = -1/dx^2;
                A(n, n-Nx) = -1/dy^2;
                A(n, n+Nx) = -1/dy^2;
                F(n) = -source_term(x(i), y(j));
            else
            % Apply boundary conditions
                if j == 1
                    % Lower boundary (Dirichlet)
                    A(n, n) = 1;
                    F(n) = 0;
                elseif i == 1
                    % Left boundary (Dirichlet)
                    A(n, n) = 1;
                    F(n) = 0;
                elseif j == Ny
                    % Upper boundary (Neumann)
                    A(n, n) = 1;
                    A(n, n-Nx) = -1;
                    F(n) = -dy*(20*x(i)*(x(i)^2-1)*cos(20*x(i))-2*sin(20*x(i)));
                elseif i == Nx
                    % Right boundary (Neumann)
                    A(n, n) = 1;
                    A(n, n-1) = -1;
                    F(n) = -dx*(20*y(j)*(1-y(j)^2)*cos(20*y(j))+2*sin(20*y(j)));
                end
            end
        end
    end
% Solve the linear system
U = A\F;
U = reshape(U, [Nx, Ny]);
end