function Uapp = FDM2Possion_Dirchlet_Zero(Nx, Ny, xleft, xright, ybottom, ytop, fside)
    format long;

    % Define the step sizes and create the grid without boundary points
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

    % Define the source term
    source_term = fside;

    % Initialize the coefficient matrix A and the right-hand side vector F
    N = (Ny-1)*(Nx-1);
    A = sparse(N,N); 
    F = zeros(N,1);

    % Loop through each inner grid point, Apply finite difference scheme (central differences)
    hx1 = hx*hx; 
    hy1 = hy*hy; 
    for j = 1:Ny-1
      for i=1:Nx-1
        k = i + (j-1)*(Nx-1);
        F(k) = fside(x(i),y(j));
        A(k,k) = -2/hx1 -2/hy1;
        
        %-- x direction --------------
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
      j = fix(k/(Nx-1)) + 1;
    end
end