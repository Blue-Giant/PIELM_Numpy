function Uapp = FDM2Biharmonic_Couple2Dirichlet_Zero(Nx, Ny, xleft, xright, ybottom, ytop, fside)
    format long;

    % Define the step sizes and create the grid without boundary points
    hx = (xright-xleft)/Nx; 
    x = zeros(Nx-1,1);
    for ix=1:Nx-1
      x(ix) = xleft+ix*hx;
    end

    hy = (ytop-ybottom)/Ny; 
    y=zeros(Ny-1,1);
    for iy=1:Ny-1
      y(iy) = ybottom+iy*hy;
    end

    % Define the source term
    source_term = fside;

    % Initialize the coefficient matrix A and the right-hand side vector F
    N = (Ny-1)*(Nx-1);
    A = sparse(N,N); 
    FV = zeros(N,1);

    % Loop through each inner grid point, Apply finite difference scheme (central differences)
    hx1 = hx*hx; 
    hy1 = hy*hy; 
    for jv = 1:Ny-1
      for iv=1:Nx-1
        kv = iv + (jv-1)*(Nx-1);
        FV(kv) = fside(x(iv),y(jv));
        A(kv,kv) = -2/hx1 -2/hy1;
        
        %-- x direction --------------
        if iv == 1
            A(kv,kv+1) = 1/hx1;
        else
           if iv==Nx-1
             A(kv,kv-1) = 1/hx1;
           else
            A(kv,kv-1) = 1/hx1;
            A(kv,kv+1) = 1/hx1;
           end
        end
        %-- y direction --------------
        if jv == 1
            A(kv,kv+Nx-1) = 1/hy1;
        else
           if jv==Ny-1
             A(kv,kv-(Nx-1)) = 1/hy1;
           else
              A(kv,kv-(Nx-1)) = 1/hy1;
              A(kv,kv+Nx-1) = 1/hy1;
           end
        end
      end
    end
    V = A\FV;

    B = sparse(N,N); 
    FU = zeros(N,1);

    % Loop through each inner grid point, Apply finite difference scheme (central differences)
    for ju = 1:Ny-1
      for iu=1:Nx-1
        ku = iu + (ju-1)*(Nx-1);
        FV(ku) = V(ku);
        B(ku,ku) = -2/hx1 -2/hy1;
        
        %-- x direction --------------
        if iu == 1
            B(ku,ku+1) = 1/hx1;
        else
           if iu==Nx-1
             B(ku,ku-1) = 1/hx1;
           else
            B(ku,ku-1) = 1/hx1;
            B(ku,ku+1) = 1/hx1;
           end
        end
        %-- y direction --------------
        if ju == 1
            B(ku,ku+Nx-1) = 1/hy1;
        else
           if ju==Ny-1
             B(ku,ku-(Nx-1)) = 1/hy1;
           else
              B(ku,ku-(Nx-1)) = 1/hy1;
              B(ku,ku+Nx-1) = 1/hy1;
           end
        end
      end
    end
    
    U = B\FV;
    %--- Transform back to (i,j) form to plot the solution ---
    j = 1;
    for k=1:N
      i = k - (j-1)*(Nx-1) ;
      Uapp(i,j) = U(k);
      j = fix(k/(Nx-1)) + 1;
    end
end