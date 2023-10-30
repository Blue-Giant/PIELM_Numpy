function geom = assemble_fmesh(geom)
%ASSEMBLE_FMESH generate fine mesh for [-1, 1] x [-1, 1]
% this requires the side length geom.L = (geom.N1+1)*geom.h

if nargin == 0
    test_assemble_fmesh;
    return
end

hx = (geom.xend - geom.xstart)/(geom.N1+1);
hy = (geom.yend - geom.ystart)/(geom.N1+1);
xline = geom.xstart:hx: geom.xend;
yline = geom.ystart:hy:geom.yend;
[x, y] = meshgrid(geom.xstart:hx:geom.xend, geom.ystart:hy:geom.yend);
geom.X = x;
geom.Y = y;

x = x'; x = x(:)'; 
y = y'; y = y(:)';

p = [x; y];

% create the element matrix t for square elements
N = geom.N1 + 1;
t = zeros(4, N^2);
nb = speye(N^2, N^2);
for i = 1:N
    for j = 1:N
        ind = i + N*(j-1);
        t(:, ind) = [
            (j-1)*(N+1)+i;
            (j-1)*(N+1)+i+1;
            j*(N+1)+i;
            j*(N+1)+i+1;
            ];
        % pt is the center of elements t
        pt(:, ind) = sum(p(:, t(:, ind)), 2)/4;
    end
end

% identify interior dofs and boundary dofs
ib = abs(x)==geom.xstart | abs(x)==geom.xend | abs(y)==geom.ystart | abs(y)==geom.yend;
in = not(ib);
pin = p(:, in);

% [ind, ~, ~, adjmatrix, ~, ~] = quadtree(pin(1, :), pin(2, :), [], 1);
% ind is the index of regions
% bx, by denotes the order in x and y direction respectively
% nb is the adjacency matrix of regions
ind = quadtreeind(geom.q);

% now reorder the dofs by the regions, this create a natural order
% multiresolution operations by using the base 4 representation of indices
% r1 = pin(1, ind)
% r2 = pin(2, ind)
pin(:, ind) = pin;
nb = sparse(nb);

geom.p = p;
geom.t = t;
geom.pt = pt;
geom.ib = ib;
geom.in = in;
geom.ind = ind;
geom.pin = pin;
% geom.adjmatrix = adjmatrix;
end

function test_assemble_fmesh
q = 3;
nl = 2;
T = [];
J = [];

% geom: (only square geometry available now)
% generating 2d square mesh for the region [-1, 1] x [-1 1]
geom.q = q;
geom.nl = nl;
geom.L = 2; % side length 
geom.dim = 2; % dimension of the problem
geom.m = 2^geom.dim; % 
geom.N1 = 2^q; % dofs in one dimension
geom.N = (geom.m)^geom.q; % dofs in the domain
geom.h = geom.L/(geom.N1+1); % grid size

geom = assemble_fmesh(geom);

meshin = geom.pin;

end