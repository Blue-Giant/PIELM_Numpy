clc
clear all
close all
% q = 6;
q = 7;
nl = 2;
T = [];
J = [];

% geom: (only square geometry available now)
% generating 2d square mesh for the region [-1, 1] x [-1 1]
geom.q = q;
geom.nl = nl;
geom.dim = 2;                      % dimension of the problem
geom.m = 2^geom.dim;               % 
geom.N1 = 2^q;                     % dofs in one dimension, uniformly sampling 2^q points in range (a, b), not including edges
geom.N = (geom.m)^geom.q;          % dofs in the domain
geom.xstart = -1;
geom.xend = 1;
geom.ystart = -1;
geom.yend = 1;


geom = assemble_fmesh(geom);

meshXY = geom.pin;          % the point pair for interior of square of [a,b]X[c,d], not including edges

if q == 3
    save('meshXY3.mat', 'meshXY')
elseif q == 4
    save('meshXY4.mat', 'meshXY')
elseif q == 5
    save('meshXY5.mat', 'meshXY')
elseif q == 6
    save('meshXY6.mat', 'meshXY')
elseif q == 7
    save('meshXY7.mat', 'meshXY')
end