clc
clear all
close all
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
geom.xstart = 0;
geom.xend = 1;
geom.ystart = 0;
geom.yend = 1;


geom = assemble_fmesh(geom);

meshXY = geom.pin;          % the point pair for interior of square of [a,b]X[c,d], not including edges
Y_coords = 0.5*ones(1, geom.N);

XYZ = [meshXY(1,:);Y_coords; meshXY(2,:)];
save('testXYZ.mat','XYZ')