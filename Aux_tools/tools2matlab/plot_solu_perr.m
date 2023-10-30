clc;
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
geom.L = 2; % side length 
geom.dim = 2; % dimension of the problem
geom.m = 2^geom.dim; % 
geom.N1 = 2^q; % dofs in one dimension
geom.N = (geom.m)^geom.q; % dofs in the domain
geom.h = geom.L/(geom.N1+1); % grid size
geom.xstart = 0;
geom.xend = 1;
geom.ystart = 0;
geom.yend = 1;

geom = assemble_fmesh(geom);


solu2exact = load('Utrue');
solu2numerical = load('Unumeri');
Solu_UTrue = solu2exact.UTRUE;
Solu_UDNN = solu2numerical.UNUMERI;
figure('name','Exact_Solu')
mesh_solu2u = plot_fun2in(geom,Solu_UTrue);
hold on

figure('name','DNN_Solu')
mesh_solu_unn = plot_fun2in(geom,Solu_UDNN);
hold on

err2solu = abs(Solu_UTrue-Solu_UDNN);
figure('name','Err2solu')
mesh_solu_err2u = plot_fun2in(geom,err2solu);
title('Absolute Error')
hold on
colorbar;
caxis([0 1e-13])
hold on


