clc;
clear all
close all
meshData = load('meshXY7.mat');
meshXY = meshData.UMESHXY;

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
geom.xstart = -1;
geom.xend = 1;
geom.ystart = -1;
geom.yend = 1;

geom = assemble_fmesh(geom);


data2solution = load('test_solus.mat');
Solu_True = data2solution.Utrue;
Solu_DNN = data2solution.Us2relu;
figure('name','Exact_Solu')
mesh_solu_true = plot_fun(geom,Solu_True);
hold on

figure('name','DNN_Solu')
mesh_solu_dnn = plot_fun(geom,Solu_DNN);
hold on

err2solu = abs(Solu_True-Solu_DNN);
figure('name','Err2solu')
mesh_solu_err = plot_fun2in(geom,err2solu);
title('Absolute Error')
hold on
colorbar;
caxis([0 0.025])
hold on
