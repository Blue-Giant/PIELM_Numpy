clc;
clear all
close all
ftsz = 15;

testPoint = load('testData2XY.mat');
XY = (testPoint.Points2XY)';
X = XY(1,:);
Y = XY(2, :);

solu2exact = load('Utrue');
solu2numerical = load('Unumeri');
Solu_UTrue = solu2exact.UTRUE;
Solu_UDNN = solu2numerical.UNUMERI;

figure('name','Exact_Solu')
ct = Solu_UTrue;
scatter_solu2u = scatter3(X, Y, Solu_UTrue,50, ct, '.');
shading interp
hold on
colorbar;
caxis([0 1])
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
hold on

figure('name','DNN_Solu')
cn = Solu_UDNN;
mesh_solu_unn =scatter3(X, Y, Solu_UDNN, 50, cn, '.');
shading interp
hold on
colorbar;
caxis([0 1])
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
hold on

err2solu = abs(Solu_UTrue-Solu_UDNN);
figure('name','Err2solu')
cp = err2solu;
mesh_solu_err2u = scatter3(X, Y, err2solu, 50, cp, '.');
shading interp
title('Absolute Error')
hold on
colorbar;
caxis([0 1e-8])
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
hold on


