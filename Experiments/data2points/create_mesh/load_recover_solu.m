clc
clear all
close all
q = 6;
N = 2^q+1;
if q==4
    data2mesh = load('meshXY4');
    data2solu = load('utrue4');
elseif q==5
    data2mesh = load('meshXY5');
    data2solu = load('utrue5');
elseif q==6
    data2mesh = load('meshXY6');
    data2solu = load('utrue6');
elseif q==7
    data2mesh = load('meshXY7');
    data2solu = load('utrue7');
end

xpoints = -1:2/N:1;
ypoints = -1:2/N:1;
[meshX, meshY] = meshgrid(xpoints, ypoints);
solu1 = data2solu.utrue;

scatter_u = reshape(solu1, [N+1, N+1]);
figure('name', 'scatter_solu')
surf(meshX, meshY, scatter_u)
hold on


solu = recover_solu(solu1, N+1);
u = reshape(solu, [N+1, N+1]);
figure('name', 'recover_solu')
surf(meshX, meshY, u)
hold on
