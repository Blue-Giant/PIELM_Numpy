clear all
close all
clc

pi=3.1415926;

q=6;
N=2^q;
xstart = 0;
xend =1;
ystart =0;
yend = 1;
hx = (xend - xstart)/(N+1);  % the step-size for x-direction
hy = (yend - ystart)/(N+1);  % the step-size for y-direction
[meshx, meshy] = meshgrid(xstart:hx:xend, ystart:hy:yend);

x = meshx(:)';  % changing the mesh-mat of x into a vector, up-->down, then left-->right
y = meshy(:)';  % changing the mesh-mat of y into a vector, up-->down, then left-->right

Uniform_XY = [x;y];
N1=N+2;
index2all = linspace(1, N1*N1,N1*N1);
shuffle_index = reorder_index(index2all,N1);
XY = Uniform_XY(:,shuffle_index);
save('testXY','XY')

% 如下为测试我们的方法。具体的可参见https://blog.csdn.net/kkxi123456/article/details/119798987
% umesh = sin(pi*meshx).*sin(pi*meshy);
% figure('name','umesh')
% h1 = surf(meshx, meshy, umesh);
% hold on
% colorbar;
% caxis([0 1.0])
% ftsz = 14;
% set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
% set(gcf, 'Renderer', 'zbuffer');

% 将一致性网格打乱
% N1=N+2;
% index2all2u = linspace(1, N1*N1,N1*N1);
% shuffle_index2u = reorder_index(index2all2u,N1);
% reorder_points = points(:,shuffle_index2u);
% urand = sin(pi*reorder_points(1,:)).*sin(pi*reorder_points(2,:));
% figure('name','urand')
% urand_mat = reshape(urand,[N1,N1]);
% h2 = surf(meshx, meshy, urand_mat);
% hold on
% colorbar;
% caxis([0 1.0])
% ftsz = 14;
% set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
% set(gcf, 'Renderer', 'zbuffer');

% 对于打乱的数据，进行恢复
% u_recover = recover_index(urand, N1);
% figure('name','u_recover')
% ur_mat = reshape(u_recover,[N1,N1]);
% h3 = surf(meshx, meshy, ur_mat);
% hold on
% colorbar;
% caxis([0 1.0])
% ftsz = 14;
% set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
% set(gcf, 'Renderer', 'zbuffer');
