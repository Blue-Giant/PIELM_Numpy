% plot function which is defined on the iterior dofs
% function plot_fun(p, e, t, in, ind, u) % with triangular mesh
% u1 = zeros(size(p,2), 1);
% u1(in) = u(ind);
% pdeplot(p,e,t,'xydata',u1,'zdata',u1);
% end

function mesh_u=plot_fun2in(geom, u)
meshX = geom.X;
meshY = geom.Y;

meshX_in = meshX(2:end-1, 2:end-1);
meshY_in = meshY(2:end-1, 2:end-1);

ftsz = 14;
% require u to be a row vector
if size(u, 2) == 1
    u=u';
end

if length(u)==sum(geom.in) % if u only defined on interior dofs
    u0 = u(geom.ind);
    u = reshape(u0, size(meshX_in))';
else
    error('dof of u does not match dof of mesh')
    return
end

mesh_u = u;

% surf(X, Y, u, 'EdgeColor', 'none', 'FaceColor', 'none', 'FaceAlpha', 0.9);
axis tight;
% surf(geom.X, geom.Y, u, 'Edgecolor', 'none');
% h = surf(geom.X, geom.Y, u, 'Edgecolor', 'none');

h = surf(meshX_in, meshY_in, u,'Edgecolor', 'none');
% colorbar;
% caxis([0 0.5])
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
end