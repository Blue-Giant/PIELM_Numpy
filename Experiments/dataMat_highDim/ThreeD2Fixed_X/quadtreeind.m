function ind = quadtreeind(q)
% construct quadtree index 
m = 4;

M = [1 2; 3 4];

for k = 2:q
    M = [M M+m^(k-1); M+2*m^(k-1) M+3*m^(k-1)];
end

ind = M(:)';

end