function index=reorder_index(ind_in, N)
% This function is used to reorder the index of element for a given matrix,
% then return the reorder index
% N is a number, it represents the number of row and column for matrix
M = N*N;
% 相邻位置交换，如[1,2,3,4,5,6,7,8,9,10]-->[2,1,4,3,6,5,8,7,10,9]
ind1 = ind_in;
for i=1:M
    if mod(i,2)==0
        ind1(i)=ind_in(i-1);
    else
        ind1(i)=ind_in(i+1);
    end
end

% 前半段的偶数位置与后半段奇数位置交换，[2,1,4,3,6,   5,8,7,10,9]-->[2,10,4,8,6    5,3,7,1,9]
ind2 = ind1;
for i=1:M/2
    if mod(i,2)==1
        ind2(i)=ind1(M-i+1);
        ind2(M-i+1)=ind1(i);
    end
end

% 将数组分为四段，第一三段镜像倒换
ind3 = ind2;
for i=1:M/4
    i;
    j=3*M/4-i+1;
    ind3(i)=ind2(3*M/4-i+1);
    ind3(3*M/4-i+1)=ind2(i);
end

% 转换为矩阵，然后转置，再变为向量
ind_mat = reshape(ind3, [N,N]);
trans_ind_mat = ind_mat';
ind4 = reshape(trans_ind_mat, [1,M]);

% 将数组分为两段，前后偶数位置互换
ind5 = ind4;
for i=1:M/2
    if mod(i,2)==0
        i;
        ind5(i)=ind4(i+M/2);
        ind5(i+M/2)=ind4(i);
    end
end

index = ind5;
% 将数组分为四段，第二四段镜像倒换
for i=1:M/4
%     i
%     j = i+ M/4
%     k = M-i+1
    index(i+ M/4)=ind5(M-i+1);
    index(M-i+1)=ind5(i+M/4);
end

sort_index = sort(index);
disp('xxxxxx')
end