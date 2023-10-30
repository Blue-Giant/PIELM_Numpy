function solu=recover_solu(solu_in,N)
M = N*N;
ind1 = solu_in;
% 将数组分为四段，第二四段镜像倒换
for i=1:M/4
%     i
%     j = i+ M/4
%     k = M-i+1
    ind1(i+ M/4)=solu_in(M-i+1);
    ind1(M-i+1)=solu_in(i+ M/4);
end

% 将数组分为两段，前后偶数位置互换
ind2 = ind1;
for i=1:M/2
    if mod(i,2)==0
        ind2(i)=ind1(i+M/2);
        ind2(i+M/2)=ind1(i);
    end
end

% 转换为矩阵，然后转置，再变为向量
ind_mat = reshape(ind2,[N,N]);
trans_ind_mat = ind_mat';
ind3 = reshape(trans_ind_mat,[1, N*N]);

% 将数组分为四段，第一三段镜像倒换
ind4 = ind3;
for i=1:M/4
    ind4(i)=ind3(3*M/4-i+1);
    ind4(3*M/4-i+1)=ind3(i);
end

% 前半段的偶数位置与后半段奇数位置交换，[2,1,4,3,6,   5,8,7,10,9]-->[2,10,4,8,6    5,3,7,1,9]
ind5 = ind4;
for i=1:M/2
    if mod(i,2)==1
        ind5(i)=ind4(M-i+1);
        ind5(M-i+1)=ind4(i);
    end
end

% 相邻位置交换，如[1,2,3,4,5,6,7,8,9,10]-->[2,1,4,3,6,5,8,7,10,9]
solu = ind5;
for i=1:M
    if mod(i,2)==0
        solu(i)=ind5(i-1);
    else
        solu(i)=ind5(i+1);
    end
end
disp('xxxxxx')
end