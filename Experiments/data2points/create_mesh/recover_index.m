function index=recover_index(ind_in,N)
% ind_in ��������˳�� 1,2,3,4,5,6,7������������
M = N*N;
ind1 = ind_in;
% �������Ϊ�ĶΣ��ڶ��Ķξ��񵹻�
for i=1:M/4
%     i
%     j = i+ M/4
%     k = M-i+1
    ind1(i+ M/4)=ind_in(M-i+1);
    ind1(M-i+1)=ind_in(i+ M/4);
end

% �������Ϊ���Σ�ǰ��ż��λ�û���
ind2 = ind1;
for i=1:M/2
    if mod(i,2)==0
        ind2(i)=ind1(i+M/2);
        ind2(i+M/2)=ind1(i);
    end
end

% ת��Ϊ����Ȼ��ת�ã��ٱ�Ϊ����
ind_mat = reshape(ind2,[N,N]);
trans_ind_mat = ind_mat';
ind3 = reshape(trans_ind_mat,[1, N*N]);

% �������Ϊ�ĶΣ���һ���ξ��񵹻�
ind4 = ind3;
for i=1:M/4
    ind4(i)=ind3(3*M/4-i+1);
    ind4(3*M/4-i+1)=ind3(i);
end

% ǰ��ε�ż��λ�����������λ�ý�����[2,1,4,3,6,   5,8,7,10,9]-->[2,10,4,8,6    5,3,7,1,9]
ind5 = ind4;
for i=1:M/2
    if mod(i,2)==1
        ind5(i)=ind4(M-i+1);
        ind5(M-i+1)=ind4(i);
    end
end

% ����λ�ý�������[1,2,3,4,5,6,7,8,9,10]-->[2,1,4,3,6,5,8,7,10,9]
index = ind5;
for i=1:M
    if mod(i,2)==0
        index(i)=ind5(i-1);
    else
        index(i)=ind5(i+1);
    end
end
sort_index = sort(index);
disp('xxxxxx')
end