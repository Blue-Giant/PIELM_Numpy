clc;
clear all
close all
% dim = input('please input a integral number:');
dim = 5;
batch_size = 1600;
XYZST = rand(batch_size, dim);
save('testXYZST.mat', 'XYZST')

% rand������������(0, 1)֮����ȷֲ����������ɵ����顣