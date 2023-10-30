clc;
clear all
close all
% dim = input('please input a integral number:');
dim = 4;
batch_size = 1600;
XYZS = rand(batch_size, dim);
save('testXYZS.mat', 'XYZS')

% rand函数产生由在(0, 1)之间均匀分布的随机数组成的数组。