clc;
clear all
close all
% dim = input('please input a integral number:');
dim = 3;
batch_size = 1600;
XYZ = rand(batch_size, dim);
save('testXYZ.mat', 'XYZ')

% rand函数产生由在(0, 1)之间均匀分布的随机数组成的数组。