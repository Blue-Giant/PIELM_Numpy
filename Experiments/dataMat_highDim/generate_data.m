clc;
clear all
close all
% dim = input('please input a integral number:');
dim = 8;
batch_size = 1600;
XYZRSTVW = rand(batch_size, dim);
save('testXYZRSTVW.mat', 'XYZRSTVW')

