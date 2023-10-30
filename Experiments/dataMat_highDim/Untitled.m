clear all
clc
close all
data2xyz=load('testXYZ.mat');
xyz = data2xyz.XYZ;
min(xyz)
max(xyz)