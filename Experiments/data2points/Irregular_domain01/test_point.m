clear all
clc
close all

point = load('testXY.mat');

XY = point.XY;
X = XY(1,:);
Y = XY(2,:);
Z = sin(pi*X).*sin(pi*Y);
plot3(X,Y,Z,'b.')