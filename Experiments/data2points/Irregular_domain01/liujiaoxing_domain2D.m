% rectangle('position',[1,1,5,5],'curvature',[1,1],'edgecolor','r','facecolor','g');
% 'position',[1,1,5,5]表示从（1,1）点开始高为5，宽为5；
% 'curvature',[1,1]表示x,y方向上的曲率都为1，即是圆弧；
% 'edgecolor','r'表示边框颜色是红色；
% 'facecolor','g'表示面内填充颜色为绿色。

clear all
clc
close all
N = 25000;
x = rand(1, N);
y = rand(1, N);
xy = [x;y];
k1 = 1;
domain = zeros(2, N);
for i =1:N
    if (xy(2,i)>0.25) && (1.5*xy(1,i)+0.25-xy(2,i)>0) && (-1.5*xy(1,i)+1.75-xy(2,i)>0)
        domain(1,k1)=xy(1,i);
        domain(2,k1)=xy(2,i);
        k1=k1+1;
    end
end
k2=k1;
for i =1:N
    if (xy(2,i)<0.75) && (-1.5*xy(1,i)+0.75-xy(2,i)<0) && (1.5*xy(1,i)-0.75-xy(2,i)<0)
        domain(1,k2)=xy(1,i);
        domain(2,k2)=xy(2,i);
        k2=k2+1;
    end
end

interiork=1;
for kk=1:N
    if domain(1,kk)~=0 && domain(2,kk)~=0
        interiorD(1,interiork)=domain(1,kk);
        interiorD(2,interiork)=domain(2,kk);
        interiork=interiork+1;
    end
end
u = sin(pi*interiorD(1,:)).*sin(pi*interiorD(2,:));

%增加边界点
M = 50;
xbt1 = linspace(0,1.0/3, M);
ybt1 = ones(1,M)*0.75;
kbt=k2;
for i=1:M
    domain(1,kbt)=xbt1(i);
    domain(2,kbt)=ybt1(i);
    kbt=kbt+1;
end

xbt2 = linspace(1.0/3,0.5, M);
ybt2 = 1.5*xbt2 + 0.25;
for i=1:M
    domain(1,kbt)=xbt2(i);
    domain(2,kbt)=ybt2(i);
    kbt=kbt+1;
end

xbt3 = linspace(0.5,2.0/3, M);
ybt3 = (-1.5)*xbt3 + 1.75;
for i=1:M
    domain(1,kbt)=xbt3(i);
    domain(2,kbt)=ybt3(i);
    kbt=kbt+1;
end

xbr1 = linspace(2.0/3,1.0, M);
ybr1 = ones(1,M)*0.75;
kbr=kbt;
for i=1:M
    domain(1,kbr)=xbr1(i);
    domain(2,kbr)=ybr1(i);
    kbr=kbr+1;
end

xbr2 = linspace(5.0/6,1.0, M);
ybr2 = 1.5*xbr2 -0.75;
for i=1:M
    domain(1,kbr)=xbr2(i);
    domain(2,kbr)=ybr2(i);
    kbr=kbr+1;
end

xbr3 = linspace(5.0/6,1.0, M);
ybr3 = (-1.5)*xbr3 + 1.75;
for i=1:M
    domain(1,kbr)=xbr3(i);
    domain(2,kbr)=ybr3(i);
    kbr=kbr+1;
end

xbb1 = linspace(2.0/3,1.0, M);
ybb1 = ones(1,M)*0.25;
kbb=kbr;
for i=1:M
    domain(1,kbb)=xbb1(i);
    domain(2,kbb)=ybb1(i);
    kbb=kbb+1;
end

xbb2 = linspace(0.5,2.0/3, M);
ybb2 = 1.5*xbb2 -0.75;
for i=1:M
    domain(1,kbb)=xbb2(i);
    domain(2,kbb)=ybb2(i);
    kbb=kbb+1;
end

xbb3 = linspace(1.0/3,0.5, M);
ybb3 = (-1.5)*xbb3 + 0.75;
for i=1:M
    domain(1,kbb)=xbb3(i);
    domain(2,kbb)=ybb3(i);
    kbb=kbb+1;
end

xbb1 = linspace(2.0/3,1.0, M);
ybb1 = ones(1,M)*0.25;
kbb=kbr;
for i=1:M
    domain(1,kbb)=xbb1(i);
    domain(2,kbb)=ybb1(i);
    kbb=kbb+1;
end

xbb2 = linspace(0.5,2.0/3, M);
ybb2 = 1.5*xbb2 -0.75;
for i=1:M
    domain(1,kbb)=xbb2(i);
    domain(2,kbb)=ybb2(i);
    kbb=kbb+1;
end

xbb3 = linspace(1.0/3,0.5, M);
ybb3 = (-1.5)*xbb3 + 0.75;
for i=1:M
    domain(1,kbb)=xbb3(i);
    domain(2,kbb)=ybb3(i);
    kbb=kbb+1;
end

xbL1 = linspace(0,1.0/3, M);
ybL1 = ones(1,M)*0.25;
kbL=kbb;
for i=1:M
    domain(1,kbL)=xbL1(i);
    domain(2,kbL)=ybL1(i);
    kbL=kbL+1;
end

xbL2 = linspace(0.0,1.0/6, M);
ybL2 = 1.5*xbL2 + 0.25;
for i=1:M
    domain(1,kbL)=xbL2(i);
    domain(2,kbL)=ybL2(i);
    kbL=kbL+1;
end

xbL3 = linspace(0.0,1.0/6, M);
ybL3 = (-1.5)*xbL3 + 0.75;
for i=1:M
    domain(1,kbL)=xbL3(i);
    domain(2,kbL)=ybL3(i);
    kbL=kbL+1;
end

irk=1;
for kk=1:N
    if domain(1,kk)~=0 || domain(2,kk)~=0
        irregularD(1,irk)=domain(1,kk);
        irregularD(2,irk)=domain(2,kk);
        irk=irk+1;
    end
end
figure('name','domian')
scatter(irregularD(1,:),irregularD(2,:),'r.')

figure('name','u')
ling = zeros(1, 12*M);
U = [u,ling];
size2U = size(U);
c = linspace(0.1,1,size2U(2));
scatter3(irregularD(1,:),irregularD(2,:),U,c,'filled');
grid on

XY = irregularD;
save('testXY.mat','XY')

% 判断边界点