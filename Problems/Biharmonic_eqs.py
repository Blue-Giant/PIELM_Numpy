import torch
import numpy as np


def get_biharmonic_infos_1D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'equation1':
        # u=sin(pi*x)
        f_side = lambda x: ((np.pi)**4)*torch.sin(np.pi*x)

        u_true = lambda x: torch.sin(np.pi*x)

        ux_left = lambda x: torch.sin(np.pi*left_bottom)
        ux_right = lambda x: torch.sin(np.pi*right_top)
        return f_side, u_true, ux_left, ux_right

    elif equa_name == 'equation2':
        # u=10.0*x*(1-2*x*x+x**3)
        f_side = lambda x: ((np.pi)**4)*torch.sin(np.pi*x)

        u_true = lambda x: 10.0*x*(1-2*x*x+x**3)

        ux_left = lambda x: 10.0*left_bottom*(1-2*left_bottom*left_bottom+left_bottom**3)
        ux_right = lambda x: torch.sin(np.pi*right_top)
        return f_side, u_true, ux_left, ux_right


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_Dirichlet_2D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'PDE1':
        # u=(sin(pi*x)*sin(pi*y))^2
        f_side = lambda x, y: 8*(np.pi)**4*np.cos(2*np.pi*x)*(np.cos(2*np.pi*y)) - \
                              8*(np.pi)**4*np.cos(2*np.pi*x)*(np.sin(np.pi*y))**2 - \
                              8*(np.pi)**4*np.cos(2*np.pi*y)*(np.sin(np.pi*x))**2

        u_true = lambda x, y: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))

        u_left = lambda x, y: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))
        u_right = lambda x, y: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))
        u_bottom = lambda x, y: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))
        u_top = lambda x, y: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))

        ux_left = lambda x, y: -2.0*np.pi*np.sin(np.pi*x)*np.cos(np.pi*x)*np.square(np.sin(np.pi*y))
        ux_right = lambda x, y: 2.0*np.pi*np.sin(np.pi*x)*np.cos(np.pi*x)*np.square(np.sin(np.pi*y))
        uy_bottom = lambda x, y: -2.0*np.square(np.sin(np.pi*x))*np.pi*np.sin(np.pi*y)*np.cos(np.pi*y)
        uy_top = lambda x, y: 2.0*np.square(np.sin(np.pi*x))*np.pi*np.sin(np.pi*y)*np.cos(np.pi*y)
        return f_side, u_true, u_left, u_right, u_bottom, u_top, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE1_1':
        # u=0.25*[1-cos(2pix)]0.25*[1-cos(2piy)]
        f_side = lambda x, y: 8*((np.pi)**4)*np.cos(2*np.pi*x)*np.cos(2*np.pi*y) - \
                              4*((np.pi)**4)*np.cos(2*np.pi*x)*(1-np.cos(2*np.pi*y)) - \
                              4*((np.pi)**4)*np.cos(2*np.pi*y)*(1-np.cos(2*np.pi*x))

        u_true = lambda x, y: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))

        u_left = lambda x, y: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))
        u_right = lambda x, y: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))
        u_bottom = lambda x, y: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))
        u_top = lambda x, y: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))

        ux_left = lambda x, y: -0.5 * np.pi*np.sin(2 * np.pi * x) * (1 - np.cos(2 * np.pi * y))
        ux_right = lambda x, y: 0.5 * np.pi*np.sin(2 * np.pi * x) * (1 - np.cos(2 * np.pi * y))
        uy_bottom = lambda x, y: -0.5 * (1 - np.cos(2 * np.pi * x)) * np.pi * np.sin(2 * np.pi * y)
        uy_top = lambda x, y: 0.5 * (1 - np.cos(2 * np.pi * x)) * np.pi * np.sin(2 * np.pi * y)
        return f_side, u_true, u_left, u_right, u_bottom, u_top, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE2':
        # u = 200*x^2*y^2*(1-x)^2*(1-y)^2  %未有1000的时候,真解的值太小，要扩大1000倍
        f_side = lambda x, y: 1600*(1-6*x+6*x*x)*(1-6*y+6*y*y) + 4800*(y*(1-y))**2 + 4800*(x*(1-x))**2
        u_true = lambda x, y: 200*torch.square(x)*torch.square(y)*torch.square(1-x)*torch.square(1-y)

        ux_left = lambda x, y: 200*torch.square(x)*torch.square(y)*torch.square(1-x)*torch.square(1-y)
        ux_right = lambda x, y: 200*torch.square(x)*torch.square(y)*torch.square(1-x)*torch.square(1-y)
        uy_bottom = lambda x, y: 200*torch.square(x)*torch.square(y)*torch.square(1-x)*torch.square(1-y)
        uy_top = lambda x, y: 200*torch.square(x)*torch.square(y)*torch.square(1-x)*torch.square(1-y)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE3':
        # u=10*(sin(pi*x))^2*(y-y^2)^2
        f_side = lambda x, y: -80.0*((np.pi)**4)*torch.cos(2*np.pi*x)*(y-torch.square(y))*(y-torch.square(y)) + \
                              80*((np.pi)**2)*torch.cos(2*np.pi*x)*(6*torch.square(y)-6*y+1) + 240.0*torch.sin(np.pi*x)*torch.sin(np.pi*x)

        u_true = lambda x, y: 10.0*torch.sin(np.pi*x)*torch.sin(np.pi*x)*(y-torch.square(y))*(y-torch.square(y))

        ux_left = lambda x, y: 10.0*torch.sin(np.pi*x)*torch.sin(np.pi*x)*(y-torch.square(y))*(y-torch.square(y))
        ux_right = lambda x, y: 10.0*torch.sin(np.pi*x)*torch.sin(np.pi*x)*(y-torch.square(y))*(y-torch.square(y))
        uy_bottom = lambda x, y: 10.0*torch.sin(np.pi*x)*torch.sin(np.pi*x)*(y-torch.square(y))*(y-torch.square(y))
        uy_top = lambda x, y: 10.0*torch.sin(np.pi*x)*torch.sin(np.pi*x)*(y-torch.square(y))*(y-torch.square(y))
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE3_1':
        # u=5*(1-cos(2*pi*x))*(y-y^2)^2
        f_side = lambda x, y: -80.0*((np.pi)**4)*torch.cos(2*np.pi*x)*torch.square(y)*torch.square(1.0-y) + \
                              80.0*((np.pi)**2)*torch.cos(2*np.pi*x)*(6*torch.square(y)-6*y+1.0) + 120.0*(1.0-torch.cos(2*np.pi*x))

        u_true = lambda x, y: 5.0*(1.0-torch.cos(2*np.pi*x))*torch.square(y)*torch.square(1.0-y)

        ux_left = lambda x, y: 5.0*(1.0-torch.cos(2*np.pi*x))*torch.square(y)*torch.square(1.0-y)
        ux_right = lambda x, y: 5.0*(1.0-torch.cos(2*np.pi*x))*torch.square(y)*torch.square(1.0-y)
        uy_bottom = lambda x, y: 5.0*(1.0-torch.cos(2*np.pi*x))*torch.square(y)*torch.square(1.0-y)
        uy_top = lambda x, y: 5.0*(1.0-torch.cos(2*np.pi*x))*torch.square(y)*torch.square(1.0-y)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_Navier_2D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'PDE4':
        # u=sin(pi*x)*sin(pi*y), f =4*pi^4*sin(pi*x)*sin(pi*y)
        f_side = lambda x, y: 4*((np.pi)**4)*np.sin(np.pi*x)*np.sin(np.pi*y)

        u_true = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)

        u_left = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)
        u_right = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)
        u_bottom = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)
        u_top = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)

        ux_left = lambda x, y: -2.0*np.pi*np.pi*np.sin(np.pi * x) * np.sin(np.pi * y)
        ux_right = lambda x, y: -2.0*np.pi*np.pi*np.sin(np.pi * x) * np.sin(np.pi * y)
        uy_bottom = lambda x, y: -2.0*np.pi*np.pi*np.sin(np.pi * x) * np.sin(np.pi * y)
        uy_top = lambda x, y: -2.0*np.pi*np.pi*np.sin(np.pi * x) * np.sin(np.pi * y)
        return f_side, u_true, u_left, u_right, u_bottom, u_top, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE5':
        # u=10x(1-2x^2+x^3)y(1-2y^2+y^3)
        f_side = lambda x, y: 240.0*y*(1-2*y*y+y**3) + 2880.0*x*(x-1)*y*(y-1) + 240.0*x*(1-2*x*x+x**3)

        u_true = lambda x, y: 10.0*x*(1-2*x*x+x**3)*y*(1-2*y*y+y**3)

        u_left = lambda x, y: 10.0 * x * (1 - 2 * x * x + x ** 3) * y * (1 - 2 * y * y + y ** 3)
        u_right = lambda x, y: 10.0 * x * (1 - 2 * x * x + x ** 3) * y * (1 - 2 * y * y + y ** 3)
        u_bottom = lambda x, y: 10.0 * x * (1 - 2 * x * x + x ** 3) * y * (1 - 2 * y * y + y ** 3)
        u_top = lambda x, y: 10.0 * x * (1 - 2 * x * x + x ** 3) * y * (1 - 2 * y * y + y ** 3)

        ux_left = lambda x, y: 10.0*(-12*x+12*x*x)*y*(1-2*y*y+y**3) + 10.0*x*(1-2*x*x+x**3)*(-12*y+12*y*y)
        ux_right = lambda x, y: 10.0*(-12*x+12*x*x)*y*(1-2*y*y+y**3) + 10.0*x*(1-2*x*x+x**3)*(-12*y+12*y*y)
        uy_bottom = lambda x, y: 10.0*(-12*x+12*x*x)*y*(1-2*y*y+y**3) + 10.0*x*(1-2*x*x+x**3)*(-12*y+12*y*y)
        uy_top = lambda x, y: 10.0*(-12*x+12*x*x)*y*(1-2*y*y+y**3) + 10.0*x*(1-2*x*x+x**3)*(-12*y+12*y*y)
        return f_side, u_true, u_left, u_right, u_bottom, u_top, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE6':
        # u=2*sin(pi*x)y(1-2y^2+y^3)
        f_side = lambda x, y: 2.0*(np.pi**4)*np.sin(np.pi*x)*(y-2*(y**3)+y**4) - 48*(np.pi**2)*np.sin(np.pi*x)*(y**2-y) + 48*np.sin(np.pi*x)

        u_true = lambda x, y: 2.0*np.sin(np.pi*x)*(y-2*(y**3)+y**4)

        u_left = lambda x, y: 2.0 * np.sin(np.pi * x) * (y - 2 * (y ** 3) + y ** 4)
        u_right = lambda x, y: 2.0 * np.sin(np.pi * x) * (y - 2 * (y ** 3) + y ** 4)
        u_bottom = lambda x, y: 2.0 * np.sin(np.pi * x) * (y - 2 * (y ** 3) + y ** 4)
        u_top = lambda x, y: 2.0 * np.sin(np.pi * x) * (y - 2 * (y ** 3) + y ** 4)

        ux_left = lambda x, y: -2.0*np.pi*np.pi*np.sin(np.pi*x)*(y-2*(y**3)+y**4)
        ux_right = lambda x, y: 2.0*np.pi*np.pi*np.sin(np.pi*x)*(y-2*(y**3)+y**4)
        uy_bottom = lambda x, y: -2.0*np.sin(np.pi*x)*(-12*y+12*y*y)
        uy_top = lambda x, y: 2.0*np.sin(np.pi*x)*(-12*y+12*y*y)
        return f_side, u_true, u_left, u_right, u_bottom, u_top, ux_left, ux_right, uy_bottom, uy_top


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_infos_2D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'Dirichlet1':
        # u=(sin(pi*x)*sin(pi*y))^2
        f_side = lambda x, y: 8*(torch.pi)**4*torch.cos(2*torch.pi*x)*(torch.cos(2*torch.pi*y)) - \
                              8*(torch.pi)**4*torch.cos(2*torch.pi*x)*(torch.sin(torch.pi*y))**2 - \
                              8*(torch.pi)**4*torch.cos(2*torch.pi*y)*(torch.sin(torch.pi*x))**2

        u_true = lambda x, y: torch.square(torch.sin(torch.pi*x))*torch.square(torch.sin(torch.pi*y))

        ux_left = lambda x, y: torch.square(torch.sin(torch.pi*x))*torch.square(torch.sin(torch.pi*y))
        ux_right = lambda x, y: torch.square(torch.sin(torch.pi*x))*torch.square(torch.sin(torch.pi*y))
        uy_bottom = lambda x, y: torch.square(torch.sin(torch.pi*x))*torch.square(torch.sin(torch.pi*y))
        uy_top = lambda x, y: torch.square(torch.sin(torch.pi*x))*torch.square(torch.sin(torch.pi*y))
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'Dirichlet1_1':
        # u=0.25*[1-cos(2pi*x)]*[1-cos(2pi*y)]
        f_side = lambda x, y: 8*((torch.pi)**4)*torch.cos(2*torch.pi*x)*torch.cos(2*torch.pi*y) - \
                              4*((torch.pi)**4)*torch.cos(2*torch.pi*x)*(1-torch.cos(2*torch.pi*y)) - \
                              4*((torch.pi)**4)*torch.cos(2*torch.pi*y)*(1-torch.cos(2*torch.pi*x))

        u_true = lambda x, y: 0.25*(1-torch.cos(2*torch.pi*x))*(1-torch.cos(2*torch.pi*y))

        ux_left = lambda x, y: 0.25*(1-torch.cos(2*torch.pi*x))*(1-torch.cos(2*torch.pi*y))
        ux_right = lambda x, y: 0.25*(1-torch.cos(2*torch.pi*x))*(1-torch.cos(2*torch.pi*y))
        uy_bottom = lambda x, y: 0.25*(1-torch.cos(2*torch.pi*x))*(1-torch.cos(2*torch.pi*y))
        uy_top = lambda x, y: 0.25*(1-torch.cos(2*torch.pi*x))*(1-torch.cos(2*torch.pi*y))
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'Dirichlet2':
        # u = 200*x^2*y^2*(1-x)^2*(1-y)^2  %未有1000的时候,真解的值太小，要扩大1000倍
        f_side = lambda x, y: 1600*(1-6*x+6*x*x)*(1-6*y+6*y*y) + 4800*(y-y*y)*(y-y*y) + 4800*(x-x*x)*(x-x*x)
        u_true = lambda x, y: 200*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)

        ux_left = lambda x, y: 200*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)
        ux_right = lambda x, y: 200*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)
        uy_bottom = lambda x, y: 200*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)
        uy_top = lambda x, y: 200*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'Dirichlet3':
        # u=10*(sin(pi*x))^2*(y-y^2)^2
        f_side = lambda x, y: -80.0*((torch.pi)**4)*torch.cos(2*torch.pi*x)*(y-y*y)*(y-y*y) + \
                              80*((torch.pi)**2)*torch.cos(2*torch.pi*x)*(6*y*y-6*y+1) + \
                              240.0*torch.sin(torch.pi*x)*torch.sin(torch.pi*x)

        u_true = lambda x, y: 10.0*torch.sin(torch.pi*x)*torch.sin(torch.pi*x)*(y-y*y)*(y-y*y)

        ux_left = lambda x, y: 10.0*torch.sin(torch.pi*x)*torch.sin(torch.pi*x)*(y-y*y)*(y-y*y)
        ux_right = lambda x, y: 10.0*torch.sin(torch.pi*x)*torch.sin(torch.pi*x)*(y-y*y)*(y-y*y)
        uy_bottom = lambda x, y: 10.0*torch.sin(torch.pi*x)*torch.sin(torch.pi*x)*(y-y*y)*(y-y*y)
        uy_top = lambda x, y: 10.0*torch.sin(torch.pi*x)*torch.sin(torch.pi*x)*(y-y*y)*(y-y*y)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'Dirichlet3_1':
        # u=5*(1-cos(2*pi*x))*(y-y^2)^2
        f_side = lambda x, y: -80.0*((np.pi)**4)*torch.cos(2*np.pi*x)*torch.square(y)*torch.square(1.0-y) + \
                              80.0*((np.pi)**2)*torch.cos(2*np.pi*x)*(6*torch.square(y)-6*y+1.0) + 120.0*(1.0-torch.cos(2*np.pi*x))

        u_true = lambda x, y: 5.0*(1.0-torch.cos(2*torch.pi*x))*(y-y*y)

        ux_left = lambda x, y: 5.0*(1.0-torch.cos(2*torch.pi*x))*(y-y*y)
        ux_right = lambda x, y: 5.0*(1.0-torch.cos(2*torch.pi*x))*(y-y*y)
        uy_bottom = lambda x, y: 5.0*(1.0-torch.cos(2*torch.pi*x))*(y-y*y)
        uy_top = lambda x, y: 5.0*(1.0-torch.cos(2*torch.pi*x))*(y-y*y)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'Navier4':
        # u=sin(pi*x)*sin(pi*y), f =4*pi^4*sin(pi*x)*sin(pi*y)
        f_side = lambda x, y: 4*((torch.pi)**4)*torch.sin(torch.pi*x)*torch.sin(torch.pi*y)

        u_true = lambda x, y: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)

        ux_left = lambda x, y: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)
        ux_right = lambda x, y: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)
        uy_bottom = lambda x, y: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)
        uy_top = lambda x, y: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'Navier5':
        # u=10x(1-2x^2+x^30)y(1-2y^2+y^3)
        f_side = lambda x, y: 240.0*y*(1-2*y*y+y**3) + 2880.0*x*(x-1)*y*(y-1) + 240.0*x*(1-2*x*x+x**3)

        u_true = lambda x, y: 10.0*x*(1-2*x*x+x**3)*y*(1-2*y*y+y**3)

        ux_left = lambda x, y: 10.0*x*(1-2*x*x+x**3)*y*(1-2*y*y+y**3)
        ux_right = lambda x, y: 10.0*x*(1-2*x*x+x**3)*y*(1-2*y*y+y**3)
        uy_bottom = lambda x, y: 10.0*x*(1-2*x*x+x**3)*y*(1-2*y*y+y**3)
        uy_top = lambda x, y: 10.0*x*(1-2*x*x+x**3)*y*(1-2*y*y+y**3)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'Navier6':
        # u=2*sin(pi*x)y(1-2y^2+y^3)
        f_side = lambda x, y: 2.0*(torch.pi**4)*torch.sin(torch.pi*x)*(y-2*(y**3)+y**4) - \
                              48*(torch.pi**2)*torch.sin(torch.pi*x)*(y**2-y) + 48*torch.sin(torch.pi*x)

        u_true = lambda x, y: 2.0*torch.sin(torch.pi*x)*(y-2*(y**3)+y**4)

        ux_left = lambda x, y: 2.0*torch.sin(torch.pi*x)*(y-2*(y**3)+y**4)
        ux_right = lambda x, y: 2.0*torch.sin(torch.pi*x)*(y-2*(y**3)+y**4)
        uy_bottom = lambda x, y: 2.0*torch.sin(torch.pi*x)*(y-2*(y**3)+y**4)
        uy_top = lambda x, y: 2.0*torch.sin(torch.pi*x)*(y-2*(y**3)+y**4)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_Dirichlet_3D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if str.upper(equa_name) == 'PDE1_1':
        # u=0.25*[1-cos2(pi x)][1-cos2(pi y)][1-cos2(pi z)]
        f_side = lambda x, y, z: -4*((np.pi)**4)*np.cos(2*np.pi*x)*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))\
                                 - 4*((np.pi)**4)*np.cos(2*np.pi*y)*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*z))\
                                 - 4*((np.pi)**4)*np.cos(2*np.pi*z)*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))\
                                 + 8*((np.pi)**4)*np.cos(2*np.pi*x)*(np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))\
                                 + 8*((np.pi)**4)*np.cos(2*np.pi*x)*(np.cos(2*np.pi*z))*(1-np.cos(2*np.pi*y))\
                                 + 8*((np.pi)**4)*np.cos(2*np.pi*z)*(np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*x))

        u_true = lambda x, y, z: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))

        u_bottom = lambda x, y, z: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))
        u_top = lambda x, y, z: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))
        u_left = lambda x, y, z: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))
        u_right = lambda x, y, z: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))
        u_front = lambda x, y, z: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))
        u_behind = lambda x, y, z: 0.25*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))

        uxyz_bottom = lambda x, y, z: np.zeros_like(x)
        uxyz_top = lambda x, y, z: np.zeros_like(x)
        uxyz_left = lambda x, y, z: np.zeros_like(x)
        uxyz_right = lambda x, y, z: np.zeros_like(x)
        uxyz_front = lambda x, y, z: np.zeros_like(x)
        uxyz_behind = lambda x, y, z: np.zeros_like(x)
    elif str.upper(equa_name) == 'PDE1':
        # u=[sin(pi x)]^2[sin(pi y)]^2[sin(pi z)]^2
        f_side = lambda x, y, z: -8*((np.pi)**4)*np.cos(2*np.pi*x)*np.square(np.sin(np.pi*y))*np.square(np.sin(np.pi*z))\
                                 -8*((np.pi)**4)*np.cos(2*np.pi*y)*np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*z))\
                                 -8*((np.pi)**4)*np.cos(2*np.pi*z)*np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))\
                                 + 8*((np.pi)**4)*np.cos(2*np.pi*x)*(np.cos(2*np.pi*y))*np.square(np.sin(np.pi*z))\
                                 + 8*((np.pi)**4)*np.cos(2*np.pi*x)*(np.cos(2*np.pi*z))*np.square(np.sin(np.pi*y))\
                                 + 8*((np.pi)**4)*np.cos(2*np.pi*z)*(np.cos(2*np.pi*y))*np.square(np.sin(np.pi*x))

        u_true = lambda x, y, z: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))*np.square(np.sin(np.pi*z))

        u_bottom = lambda x, y, z: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))*np.square(np.sin(np.pi*z))
        u_top = lambda x, y, z: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))*np.square(np.sin(np.pi*z))
        u_left = lambda x, y, z: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))*np.square(np.sin(np.pi*z))
        u_right = lambda x, y, z: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))*np.square(np.sin(np.pi*z))
        u_front = lambda x, y, z: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))*np.square(np.sin(np.pi*z))
        u_behind = lambda x, y, z: np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))*np.square(np.sin(np.pi*z))

        uxyz_bottom = lambda x, y, z: -1.0*np.pi*np.sin(2*np.pi*z)*np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))
        uxyz_top = lambda x, y, z: np.pi*np.sin(2*np.pi*z)*np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*y))
        uxyz_left = lambda x, y, z: -1.0*np.pi*np.sin(2*np.pi*x)*np.square(np.sin(np.pi*y))*np.square(np.sin(np.pi*z))
        uxyz_right = lambda x, y, z: np.pi*np.sin(2*np.pi*x)*np.square(np.sin(np.pi*y))*np.square(np.sin(np.pi*z))
        uxyz_front = lambda x, y, z: np.pi*np.sin(2*np.pi*y)*np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*z))
        uxyz_behind = lambda x, y, z: -1.0*np.pi*np.sin(2*np.pi*y)*np.square(np.sin(np.pi*x))*np.square(np.sin(np.pi*z))
    elif str.upper(equa_name) == 'PDE2':
        # u = 200*x^2*y^2*(1-x)^2*(1-y)^2  %未有1000的时候,真解的值太小，要扩大1000倍
        f_side = lambda x, y: 1600 * (1 - 6 * x + 6 * x * x) * (1 - 6 * y + 6 * y * y) + 4800 * (y - y * y) * (
                    y - y * y) + 4800 * (x - x * x) * (x - x * x)
        u_true = lambda x, y, z: 1000*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z)

        u_left = lambda x, y, z: 1000*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z)
        u_right = lambda x, y, z: 1000*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z)
        u_behind = lambda x, y, z: 1000*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z)
        u_front = lambda x, y, z: 1000*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z)
        u_bottom = lambda x, y, z: 1000*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z)
        u_top = lambda x, y, z: 1000*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z)

        uxyz_bottom = lambda x, y, z: np.zeros_like(x)
        uxyz_top = lambda x, y, z: np.zeros_like(x)
        uxyz_left = lambda x, y, z: np.zeros_like(x)
        uxyz_right = lambda x, y, z: np.zeros_like(x)
        uxyz_front = lambda x, y, z: np.zeros_like(x)
        uxyz_behind = lambda x, y, z: np.zeros_like(x)

    return f_side, u_true, u_bottom, u_top, u_left, u_right, u_front, u_behind, uxyz_bottom, uxyz_top, uxyz_left, \
           uxyz_right, uxyz_front, uxyz_behind


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_Navier_3D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'Navier1':
        # u=sin(pi*x)*sin(pi*y)*sin(pi*z)
        f_side = lambda x, y, z: 9.0*((np.pi)**4)*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)

        u_true = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)

        u_bottom = lambda x, y, z: np.zeros_like(x)
        u_top = lambda x, y, z: np.zeros_like(x)
        u_left = lambda x, y, z: np.zeros_like(x)
        u_right = lambda x, y, z: np.zeros_like(x)
        u_front = lambda x, y, z: np.zeros_like(x)
        u_behind = lambda x, y, z: np.zeros_like(x)

        uxyz_bottom = lambda x, y, z: np.zeros_like(x)
        uxyz_top = lambda x, y, z: np.zeros_like(x)
        uxyz_left = lambda x, y, z: np.zeros_like(x)
        uxyz_right = lambda x, y, z: np.zeros_like(x)
        uxyz_front = lambda x, y, z: np.zeros_like(x)
        uxyz_behind = lambda x, y, z: np.zeros_like(x)
        return f_side, u_true, u_bottom, u_top, u_left, u_right, u_front, u_behind, uxyz_bottom, uxyz_top, uxyz_left, \
                uxyz_right, uxyz_front, uxyz_behind
    elif equa_name == 'Navier2':
        # 10x(1-2x^2+x^3)y(1-2y^2+y^3)z(1-2z^2+z^3)
        f_side = lambda x, y, z: 240.0 * y * (1 - 2 * y * y + y ** 3) * z * (1 - 2 * z * z + z ** 3) + 240.0 * x * (
                    1 - 2 * x * x + x ** 3) * z * (1 - 2 * z * z + z ** 3) + \
                                 + 240.0 * x * (1 - 2 * x * x + x ** 3) * y * (1 - 2 * y * y + y ** 3) + 2880.0 * x * (
                                             x - 1) * y * (y - 1) * z * (1 - 2 * z * z + z ** 3) \
                                 + 2880.0 * x * (x - 1) * z * (z - 1) * y * (1 - 2 * y * y + y ** 3) + 2880.0 * y * (
                                             y - 1) * z * (z - 1) * x * (1 - 2 * x * x + x ** 3)

        u_true = lambda x, y, z: 10.0 * x * (1 - 2 * x * x + x ** 3) * y * (1 - 2 * y * y + y ** 3) * z * (
                    1 - 2 * z * z + z ** 3)

        u_bottom = lambda x, y, z: np.zeros_like(x)
        u_top = lambda x, y, z: np.zeros_like(x)
        u_left = lambda x, y, z: np.zeros_like(x)
        u_right = lambda x, y, z: np.zeros_like(x)
        u_front = lambda x, y, z: np.zeros_like(x)
        u_behind = lambda x, y, z: np.zeros_like(x)

        uxyz_bottom = lambda x, y, z: np.zeros_like(x)
        uxyz_top = lambda x, y, z: np.zeros_like(x)
        uxyz_left = lambda x, y, z: np.zeros_like(x)
        uxyz_right = lambda x, y, z: np.zeros_like(x)
        uxyz_front = lambda x, y, z: np.zeros_like(x)
        uxyz_behind = lambda x, y, z: np.zeros_like(x)
        return f_side, u_true, u_bottom, u_top, u_left, u_right, u_front, u_behind, uxyz_bottom, uxyz_top, uxyz_left, \
                uxyz_right, uxyz_front, uxyz_behind


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_infos_3D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'Dirichlet_equation':
        # u=0.25*[1-cos2(pi x)][1-cos2(pi y)][1-cos2(pi z)]
        f_side = lambda x, y, z: -4*((np.pi)**4)*torch.cos(2*np.pi*x)*(1-torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*z))\
                                 - 4*((np.pi)**4)*torch.cos(2*np.pi*y)*(1-torch.cos(2*np.pi*x))*(1-torch.cos(2*np.pi*z))\
                                 - 4*((np.pi)**4)*torch.cos(2*np.pi*z)*(1-torch.cos(2*np.pi*x))*(1-torch.cos(2*np.pi*y))\
                                 + 8*((np.pi)**4)*torch.cos(2*np.pi*x)*(torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*z))\
                                 + 8*((np.pi)**4)*torch.cos(2*np.pi*x)*(torch.cos(2*np.pi*z))*(1-torch.cos(2*np.pi*y))\
                                 + 8*((np.pi)**4)*torch.cos(2*np.pi*z)*(torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*x))

        u_true = lambda x, y, z: 0.25*(1-torch.cos(2*np.pi*x))*(1-torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*z))

        uxyz_bottom = lambda x, y, z: np.zeros_like(x)
        uxyz_top = lambda x, y, z: 0
        uxyz_left = lambda x, y, z: 0
        uxyz_right = lambda x, y, z: 0
        uxyz_front = lambda x, y, z: 0
        uxyz_behind = lambda x, y, z: 0
        return f_side, u_true, uxyz_bottom, uxyz_top, uxyz_left, uxyz_right, uxyz_front, uxyz_behind

    elif equa_name == 'Navier_equation1':
        # u=sin(pi*x)*sin(pi*y)*sin(pi*z)
        f_side = lambda x, y, z: 9.0*((np.pi)**4)*torch.sin(np.pi*x)*torch.sin(np.pi*y)*torch.sin(np.pi*z)

        u_true = lambda x, y, z: torch.sin(np.pi*x)*torch.sin(np.pi*y)*torch.sin(np.pi*z)

        uxyz_bottom = lambda x, y, z: 0
        uxyz_top = lambda x, y, z: 0
        uxyz_left = lambda x, y, z: 0
        uxyz_right = lambda x, y, z: 0
        uxyz_front = lambda x, y, z: 0
        uxyz_behind = lambda x, y, z: 0
        return f_side, u_true, uxyz_bottom, uxyz_top, uxyz_left, uxyz_right, uxyz_front, uxyz_behind
    elif equa_name == 'Navier_equation2':
        # 10x(1-2x^2+x^3)y(1-2y^2+y^3)z(1-2z^2+z^3)
        f_side = lambda x, y, z: 240.0 * y * (1 - 2 * y * y + y ** 3) * z * (1 - 2 * z * z + z ** 3) + 240.0 * x * (
                    1 - 2 * x * x + x ** 3) * z * (1 - 2 * z * z + z ** 3) + \
                                 + 240.0 * x * (1 - 2 * x * x + x ** 3) * y * (1 - 2 * y * y + y ** 3) + 2880.0 * x * (
                                             x - 1) * y * (y - 1) * z * (1 - 2 * z * z + z ** 3) \
                                 + 2880.0 * x * (x - 1) * z * (z - 1) * y * (1 - 2 * y * y + y ** 3) + 2880.0 * y * (
                                             y - 1) * z * (z - 1) * x * (1 - 2 * x * x + x ** 3)

        u_true = lambda x, y, z: 10.0 * x * (1 - 2 * x * x + x ** 3) * y * (1 - 2 * y * y + y ** 3) * z * (
                    1 - 2 * z * z + z ** 3)

        uxyz_bottom = lambda x, y, z: 0
        uxyz_top = lambda x, y, z: 0
        uxyz_left = lambda x, y, z: 0
        uxyz_right = lambda x, y, z: 0
        uxyz_front = lambda x, y, z: 0
        uxyz_behind = lambda x, y, z: 0
        return f_side, u_true, uxyz_bottom, uxyz_top, uxyz_left, uxyz_right, uxyz_front, uxyz_behind


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_Dirichlet_4D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    # u=0.25*[1-cos2(pi x)][1-cos2(pi y)][1-cos2(pi z)]
    f_side = lambda x, y, z, s: -4*((np.pi)**4)*torch.cos(2*np.pi*x)*(1-torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*z))*(1-torch.cos(2*np.pi*s))\
                             - 4*((np.pi)**4)*torch.cos(2*np.pi*y)*(1-torch.cos(2*np.pi*x))*(1-torch.cos(2*np.pi*z))*(1-torch.cos(2*np.pi*s))\
                             - 4*((np.pi)**4)*torch.cos(2*np.pi*z)*(1-torch.cos(2*np.pi*x))*(1-torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*s))\
                             - 4*((np.pi)**4)*torch.cos(2*np.pi*s)*(1-torch.cos(2*np.pi*x))*(1-torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*z))\
                             + 8*((np.pi)**4)*torch.cos(2*np.pi*x)*(torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*z))*(1-torch.cos(2*np.pi*s))\
                             + 8*((np.pi)**4)*torch.cos(2*np.pi*x)*(torch.cos(2*np.pi*z))*(1-torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*s))\
                             + 8*((np.pi)**4)*torch.cos(2*np.pi*x)*(torch.cos(2*np.pi*s))*(1-torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*z))\
                             + 8*((np.pi)**4)*torch.cos(2*np.pi*y)*(torch.cos(2*np.pi*z))*(1-torch.cos(2*np.pi*x))*(1-torch.cos(2*np.pi*s))\
                             + 8*((np.pi)**4)*torch.cos(2*np.pi*y)*(torch.cos(2*np.pi*s))*(1-torch.cos(2*np.pi*x))*(1-torch.cos(2*np.pi*z))\
                             + 8*((np.pi)**4)*torch.cos(2*np.pi*z)*(torch.cos(2 * np.pi*s))*(1-torch.cos(2*np.pi*x))*(1-torch.cos(2*np.pi*y))

    u_true = lambda x, y, z, s: 0.25*(1-torch.cos(2*np.pi*x))*(1-torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*z))*(1-torch.cos(2*np.pi*s))

    return f_side, u_true


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_Navier_4D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    f_side = lambda x, y, z, s: 16*((np.pi)**4)*torch.sin(np.pi*x)*torch.sin(np.pi*y)*torch.sin(np.pi*z)*torch.sin(np.pi*s)

    u_true = lambda x, y, z, s: torch.sin(np.pi*x)*torch.sin(np.pi*y)*torch.sin(np.pi*z)*torch.sin(np.pi*s)

    return f_side, u_true


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_infos_4D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'Dirichlet_equation':
        # u=0.25*[1-cos2(pi x)][1-cos2(pi y)][1-cos2(pi z)]
        f_side = lambda x, y, z, s: -4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (1 - torch.cos(2 * np.pi * y)) * (
                    1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s)) \
                                    - 4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * y) * (1 - torch.cos(2 * np.pi * x)) * (
                                                1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s)) \
                                    - 4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * z) * (1 - torch.cos(2 * np.pi * x)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * s)) \
                                    - 4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * s) * (1 - torch.cos(2 * np.pi * x)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * z)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (torch.cos(2 * np.pi * y)) * (
                                                1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (torch.cos(2 * np.pi * z)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * s)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (torch.cos(2 * np.pi * s)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * z)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * y) * (torch.cos(2 * np.pi * z)) * (
                                                1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * s)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * y) * (torch.cos(2 * np.pi * s)) * (
                                                1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * z)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * z) * (torch.cos(2 * np.pi * s)) * (
                                                1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * y))

        u_true = lambda x, y, z, s: 0.25 * (1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * y)) * (
                    1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s))
        return f_side, u_true
    elif equa_name == 'Navier_equation':
        f_side = lambda x, y, z, s: 16 * ((np.pi) ** 4) * torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.sin(
            np.pi * z) * torch.sin(np.pi * s)

        u_true = lambda x, y, z, s: torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.sin(np.pi * z) * torch.sin(np.pi * s)
        return f_side, u_true


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_Dirichlet_5D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    f_side = lambda x, y, z, s, t: 25.0*((np.pi)**4)*torch.sin(np.pi*x)*torch.sin(np.pi*y)*torch.sin(np.pi*z)*torch.sin(np.pi*s)*torch.sin(np.pi*t)

    u_true = lambda x, y, z, s, t: 0.25*(1-torch.cos(2*np.pi*x))*(1-torch.cos(2*np.pi*y))*(1-torch.cos(2*np.pi*z))*(1-torch.cos(2*np.pi*s))

    return f_side, u_true


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_Navier_5D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'Navier_5D_1':
        f_side = lambda x, y, z, s, t: 25.0*((np.pi)**4)*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)*np.sin(np.pi*s)*np.sin(np.pi*t)

        u_true = lambda x, y, z, s, t: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)*np.sin(np.pi*s)*np.sin(np.pi*t)

        return f_side, u_true
    if equa_name == 'Navier_5D_2':
        f_side = lambda x, y, z, s, t: 50.0 * ((np.pi) ** 4) * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(
            np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)

        u_true = lambda x, y, z, s, t: 2.0*np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(
            np.pi * s) * np.sin(np.pi * t)

        return f_side, u_true


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_infos_5D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'Dirichlet_equation':
        # u=0.25*[1-cos2(pi x)][1-cos2(pi y)][1-cos2(pi z)]
        f_side = lambda x, y, z, s, t: -4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (1 - torch.cos(2 * np.pi * y)) * (
                    1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s)) \
                                    - 4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * y) * (1 - torch.cos(2 * np.pi * x)) * (
                                                1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s)) \
                                    - 4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * z) * (1 - torch.cos(2 * np.pi * x)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * s)) \
                                    - 4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * s) * (1 - torch.cos(2 * np.pi * x)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * z)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (torch.cos(2 * np.pi * y)) * (
                                                1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (torch.cos(2 * np.pi * z)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * s)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (torch.cos(2 * np.pi * s)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * z)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * y) * (torch.cos(2 * np.pi * z)) * (
                                                1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * s)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * y) * (torch.cos(2 * np.pi * s)) * (
                                                1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * z)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * z) * (torch.cos(2 * np.pi * s)) * (
                                                1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * y))

        u_true = lambda x, y, z, s, t: 0.25 * (1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * y)) * (
                    1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s)) * (1 - torch.cos(2 * np.pi * t))
        return f_side, u_true
    elif equa_name == 'Navier_equation':
        f_side = lambda x, y, z, s, t: 25.0 * ((np.pi) ** 4) * torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.sin(
            np.pi * z) * torch.sin(np.pi * s) * torch.sin(np.pi * t)

        u_true = lambda x, y, z, s, t: torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.sin(np.pi * z) * torch.sin(
            np.pi * s) * torch.sin(np.pi * t)
        return f_side, u_true


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数，其中一阶偏导数要满足为0
def get_biharmonic_infos_8D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'Dirichlet_equation':
        # u=0.25*[1-cos2(pi x)][1-cos2(pi y)][1-cos2(pi z)]
        f_side = lambda x, y, z, s, t: -4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (1 - torch.cos(2 * np.pi * y)) * (
                    1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s)) \
                                    - 4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * y) * (1 - torch.cos(2 * np.pi * x)) * (
                                                1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s)) \
                                    - 4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * z) * (1 - torch.cos(2 * np.pi * x)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * s)) \
                                    - 4 * ((np.pi) ** 4) * torch.cos(2 * np.pi * s) * (1 - torch.cos(2 * np.pi * x)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * z)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (torch.cos(2 * np.pi * y)) * (
                                                1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (torch.cos(2 * np.pi * z)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * s)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * x) * (torch.cos(2 * np.pi * s)) * (
                                                1 - torch.cos(2 * np.pi * y)) * (1 - torch.cos(2 * np.pi * z)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * y) * (torch.cos(2 * np.pi * z)) * (
                                                1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * s)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * y) * (torch.cos(2 * np.pi * s)) * (
                                                1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * z)) \
                                    + 8 * ((np.pi) ** 4) * torch.cos(2 * np.pi * z) * (torch.cos(2 * np.pi * s)) * (
                                                1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * y))

        u_true = lambda x, y, z, s, t: 0.25 * (1 - torch.cos(2 * np.pi * x)) * (1 - torch.cos(2 * np.pi * y)) * (
                    1 - torch.cos(2 * np.pi * z)) * (1 - torch.cos(2 * np.pi * s)) * (1 - torch.cos(2 * np.pi * t))
        return f_side, u_true
    elif equa_name == 'Navier_equation':
        f_side = lambda x, y, z, s, t: 25.0 * ((np.pi) ** 4) * torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.sin(
            np.pi * z) * torch.sin(np.pi * s) * torch.sin(np.pi * t)

        u_true = lambda x, y, z, s, t: torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.sin(np.pi * z) * torch.sin(
            np.pi * s) * torch.sin(np.pi * t)
        return f_side, u_true