import os
import sys
import platform
import shutil

import torch
import numpy as np

torch.set_default_dtype(torch.float64)
import time
import datetime
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from Networks import ELM_Base
from utilizers import DNN_tools
from utilizers import plot_data
from utilizers import saveData
from utilizers import RFN_Log_Print
from utilizers import dataUtilizer2numpy
from Problems import General_Laplace


# Global random feature network
class GELM2Poisson(object):
    def __init__(self, indim=1, outdim=1, num2hidden=None, name2Model='DNN', actName2hidden='tanh', actName2Deri='tanh',
                 actName2DDeri='tanh', type2float='float32', opt2init_W='xavier_normal',  opt2init_B='xavier_uniform',
                 W_sigma=1.0, B_sigma=1.0, freq=None, repeatHighFreq=False):
        super(GELM2Poisson, self).__init__()

        self.indim = indim
        self.outdim = outdim
        self.num2hidden = num2hidden
        self.name2Model = name2Model
        self.actName2hidden = actName2hidden
        self.actName2derivative = actName2Deri
        self.actName2DDeri = actName2DDeri

        # The ELM solver for solving PDEs
        if str.upper(name2Model) == 'PIELM':
            self.ELM_Bases = ELM_Base.PIELM(
                dim2in=indim, dim2out=outdim, num2hidden_units=num2hidden, name2Model='DNN', actName2hidden=actName2hidden,
                actName2Deri=actName2Deri, actName2DDeri=actName2DDeri, type2float=type2float,
                opt2init_hiddenW=opt2init_W, opt2init_hiddenB=opt2init_B, sigma2W=W_sigma, sigma2B=B_sigma)
        elif str.upper(name2Model) == 'MULTI_SCALE_FOURIER_ELM':
            self.ELM_Bases = ELM_Base.MultiscaleFourierPIELM(
                dim2in=indim, dim2out=outdim, num2hidden_units=num2hidden, name2Model='DNN', actName2hidden=actName2hidden,
                actName2Deri=actName2Deri, actName2DDeri=actName2DDeri, type2float=type2float, scale=freq,
                opt2init_hiddenW=opt2init_W, opt2init_hiddenB=opt2init_B, sigma2W=W_sigma, sigma2B=B_sigma,
                repeat_Highfreq=repeatHighFreq)
        elif str.upper(name2Model) == 'MULTI_4FF_ELM':
            self.ELM_Bases = ELM_Base.Multi4FF_PIELM(
                dim2in=indim, dim2out=outdim, num2hidden_units=num2hidden, name2Model='DNN', actName2hidden=actName2hidden,
                actName2Deri=actName2Deri, actName2DDeri=actName2DDeri, type2float=type2float,
                opt2init_hiddenW=opt2init_W, opt2init_hiddenB=opt2init_B, sigma_W1=1.0, sigma_W2=5.0, sigma_W3=10.0,
                sigma_W4=15.0, sigma_B1=1.0, sigma_B2=5.0, sigma_B3=10.0, sigma_B4=15.0)
        else:
            self.ELM_Bases = ELM_Base.FourierFeaturePIELM(
                dim2in=indim, dim2out=outdim, num2hidden_units=num2hidden, name2Model='DNN', actName2hidden=actName2hidden,
                actName2Deri=actName2Deri, actName2DDeri=actName2DDeri, type2float=type2float,
                opt2init_hiddenW=opt2init_W, opt2init_hiddenB=opt2init_B, sigma2W=W_sigma, sigma2B=B_sigma)

        if type2float == 'float32':
            self.float_type = np.float32
        elif type2float == 'float64':
            self.float_type = np.float64
        elif type2float == 'float16':
            self.float_type = np.float16

    # Assembling the matrix A,f in inner domain
    def assemble_matrix2inner(self, XY_inner=None, fside=None, if_lambda2fside=True):
        shape2XY = XY_inner.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        A = self.ELM_Bases.assemble_matrix2Laplace_2D(XY_input=XY_inner)

        X = np.reshape(XY_inner[:, 0], newshape=[-1, 1])
        Y = np.reshape(XY_inner[:, 1], newshape=[-1, 1])

        if if_lambda2fside:
            f = fside(X, Y)
        else:
            f = fside

        return (A, f)

    # Assembling the matrix B,g in boundary domain
    def assemble_matrix2boundary(self, XY_boundary=None, gside=None, if_lambda2gside=True):
        shape2XY = XY_boundary.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        B = self.ELM_Bases.assemble_matrix2boundary_2D(XY_input=XY_boundary)

        X = np.reshape(XY_boundary[:, 0], newshape=[-1, 1])
        Y = np.reshape(XY_boundary[:, 1], newshape=[-1, 1])

        if if_lambda2gside:
            g = gside(X, Y)
        else:
            g = gside

        return (B, g)

    def test_model(self, points=None):
        ELM_Bases = self.ELM_Bases.assemble_matrix2interior_2D(XY_input=points)
        return ELM_Bases


def solve_possion(Rdic=None):
    log_out_path = Rdic['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s.txt' % ('log')
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    RFN_Log_Print.dictionary_out2file(Rdic, log_fileout)

    time_begin = time.time()

    left = 0.0
    right = 1.0

    if Rdic['Equa_name'] == 'PDE2':
        right = 1.0
    elif Rdic['Equa_name'] == 'PDE3':
        right = 1.0
    elif Rdic['Equa_name'] == 'PDE4':
        right = 1.0
    Model = GELM2Poisson(
        indim=Rdic['input_dim'], outdim=Rdic['out_dim'], num2hidden=Rdic['rfn_hidden'], name2Model=Rdic['name2model'],
        actName2hidden=Rdic['act_name'], actName2Deri=Rdic['act_name'], actName2DDeri=Rdic['act_name'],
        type2float='float64', opt2init_W=Rdic['opt2initW'], opt2init_B=Rdic['opt2initB'], W_sigma=Rdic['sigma'],
        B_sigma=Rdic['sigma'])

    type2float = np.float64
    # points = dataUtilizer2numpy.gene_2Drand_points2inner(
    #     num2point=Rdic['point_num2inner'], variable_dim=Rdic['input_dim'], region_l=left, region_r=right,
    #     region_b=left, region_t=right, to_float=True, float_type=type2float, opt2rand='uniform', shuffle_point=True)

    if Rdic['model2generate_data'] == 'load_porous_data':
        points = dataUtilizer2numpy.load_data2porous_domain(
            region_left=left, region_right=right, region_bottom=left, region_top=right, float_type=type2float)
        saveData.save_testData_or_solus2mat(points, 'testxy', Rdic['FolderName'])
    else:
        points, mesh = dataUtilizer2numpy.gene_mesh_gene_rand_2Dinner(
            num2mesh=50, num2point=Rdic['point_num2inner'], variable_dim=Rdic['input_dim'], region_l=left, region_r=right,
            region_b=left, region_t=right, to_float=True, float_type=type2float, opt2rand='uniform', shuffle_point=True)
        saveData.save_testData_or_solus2mat(mesh, 'testxy', Rdic['FolderName'])

    left_bd, right_bd, bottom_bd, top_bd = dataUtilizer2numpy.gene_2Drand_points2bd(
        num2point=Rdic['point_num2boundary'], variable_dim=Rdic['input_dim'], region_l=left, region_r=right,
        region_b=left, region_t=right, to_float=True, float_type=type2float, opt2rand='random', shuffle_uniform=True)

    f_side, u_true, ux_left, ux_right, uy_bottom, uy_top = General_Laplace.get_Laplace2D_infos(equa_name=Rdic['Equa_name'])

    A_I, f_i = Model.assemble_matrix2inner(XY_inner=points, fside=f_side, if_lambda2fside=True)

    B_l, gl = Model.assemble_matrix2boundary(XY_boundary=left_bd, gside=ux_left)

    B_r, gr = Model.assemble_matrix2boundary(XY_boundary=right_bd, gside=ux_right)

    B_b, gb = Model.assemble_matrix2boundary(XY_boundary=bottom_bd, gside=uy_bottom)

    B_t, gt = Model.assemble_matrix2boundary(XY_boundary=top_bd, gside=uy_top)

    # num2inner = Rdic['point_num2inner']
    num2inner = points.shape[0]
    num2boundary = Rdic['point_num2boundary']
    rfn_hidden = Rdic['rfn_hidden']

    if str.upper(Rdic['name2model']) == 'MULTI_4FF_ELM':
        A = np.zeros([num2inner + 4 * num2boundary, 4 * rfn_hidden])
    else:
        A = np.zeros([num2inner + 4 * num2boundary, rfn_hidden])

    F = np.zeros([num2inner + 4 * num2boundary, 1])

    A[0:num2inner, :] = A_I

    A[num2inner:num2inner + num2boundary, :] = B_l
    A[num2inner + num2boundary:num2inner + 2 * num2boundary, :] = B_r
    A[num2inner + 2 * num2boundary:num2inner + 3 * num2boundary, :] = B_b
    A[num2inner + 3 * num2boundary:num2inner + 4 * num2boundary, :] = B_t

    F[0:num2inner, :] = f_i

    F[num2inner:num2inner + num2boundary, :] = gl
    F[num2inner + num2boundary:num2inner + 2 * num2boundary, :] = gr
    F[num2inner + 2 * num2boundary:num2inner + 3 * num2boundary, :] = gb
    F[num2inner + 3 * num2boundary:num2inner + 4 * num2boundary, :] = gt

    # # rescaling
    c = 100.0

    for i in range(len(A)):
        ratio = c / A[i, :].max()
        A[i, :] = A[i, :] * ratio
        F[i] = F[i] * ratio
    # solve
    w = lstsq(A, F)[0]

    if Rdic['model2generate_data'] == 'load_porous_data':
        temp = Model.test_model(points=points)
    else:
        temp = Model.test_model(points=mesh)
    numeri_solu = np.matmul(temp, w)

    time_end = time.time()
    run_time = time_end - time_begin

    if Rdic['model2generate_data'] == 'load_porous_data':
        exact_solu = u_true(np.reshape(points[:, 0], newshape=[-1, 1]), np.reshape(points[:, 1], newshape=[-1, 1]))
    else:
        exact_solu = u_true(np.reshape(mesh[:, 0], newshape=[-1, 1]), np.reshape(mesh[:, 1], newshape=[-1, 1]))

    abs_diff = np.abs(exact_solu - numeri_solu)

    max_diff = np.max(abs_diff)

    print('max_diff:', max_diff)

    rel_err = np.sqrt(np.sum(np.square(exact_solu - numeri_solu), axis=0) / np.sum(np.square(exact_solu), axis=0))

    print('relative error:', rel_err[0])

    print('running time:', run_time)

    DNN_tools.log_string('The max absolute error: %.18f\n' % max_diff, log_fileout)
    DNN_tools.log_string('The relative error: %.18f\n' % rel_err[0], log_fileout)
    DNN_tools.log_string('The running time: %.18f s\n' % run_time, log_fileout)

    if Rdic['model2generate_data'] == 'load_porous_data':
        plot_data.plot_scatter_solu(abs_diff, points, color='m', actName='diff', outPath=FolderName)
        plot_data.plot_scatter_solu(exact_solu, points, color='r', actName='exact', outPath=FolderName)
        plot_data.plot_scatter_solu(numeri_solu, points, color='b', actName='numerical', outPath=FolderName)
    else:
        plot_data.plot_scatter_solu(abs_diff, mesh, color='m', actName='diff', outPath=FolderName)
        plot_data.plot_scatter_solu(exact_solu, mesh, color='r', actName='exact', outPath=FolderName)
        plot_data.plot_scatter_solu(numeri_solu, mesh, color='b', actName='numerical', outPath=FolderName)

    saveData.save_testData_or_solus2mat(exact_solu, dataName='true', outPath=FolderName)
    saveData.save_testData_or_solus2mat(numeri_solu, dataName='numeri', outPath=FolderName)
    saveData.save_testData_or_solus2mat(abs_diff, dataName='perr', outPath=FolderName)

    saveData.save_run_time2mat(run_time=run_time, num2hidden=Rdic['rfn_hidden'], outPath=FolderName)
    saveData.save_max_rel2mat(max_err=max_diff, rel_err=rel_err, num2hidden=Rdic['rfn_hidden'], outPath=FolderName)


if __name__ == "__main__":
    R = {}
    # 文件保存路径设置
    file2results = 'Results'
    store_file = 'Laplace2D'
    BASE_DIR2FILE = os.path.dirname(os.path.abspath(__file__))
    split_BASE_DIR2FILE = os.path.split(BASE_DIR2FILE)
    split_BASE_DIR2FILE = os.path.split(split_BASE_DIR2FILE[0])
    BASE_DIR = split_BASE_DIR2FILE[0]
    sys.path.append(BASE_DIR)
    OUT_DIR_BASE = os.path.join(BASE_DIR, file2results)
    OUT_DIR = os.path.join(OUT_DIR_BASE, store_file)
    sys.path.append(OUT_DIR)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    current_day_time = datetime.datetime.now()  # 获取当前时间
    date_time_dir = str(current_day_time.month) + str('m_') + \
                    str(current_day_time.day) + str('d_') + str(current_day_time.hour) + str('h_') + \
                    str(current_day_time.minute) + str('m_') + str(current_day_time.second) + str('s')
    FolderName = os.path.join(OUT_DIR, date_time_dir)  # 路径连接
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    R['FolderName'] = FolderName

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    R['model2generate_data'] = 'load_porous_data'
    # R['model2generate_data'] = 'generate_mesh_data'

    R['PDE_type'] = 'Laplace'
    # R['Equa_name'] = 'PDE1'
    R['Equa_name'] = 'PDE2'
    # R['Equa_name'] = 'PDE3'
    # R['Equa_name'] = 'PDE4'

    R['point_num2inner'] = 5000
    R['point_num2boundary'] = 15
    R['rfn_hidden'] = 1000
    R['input_dim'] = 2
    R['out_dim'] = 1

    # R['name2model'] = 'PIELM'
    R['name2model'] = 'FF_PIELM'
    # R['name2model'] = 'Multi_4FF_ELM'

    if R['name2model'] == 'Multi_4FF_ELM':
        # R['rfn_hidden'] = 200
        R['rfn_hidden'] = 250
        # R['rfn_hidden'] = 300
        # R['rfn_hidden'] = 500

    # R['sigma'] = 0.25
    # R['sigma'] = 0.5
    # R['sigma'] = 0.75
    # R['sigma'] = 1.0
    # R['sigma'] = 1.5
    # R['sigma'] = 2.0
    # R['sigma'] = 2.5
    # R['sigma'] = 3.0
    # R['sigma'] = 6
    # R['sigma'] = 8
    # R['sigma'] = 10
    # R['sigma'] = 15
    # R['sigma'] = 20
    # R['sigma'] = 25
    # R['sigma'] = 30
    R['sigma'] = 35

    if R['name2model'] == 'Multi_4FF_ELM' or R['name2model'] == 'FF_PIELM':
        R['act_name'] = 'fourier'
    else:
        # R['act_name'] = 'tanh'
        R['act_name'] = 'sin'
        # R['act_name'] = 'gauss'
        # R['act_name'] = 'sinAddcos'

    # R['opt2initW'] = 'normal'
    R['opt2initW'] = 'scale_uniform11'

    # R['opt2initB'] = 'normal'
    # R['opt2initB'] = 'uniform11'
    R['opt2initB'] = 'scale_uniform11'
    solve_possion(Rdic=R)
