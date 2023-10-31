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
from Problems import General_Laplace
from utilizers import dataUtilizer2numpy


# Global random feature network
class GELM2Possion(object):
    def __init__(self, indim=1, outdim=1, num2hidden=None, name2Model='DNN', actName2hidden='tanh', actName2Deri='tanh',
                 actName2DDeri='tanh', type2float='float32', opt2init_W='xavier_normal',  opt2init_B='xavier_uniform',
                 W_sigma=1.0, B_sigma=1.0):
        super(GELM2Possion, self).__init__()

        self.indim = indim
        self.outdim = outdim
        self.num2hidden = num2hidden
        self.name2Model = name2Model
        self.actName2hidden = actName2hidden
        self.actName2derivative = actName2Deri
        self.actName2DDeri = actName2DDeri
        self.act_func2hidden = RFN_Base.my_actFunc(actName=actName2hidden)
        self.act_func2Deri = RFN_Base.ActFunc2Derivate(actName=actName2Deri)
        self.act_func2DDeri = RFN_Base.ActFunc2DDerivate(actName=actName2DDeri)

        self.ELM_Bases = RFN_Base.PIELM(
            dim2in=indim, dim2out=1, num2hidden_units=num2hidden, name2Model='DNN', actName2hidden=actName2hidden,
            actName2Deri=actName2Deri, actName2DDeri=actName2DDeri, type2float=type2float,  opt2init_hiddenW=opt2init_W,
            opt2init_hiddenB=opt2init_B, sigma2W=W_sigma, sigma2B=B_sigma)

        if type2float == 'float32':
            self.float_type = np.float32
        elif type2float == 'float64':
            self.float_type = np.float64
        elif type2float == 'float16':
            self.float_type = np.float16

    # Assembling the matrix A,f in inner domain
    def assemble_matrix2inner(self, X_inner=None,  fside=None, if_lambda2fside=True):
        A = self.ELM_Bases.assemble_matrix2DDerivative_1D(X_input=X_inner)

        shape2XY = X_inner.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 1)

        if if_lambda2fside:
            f = fside(X_inner)
        else:
            f = fside

        return (A, f)

    # Assembling the matrix B,g in boundary domain
    def assemble_matrix2boundary(self, X_bd=None, gside=None, if_lambda2gside=True):
        B = self.ELM_Bases.assemble_matrix2boundary_1D(X_input=X_bd)
        shape2XY = X_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 1)

        if if_lambda2gside:
            g = gside(X_bd)
        else:
            g = gside

        return (B, g)

    def test_rfn(self, points=None):
        ELM_Bases = self.ELM_Bases.get_out2Hidden(input_data=points)
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

    Model = GELM2Possion(indim=Rdic['input_dim'], outdim=Rdic['out_dim'], num2hidden=Rdic['rfn_hidden'],
                         name2Model='DNN', actName2hidden=Rdic['act_name'], type2float='float64', opt2init_W='normal',
                         opt2init_B='uniform', W_sigma=1.0, B_sigma=1.0)

    type2float = np.float64
    # points = dataUtilizer2numpy.gene_1Drand_points2inner(num2point=Rdic['point_num2inner'],
    #                                                      variable_dim=Rdic['input_dim'],
    #                                                      region_l=left, region_r=right, to_float=True,
    #                                                      float_type=type2float, opt2rand='random')

    points = dataUtilizer2numpy.gene_1Dmesh_points2inner(num2point=Rdic['point_num2inner'],
                                                         variable_dim=Rdic['input_dim'],
                                                         region_l=left, region_r=right, to_float=True,
                                                         float_type=type2float)

    left_bd, right_bd = dataUtilizer2numpy.gene_1Dinterval2bd(
        num2point=Rdic['point_num2boundary'], variable_dim=Rdic['input_dim'], region_l=left, region_r=right,
        to_float=True, float_type=type2float)

    f_side, u_true, ux_left, ux_right = General_Laplace.get_infos2Laplace_1D(equa_name=Rdic['Equa_name'])

    A_I, f_i = Model.assemble_matrix2inner(X_inner=points, fside=f_side)

    B_l, gl = Model.assemble_matrix2boundary(X_bd=left_bd, gside=ux_left)

    B_r, gr = Model.assemble_matrix2boundary(X_bd=right_bd, gside=ux_right)

    num2inner = Rdic['point_num2inner']
    num2boundary = Rdic['point_num2boundary']
    rfn_hidden = Rdic['rfn_hidden']

    A = np.zeros([num2inner + 2 * num2boundary, rfn_hidden])
    F = np.zeros([num2inner + 2 * num2boundary, 1])

    A[0:num2inner, :] = A_I

    A[num2inner:num2inner+num2boundary, :] = B_l
    A[num2inner+num2boundary:num2inner + 2 * num2boundary, :] = B_r

    F[0:num2inner, :] = f_i

    F[num2inner:num2inner + num2boundary, :] = gl
    F[num2inner + num2boundary:num2inner + 2 * num2boundary, :] = gr

    # # rescaling
    c = 100.0

    for i in range(len(A)):
        ratio = c / A[i, :].max()
        A[i, :] = A[i, :] * ratio
        F[i] = F[i] * ratio
    # solve
    w = lstsq(A, F)[0]

    temp = Model.test_rfn(points=points)
    numeri_solu = np.matmul(temp, w)

    time_end = time.time()
    run_time = time_end - time_begin

    exact_solu = u_true(points)

    abs_diff = np.abs(exact_solu - numeri_solu)

    max_diff = np.max(abs_diff)

    print('max_diff:', max_diff)

    rel_err = np.sqrt(np.sum(np.square(exact_solu - numeri_solu), axis=0) / np.sum(np.square(exact_solu), axis=0))

    print('relative error:', rel_err[0])

    print('running time:', run_time)

    DNN_tools.log_string('The max absolute error: %.18f\n' % max_diff, log_fileout)
    DNN_tools.log_string('The relative error: %.18f\n' % rel_err[0], log_fileout)
    DNN_tools.log_string('The running time: %.18f s\n' % run_time, log_fileout)

    plot_data.plot_solu_1D(solu=abs_diff, coords=points, color='m', actName='diff', outPath=FolderName)
    plot_data.plot_solu_1D(solu=exact_solu, coords=points, color='r', actName='exact', outPath=FolderName)
    plot_data.plot_solu_1D(solu=numeri_solu, coords=points, color='b', actName='numerical', outPath=FolderName)

    saveData.save_testData_or_solus2mat(exact_solu, dataName='true', outPath=FolderName)
    saveData.save_testData_or_solus2mat(numeri_solu, dataName='numeri', outPath=FolderName)
    saveData.save_testData_or_solus2mat(abs_diff, dataName='perr', outPath=FolderName)

    saveData.save_run_time2mat(run_time=run_time, num2hidden=Rdic['rfn_hidden'], outPath=FolderName)
    saveData.save_max_rel2mat(max_err=max_diff, rel_err=rel_err, num2hidden=Rdic['rfn_hidden'], outPath=FolderName)


if __name__ == "__main__":
    R = {}
    # 文件保存路径设置
    file2results = 'Results'
    store_file = 'Laplace1D'
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

    R['PDE_type'] = 'Laplace'
    R['Equa_name'] = 'PDE1'
    # R['Equa_name'] = 'PDE2'

    R['point_num2inner'] = 2000
    R['point_num2boundary'] = 5
    R['rfn_hidden'] = 200
    R['input_dim'] = 1
    R['out_dim'] = 1

    R['sigma'] = 2

    R['act_name'] = 'tanh'
    # R['act_name'] = 'sin'
    # R['act_name'] = 'gauss'
    # R['act_name'] = 'sinAddcos'

    # R['act_name'] = 'fourier'

    # R['opt2initW'] = 'normal'
    # R['opt2initW'] = 'uniform'
    R['opt2initW'] = 'scale_uniform'

    # R['opt2initB'] = 'normal'
    # R['opt2initB'] = 'uniform'
    R['opt2initB'] = 'scale_uniform'
    solve_possion(Rdic=R)