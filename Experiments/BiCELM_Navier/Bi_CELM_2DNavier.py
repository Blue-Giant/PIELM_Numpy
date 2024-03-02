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
from utilizers import dataUtilizer2numpy
from utilizers import Load_data2Mat
from utilizers import RFN_Log_Print
from Problems import General_Laplace
from Problems import Biharmonic_eqs


# Global random feature network
class GELM(object):
    def __init__(self, indim=1, outdim=1, num2hidden=None, name2Model='PIELM', actName2hidden='tanh', actName2Deri='tanh',
                 actName2DDeri='tanh', type2float='float32', opt2init_W='xavier_normal',  opt2init_B='xavier_uniform',
                 W_sigma=1.0, B_sigma=1.0):
        super(GELM, self).__init__()

        self.indim = indim
        self.outdim = outdim
        self.num2hidden = num2hidden
        self.name2Model = name2Model
        self.actName2hidden = actName2hidden
        self.actName2derivative = actName2Deri
        self.actName2DDeri = actName2DDeri

        if type2float == 'float32':
            self.float_type = np.float32
        elif type2float == 'float64':
            self.float_type = np.float64
        elif type2float == 'float16':
            self.float_type = np.float16

        # The ELM solver for solving PDEs
        if str.upper(name2Model) == 'PIELM':
            self.ELM2U = ELM_Base.PIELM(
                dim2in=indim, dim2out=1, num2hidden_units=num2hidden, name2Model='DNN', actName2hidden=actName2hidden,
                actName2Deri=actName2Deri, actName2DDeri=actName2DDeri, type2float=type2float,
                opt2init_hiddenW=opt2init_W, opt2init_hiddenB=opt2init_B, sigma2W=W_sigma, sigma2B=B_sigma)
            self.ELM2V = ELM_Base.PIELM(
                dim2in=indim, dim2out=1, num2hidden_units=num2hidden, name2Model='DNN', actName2hidden=actName2hidden,
                actName2Deri=actName2Deri, actName2DDeri=actName2DDeri, type2float=type2float,
                opt2init_hiddenW=opt2init_W, opt2init_hiddenB=opt2init_B, sigma2W=W_sigma, sigma2B=B_sigma)
        else:
            self.ELM2U = ELM_Base.FourierFeaturePIELM(
                dim2in=indim, dim2out=1, num2hidden_units=num2hidden, name2Model='DNN', actName2hidden=actName2hidden,
                actName2Deri=actName2Deri, actName2DDeri=actName2DDeri, type2float=type2float,
                opt2init_hiddenW=opt2init_W, opt2init_hiddenB=opt2init_B, sigma2W=W_sigma, sigma2B=B_sigma)
            self.ELM2V = ELM_Base.FourierFeaturePIELM(
                dim2in=indim, dim2out=1, num2hidden_units=num2hidden, name2Model='DNN', actName2hidden=actName2hidden,
                actName2Deri=actName2Deri, actName2DDeri=actName2DDeri, type2float=type2float,
                opt2init_hiddenW=opt2init_W, opt2init_hiddenB=opt2init_B, sigma2W=W_sigma, sigma2B=B_sigma)

    def load_meshData(self, path2file=None, mesh_number=2, to_float=True, shuffle_data=True):
        test_meshXY_file = path2file + str('meshXY') + str(mesh_number) + str('.mat')
        mesh_points = Load_data2Mat.load_Matlab_data(test_meshXY_file)
        XY_points = mesh_points['meshXY']
        shape2XY = np.shape(XY_points)
        assert (len(shape2XY) == 2)
        if shape2XY[0] == 2:
            xy_data = np.transpose(XY_points, (1, 0))
        else:
            xy_data = XY_points

        if to_float:
            xy_data = xy_data.astype(dtype=self.float_type)
        if shuffle_data:
            np.random.shuffle(xy_data)
        return xy_data

    def load_RandPointsData(self, path2file=None, to_float=True, shuffle_data=True):
        test_meshXY_file = path2file + str('testXY') + str('.mat')
        mesh_points = Load_data2Mat.load_Matlab_data(test_meshXY_file)
        XY_points = mesh_points['XY']
        shape2XY = np.shape(XY_points)
        assert (len(shape2XY) == 2)
        if shape2XY[0] == 2:
            xy_data = np.transpose(XY_points, (1, 0))
        else:
            xy_data = XY_points

        if to_float:
            xy_data = xy_data.astype(dtype=self.float_type)
        if shuffle_data:
            np.random.shuffle(xy_data)
        return xy_data

    # Assembling the matrix A,f in inner domain
    def assemble_matrix2inner(self, XY_inner=None, fside=None, if_lambda2fside=True):
        shape2XY = XY_inner.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        A12 = self.ELM2V.assemble_matrix2Laplace_2D(XY_input=XY_inner)

        A21 = self.ELM2U.assemble_matrix2Laplace_2D(XY_input=XY_inner)

        A22 = -1.0 * self.ELM2V.assemble_matrix2interior_2D(XY_input=XY_inner)

        X = np.reshape(XY_inner[:, 0], newshape=[-1, 1])
        Y = np.reshape(XY_inner[:, 1], newshape=[-1, 1])

        if if_lambda2fside:
            f = fside(X, Y)
        else:
            f = fside

        return (A12, A21, A22, f)

    # Assembling the matrix B,g in boundary domain
    def assemble_matrix2boundary(self, XY_bd=None,  gside=None, if_lambda2gside=True):
        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        B = self.ELM2U.assemble_matrix2interior_2D(XY_input=XY_bd)

        X = np.reshape(XY_bd[:, 0], newshape=[-1, 1])
        Y = np.reshape(XY_bd[:, 1], newshape=[-1, 1])

        if if_lambda2gside:
            g = gside(X, Y)
        else:
            g = gside

        return (B, g)

    # Assembling the matrix B,g in boundary domain
    def assemble_matrix2_2ndDeriBD(self, XY_bd=None, hside=None, if_lambda2hside=True):
        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        B = self.ELM2V.assemble_matrix2interior_2D(XY_input=XY_bd)

        X = np.reshape(XY_bd[:, 0], newshape=[-1, 1])
        Y = np.reshape(XY_bd[:, 1], newshape=[-1, 1])

        if if_lambda2hside:
            h = hside(X, Y)
        else:
            h= hside

        return (B, h)

    def test_rfn(self, points=None):
        rfn_basis = self.ELM2U.assemble_matrix2interior_2D(XY_input=points)
        return rfn_basis


def solve_biharmonic(Rdic=None):
    log_out_path = Rdic['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s%s.txt' % ('log2', Rdic['act_name'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    RFN_Log_Print.dictionary_out2file(Rdic, log_fileout)

    time_begin = time.time()

    model = GELM(indim=Rdic['input_dim'], outdim=Rdic['out_dim'], num2hidden=Rdic['rfn_hidden'],
                 name2Model=Rdic['name2model'], actName2hidden=Rdic['act_name'], actName2Deri=Rdic['act_name'],
                 actName2DDeri=Rdic['act_name'], type2float='float64', opt2init_W=Rdic['opt2initW'],
                 opt2init_B=Rdic['opt2initB'], W_sigma=Rdic['sigma'], B_sigma=Rdic['sigma'])

    left = 0.0
    right = 1.0
    if Rdic['Equa_name'] == 'PDE4':
        left = -1.0
        right = 1.0
        # file_path2points = '../data2points/gene_mesh01/'
        file_path2points = '../data2points/gene_mesh11/'
        points = model.load_meshData(path2file=file_path2points, mesh_number=7, to_float=True, shuffle_data=False)
    else:
        left = 0.0
        right = 1.0
        file_path2points = '../data2points/Irregular_domain01/'
        # file_path2points = '../data2points/Irregular_domain11/'
        points = model.load_RandPointsData(path2file=file_path2points, to_float=True, shuffle_data=True)
        shape2points = np.shape(points)
        Rdic['point_num2inner'] = shape2points[0]
        saveData.save_testData_or_solus2mat(points, dataName='testXY', outPath=FolderName)

    left_bd, right_bd, bottom_bd, top_bd = dataUtilizer2numpy.gene_2Drand_points2bd(
        num2point=Rdic['point_num2boundary'], variable_dim=Rdic['input_dim'], region_l=left, region_r=right,
        region_b=left, region_t=right, to_float=True, float_type=np.float64, eps=0.001, opt2rand='random',
        shuffle_uniform=True)

    f_side, u_true, u_left, u_right, u_bottom, u_top, ux_left, ux_right, uy_bottom, uy_top, lapU_left, lapU_right, \
    lapU_bottom, lapU_top = Biharmonic_eqs.get_biharmonic_Navier_2D(equa_name=Rdic['Equa_name'])

    A12, A21, A22, f_i = model.assemble_matrix2inner(XY_inner=points, fside=f_side)

    B_l, gl = model.assemble_matrix2boundary(XY_bd=left_bd, gside=u_left)

    B_r, gr = model.assemble_matrix2boundary(XY_bd=right_bd, gside=u_right)

    B_b, gb = model.assemble_matrix2boundary(XY_bd=bottom_bd, gside=u_bottom)

    B_t, gt = model.assemble_matrix2boundary(XY_bd=top_bd,  gside=u_top)

    N_l, hl = model.assemble_matrix2_2ndDeriBD(XY_bd=left_bd, hside=lapU_left)

    N_r, hr = model.assemble_matrix2_2ndDeriBD(XY_bd=right_bd, hside=lapU_right)

    N_b, hb = model.assemble_matrix2_2ndDeriBD(XY_bd=bottom_bd, hside=lapU_bottom)

    N_t, ht = model.assemble_matrix2_2ndDeriBD(XY_bd=top_bd, hside=lapU_top)

    num2inner = Rdic['point_num2inner']
    num2boundary = Rdic['point_num2boundary']
    rfn_hidden = Rdic['rfn_hidden']

    A = np.zeros([2*num2inner + 8 * num2boundary, 2*rfn_hidden])
    F = np.zeros([2*num2inner + 8 * num2boundary, 1])

    A[0:num2inner, rfn_hidden: 2*rfn_hidden] = A12
    A[num2inner: 2*num2inner, 0: rfn_hidden] = A21
    A[num2inner: 2 * num2inner, rfn_hidden: 2*rfn_hidden] = A22

    A[2*num2inner:2*num2inner+num2boundary, 0: rfn_hidden] = B_l
    A[2*num2inner + num2boundary:2*num2inner+2*num2boundary, 0: rfn_hidden] = B_r
    A[2*num2inner + 2*num2boundary:2*num2inner+3*num2boundary, 0: rfn_hidden] = B_b
    A[2*num2inner + 3*num2boundary:2*num2inner+4*num2boundary, 0: rfn_hidden] = B_t

    A[2*num2inner + 4 * num2boundary:2*num2inner + 5 * num2boundary, rfn_hidden: 2*rfn_hidden] = N_l
    A[2*num2inner + 5 * num2boundary:2*num2inner + 6 * num2boundary, rfn_hidden: 2*rfn_hidden] = N_r
    A[2*num2inner + 6 * num2boundary:2*num2inner + 7 * num2boundary, rfn_hidden: 2*rfn_hidden] = N_b
    A[2*num2inner + 7 * num2boundary:2*num2inner + 8 * num2boundary, rfn_hidden: 2*rfn_hidden] = N_t

    F[0:num2inner, :] = f_i

    F[2*num2inner:2*num2inner + num2boundary, :] = gl
    F[2*num2inner + num2boundary:2*num2inner + 2 * num2boundary, :] = gr
    F[2*num2inner + 2 * num2boundary:2*num2inner + 3 * num2boundary, :] = gb
    F[2*num2inner + 3 * num2boundary:2*num2inner + 4 * num2boundary, :] = gt

    F[2*num2inner + 4 * num2boundary:2*num2inner + 5 * num2boundary, :] = hl
    F[2*num2inner + 5 * num2boundary:2*num2inner + 6 * num2boundary, :] = hr
    F[2*num2inner + 6 * num2boundary:2*num2inner + 7 * num2boundary, :] = hb
    F[2*num2inner + 7 * num2boundary:2*num2inner + 8 * num2boundary, :] = ht

    # # rescaling
    c = 100.0

    for i in range(len(A)):
        ratio = c / A[i, :].max()
        A[i, :] = A[i, :] * ratio
        F[i] = F[i] * ratio
    # solve
    w = lstsq(A, F)[0]

    temp = model.test_rfn(points=points)
    numeri_solu = np.matmul(temp, np.reshape(w[0: rfn_hidden, :], newshape=[-1, 1]))

    time_end = time.time()
    run_time = time_end - time_begin

    exact_solu = u_true(np.reshape(points[:, 0], newshape=[-1, 1]), np.reshape(points[:, 1], newshape=[-1, 1]))

    abs_diff = np.abs(exact_solu - numeri_solu)

    max_diff = np.max(abs_diff)

    print('max_diff:', max_diff)

    rel_err = np.sqrt(np.sum(np.square(exact_solu - numeri_solu), axis=0)/np.sum(np.square(exact_solu), axis=0))

    print('relative error:', rel_err[0])

    print('running time:', run_time)

    DNN_tools.log_string('The max absolute error: %.18f\n' % max_diff, log_fileout)
    DNN_tools.log_string('The relative error: %.18f\n' % rel_err[0], log_fileout)
    DNN_tools.log_string('The running time: %.18f s\n' % run_time, log_fileout)

    plot_data.plot_scatter_solu(abs_diff, points, color='m', actName='diff', outPath=FolderName)
    plot_data.plot_scatter_solu(exact_solu, points, color='r', actName='exact', outPath=FolderName)
    plot_data.plot_scatter_solu(numeri_solu, points, color='b', actName='numerical', outPath=FolderName)

    saveData.save_testData_or_solus2mat(exact_solu, dataName='true', outPath=FolderName)
    saveData.save_testData_or_solus2mat(numeri_solu, dataName='numeri', outPath=FolderName)
    saveData.save_testData_or_solus2mat(abs_diff, dataName='perr', outPath=FolderName)

    saveData.save_run_time2mat(run_time=run_time, num2hidden=Rdic['rfn_hidden'], outPath=FolderName)
    saveData.save_max_rel2mat(max_err=max_diff, rel_err=rel_err, num2hidden=Rdic['rfn_hidden'], outPath=FolderName)


if __name__ == "__main__":
    R = {}
    # 文件保存路径设置
    file2results = 'Results'
    store_file = 'Bi2Navier2D_CRFN'
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

    R['PDE_type'] = 'Biharmonic_ELM'
    R['Equa_name'] = 'PDE4'
    # R['Equa_name'] = 'PDE5'

    # R['point_num2inner'] = 20000
    R['point_num2inner'] = 16384
    R['point_num2boundary'] = 2000

    R['name2model'] = 'PIELM'
    # R['name2model'] = 'FF_PIELM'

    # R['rfn_hidden'] = 50
    # R['rfn_hidden'] = 100
    # R['rfn_hidden'] = 250
    # R['rfn_hidden'] = 500
    # R['rfn_hidden'] = 750
    R['rfn_hidden'] = 1000
    # R['rfn_hidden'] = 1250
    # R['rfn_hidden'] = 1500
    # R['rfn_hidden'] = 1750
    # R['rfn_hidden'] = 2000
    # R['rfn_hidden'] = 2250
    # R['rfn_hidden'] = 2500

    R['input_dim'] = 2
    R['out_dim'] = 1

    # R['sigma'] = 0.5
    # R['sigma'] = 1.0
    # R['sigma'] = 2.0
    # R['sigma'] = 4.0
    # R['sigma'] = 6.0
    R['sigma'] = 8.0
    # R['sigma'] = 10.0
    # R['sigma'] = 12.0
    # R['sigma'] = 14.0
    # R['sigma'] = 16.0
    # R['sigma'] = 18.0
    # R['sigma'] = 20.0

    if R['name2model'] == 'FF_PIELM':
        R['act_name'] = 'fourier'
    else:
        # R['act_name'] = 'tanh'
        # R['act_name'] = 'enh_tanh'
        R['act_name'] = 'sin'
        # R['act_name'] = 'gauss'
        # R['act_name'] = 'sinAddcos'

    # R['opt2initW'] = 'normal'
    # R['opt2initW'] = 'uniform11'
    R['opt2initW'] = 'scale_uniform11'

    # R['opt2initB'] = 'normal'
    # R['opt2initB'] = 'uniform11'
    R['opt2initB'] = 'scale_uniform11'

    solve_biharmonic(Rdic=R)