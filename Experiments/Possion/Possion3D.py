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
from utilizers import Load_data2Mat
from Problems import General_Laplace


# Global random feature network
class GELM(object):
    def __init__(self, indim=1, outdim=1, num2hidden=None, name2Model='DNN', actName2hidden='tanh', actName2Deri='tanh',
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

        # The ELM solver for solving PDEs
        self.ELM_Bases = ELM_Base.PIELM(
            dim2in=indim, dim2out=1, num2hidden_units=num2hidden, name2Model='DNN', actName2hidden=actName2hidden,
            actName2Deri=actName2Deri, actName2DDeri=actName2DDeri, type2float=type2float,
            opt2init_hiddenW=opt2init_W, opt2init_hiddenB=opt2init_B, sigma2W=W_sigma, sigma2B=B_sigma)

        if type2float == 'float32':
            self.float_type = np.float32
        elif type2float == 'float64':
            self.float_type = np.float64
        elif type2float == 'float16':
            self.float_type = np.float16

    def gene_mesh_points2inner(self, num2mesh=100, variable_dim=2, region_l=0.0, region_r=0.0, region_b=0.0,
                               region_t=0.0, to_float=True):
        assert variable_dim == 2
        gridx = np.linspace(region_l, region_r, num2mesh, dtype=np.float32)
        gridy = np.linspace(region_b, region_t, num2mesh, dtype=np.float32)
        meshX, meshY = np.meshgrid(gridx, gridy)

        XYpoints = np.concatenate((np.reshape(meshX, newshape=[-1, 1]), np.reshape(meshY, newshape=[-1, 1])), axis=-1)

        if to_float:
            XYpoints = XYpoints.astype(dtype=self.float_type)

        return XYpoints

    def gene_rand_points2inner(self, num2point=100, variable_dim=2, region_l=0.0, region_r=0.0, region_b=0.0,
                               region_t=0.0, to_float=True):
        assert variable_dim == 3
        randx = (region_r - region_l) * np.random.rand(num2point, 1) + region_l
        randy = (region_t - region_b) * np.random.rand(num2point, 1) + region_b
        randz = (region_t - region_b) * np.random.rand(num2point, 1) + region_b
        XYpoints = np.concatenate((randx, randy, randz), axis=-1)

        if to_float:
            XYpoints = XYpoints.astype(dtype=self.float_type)

        return XYpoints

    def gene_rand_points2boundary(self, num2point=100, variable_dim=2, region_l=0.0, region_r=0.0, region_b=0.0,
                                  region_t=0.0, to_float=True):
        # np.asarray 将输入转为矩阵格式。
        # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
        # [0,1] 转换为 矩阵，然后
        # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
        region_a = float(region_l)
        region_b = float(region_r)
        assert (int(variable_dim) == 3)

        bottom_bd = (region_b - region_a) * np.random.rand(num2point, 3) + region_a
        top_bd = (region_b - region_a) * np.random.rand(num2point, 3) + region_a
        left_bd = (region_b - region_a) * np.random.rand(num2point, 3) + region_a
        right_bd = (region_b - region_a) * np.random.rand(num2point, 3) + region_a
        front_bd = (region_b - region_a) * np.random.rand(num2point, 3) + region_a
        behind_bd = (region_b - region_a) * np.random.rand(num2point, 3) + region_a
        for ii in range(num2point):
            bottom_bd[ii, 2] = region_a
            top_bd[ii, 2] = region_b
            left_bd[ii, 1] = region_a
            right_bd[ii, 1] = region_b
            behind_bd[ii, 0] = region_a
            front_bd[ii, 0] = region_b

        if to_float:
            bottom_bd = bottom_bd.astype(self.float_type)
            top_bd = top_bd.astype(self.float_type)
            left_bd = left_bd.astype(self.float_type)
            right_bd = right_bd.astype(self.float_type)
            front_bd = front_bd.astype(self.float_type)
            behind_bd = behind_bd.astype(self.float_type)

        return bottom_bd, top_bd, left_bd, right_bd, front_bd, behind_bd

    def load_meshData(self, path2file=None, mesh_number=2, to_float=True, shuffle_data=False):
        test_meshXY_file = path2file + str('testXYZ') + str(mesh_number) + str('.mat')
        mesh_points = Load_data2Mat.load_Matlab_data(test_meshXY_file)
        XYZ_points = mesh_points['XYZ']
        shape2XYZ = np.shape(XYZ_points)
        assert (len(shape2XYZ) == 2)
        if shape2XYZ[0] == 3:
            xyz_data = np.transpose(XYZ_points, (1, 0))
        else:
            xyz_data = XYZ_points

        if to_float:
            xyz_data = xyz_data.astype(dtype=self.float_type)
        if shuffle_data:
            np.random.shuffle(xyz_data)
        return xyz_data

    def gene_rand_meshData(self, path2file=None, mesh_number=2, num2point=100, variable_dim=2, region_l=0.0,
                           region_r=0.0, to_float=True, shuffle_data=False):
        test_meshXY_file = path2file + str('testXYZ') + str(mesh_number) + str('.mat')
        mesh_points = Load_data2Mat.load_Matlab_data(test_meshXY_file)
        mesh_XYZ = mesh_points['XYZ']
        shape2XYZ = np.shape(mesh_XYZ)
        assert (len(shape2XYZ) == 2)
        if shape2XYZ[0] == 3:
            xyz_mesh = np.transpose(mesh_XYZ, (1, 0))
        else:
            xyz_mesh = mesh_XYZ

        if to_float:
            xyz_mesh = xyz_mesh.astype(dtype=self.float_type)

        assert variable_dim == 3
        randx = (region_r - region_l) * np.random.rand(num2point, 1) + region_l
        randy = (region_r - region_l) * np.random.rand(num2point, 1) + region_l
        randz = (region_r - region_l) * np.random.rand(num2point, 1) + region_l

        xyz_rand = np.concatenate((randx, randy, randz), axis=-1)

        if to_float:
            xyz_rand = xyz_rand.astype(dtype=self.float_type)
        XYZ = np.concatenate([xyz_mesh, xyz_rand], axis=0)
        if shuffle_data:
            np.random.shuffle(XYZ)
        return (XYZ, xyz_mesh)

    # Assembling the matrix A,f in inner domain
    def assemble_matrix2inner(self, XYZ_inner=None, fside=None, if_lambda2fside=True):
        shape2XYZ = XYZ_inner.shape
        lenght2XYZ_shape = len(shape2XYZ)
        assert (lenght2XYZ_shape == 2)
        assert (shape2XYZ[-1] == 3)

        A = self.ELM_Bases.assemble_matrix2Laplace_3D(XYZ_input=XYZ_inner)

        X = np.reshape(XYZ_inner[:, 0], newshape=[-1, 1])
        Y = np.reshape(XYZ_inner[:, 1], newshape=[-1, 1])
        Z = np.reshape(XYZ_inner[:, 2], newshape=[-1, 1])

        if if_lambda2fside:
            f = fside(X, Y, Z)
        else:
            f = fside

        return (A, f)

    # Assembling the matrix B,g in boundary domain
    def assemble_matrix2boundary(self, XYZ_bd=None, gside=None, if_lambda2gside=True):
        shape2XYZ = XYZ_bd.shape
        lenght2XYZ_shape = len(shape2XYZ)
        assert (lenght2XYZ_shape == 2)
        assert (shape2XYZ[-1] == 3)

        B = self.ELM_Bases.assemble_matrix2boundary_3D(XYZ_input=XYZ_bd)

        X = np.reshape(XYZ_bd[:, 0], newshape=[-1, 1])
        Y = np.reshape(XYZ_bd[:, 1], newshape=[-1, 1])
        Z = np.reshape(XYZ_bd[:, 2], newshape=[-1, 1])

        if if_lambda2gside:
            g = gside(X, Y, Z)
        else:
            g = gside

        return (B, g)

    def test_model(self, points=None):
        ELM_Basis = self.ELM_Bases.assemble_matrix2interior_3D(XYZ_input=points)
        return ELM_Basis


def solve_possion(Rdic=None):
    log_out_path = Rdic['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s.txt' % ('log')
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    RFN_Log_Print.dictionary_out2file(Rdic, log_fileout)

    time_begin = time.time()

    left = 0.0
    right = 2.0

    model = GELM(indim=Rdic['input_dim'], outdim=Rdic['out_dim'], num2hidden=Rdic['rfn_hidden'], name2Model='DNN',
                 actName2hidden=Rdic['act_name'], actName2Deri=Rdic['act_name'], actName2DDeri=Rdic['act_name'],
                 type2float='float64', opt2init_W=Rdic['opt2initW'], opt2init_B=Rdic['opt2initB'],
                 W_sigma=Rdic['sigma'], B_sigma=Rdic['sigma'])

    type2float = np.float64
    # points = dataUtilizer2numpy.gene_3Drand_points2inner(
    #     num2point=Rdic['point_num2inner'], variable_dim=Rdic['input_dim'], region_l=left, region_r=right,
    #     region_b=left, region_t=right, to_float=True, float_type=type2float, opt2rand='uniform', shuffle_point=True)

    # points, mesh = dataUtilizer2numpy.gene_mesh_gene_rand_3Dinner(
    #     num2mesh=50, num2point=Rdic['point_num2inner'], variable_dim=Rdic['input_dim'], region_left=left,
    #     region_right=right, region_bottom=left, region_top=right, region_front=right,
    #     region_behind=left, to_float=True, float_type=type2float, opt2rand='uniform', shuffle_point=True)

    left = 0.0
    right = 1.0
    # points = model.gene_rand_points2inner(num2point=Rdic['point_num2inner'], variable_dim=Rdic['input_dim'],
    #                                           region_l=left, region_r=right, region_b=left, region_t=right,
    #                                           to_float=True)
    # saveData.save_testData_or_solus2mat(points, dataName='testXYZ', outPath=FolderName)

    left = 0.0
    right = 1.0
    # file_path2mesh_points = '../dataMat_highDim/ThreeD2Fixed_X/'
    # # file_path2mesh_points = '../dataMat_highDim/ThreeD2Fixed_Y/'
    file_path2mesh_points = '../dataMat_highDim/ThreeD2Fixed_Z/'
    points, points2mesh = model.gene_rand_meshData(path2file=file_path2mesh_points, mesh_number=7, num2point=10000,
                                                   variable_dim=Rdic['input_dim'], region_l=left, region_r=right,
                                                   to_float=True, shuffle_data=True)
    saveData.save_testData_or_solus2mat(points, dataName='testXYZ', outPath=FolderName)

    # left = 0.0
    # right = 1.0
    # # file_path2mesh_points = '../dataMat_highDim/ThreeD2Fixed_X/'
    # # file_path2mesh_points = '../dataMat_highDim/ThreeD2Fixed_Y/'
    # file_path2mesh_points = '../dataMat_highDim/ThreeD2Fixed_Z/'
    # points =model.load_meshData(path2file=file_path2mesh_points, mesh_number=7, to_float=True, shuffle_data=True)

    bottom_bd, top_bd, left_bd, right_bd, front_bd, behind_bd = model.gene_rand_points2boundary(
        num2point=Rdic['point_num2boundary'], variable_dim=Rdic['input_dim'], region_l=left, region_r=right,
        region_b=left, region_t=right, to_float=True)

    f_side, u_true, u_left, u_right, u_bottom, u_top, u_front, u_behind = General_Laplace.get_infos2Laplace_3D(
        equa_name=Rdic['Equa_name'])

    A_I, f_i = model.assemble_matrix2inner(XYZ_inner=points, fside=f_side, if_lambda2fside=True)

    B_l, gl = model.assemble_matrix2boundary(XYZ_bd=left_bd, gside=u_left)

    B_r, gr = model.assemble_matrix2boundary(XYZ_bd=right_bd, gside=u_right)

    B_b, gb = model.assemble_matrix2boundary(XYZ_bd=bottom_bd, gside=u_bottom)

    B_t, gt = model.assemble_matrix2boundary(XYZ_bd=top_bd, gside=u_top)

    B_f, gf = model.assemble_matrix2boundary(XYZ_bd=front_bd, gside=u_front)

    B_bh, gbh = model.assemble_matrix2boundary(XYZ_bd=behind_bd, gside=u_behind)

    # num2inner = Rdic['point_num2inner']
    num2inner = points.shape[0]
    num2boundary = Rdic['point_num2boundary']
    rfn_hidden = Rdic['rfn_hidden']

    A = np.zeros([num2inner + 6 * num2boundary, rfn_hidden])
    F = np.zeros([num2inner + 6 * num2boundary, 1])

    A[0:num2inner, :] = A_I

    A[num2inner:num2inner + num2boundary, :] = B_l
    A[num2inner + num2boundary:num2inner + 2 * num2boundary, :] = B_r
    A[num2inner + 2 * num2boundary:num2inner + 3 * num2boundary, :] = B_b
    A[num2inner + 3 * num2boundary:num2inner + 4 * num2boundary, :] = B_t
    A[num2inner + 4 * num2boundary:num2inner + 5 * num2boundary, :] = B_f
    A[num2inner + 5 * num2boundary:num2inner + 6 * num2boundary, :] = B_bh

    F[0:num2inner, :] = f_i

    F[num2inner:num2inner + num2boundary, :] = gl
    F[num2inner + num2boundary:num2inner + 2 * num2boundary, :] = gr
    F[num2inner + 2 * num2boundary:num2inner + 3 * num2boundary, :] = gb
    F[num2inner + 3 * num2boundary:num2inner + 4 * num2boundary, :] = gt
    F[num2inner + 4 * num2boundary:num2inner + 5 * num2boundary, :] = gf
    F[num2inner + 5 * num2boundary:num2inner + 6 * num2boundary, :] = gbh

    # # rescaling
    c = 100.0

    for i in range(len(A)):
        ratio = c / A[i, :].max()
        A[i, :] = A[i, :] * ratio
        F[i] = F[i] * ratio
    # solve
    w = lstsq(A, F)[0]

    temp = model.test_model(points=points2mesh)
    numeri_solu = np.matmul(temp, w)

    time_end = time.time()
    run_time = time_end - time_begin

    exact_solu = u_true(np.reshape(points2mesh[:, 0], newshape=[-1, 1]),
                        np.reshape(points2mesh[:, 1], newshape=[-1, 1]),
                        np.reshape(points2mesh[:, 2], newshape=[-1, 1]))

    abs_diff = np.abs(exact_solu - numeri_solu)

    max_diff = np.max(abs_diff)

    print('max_diff:', max_diff)

    rel_err = np.sqrt(np.sum(np.square(exact_solu - numeri_solu), axis=0) / np.sum(np.square(exact_solu), axis=0))

    print('relative error:', rel_err[0])

    print('running time:', run_time)

    DNN_tools.log_string('The max absolute error: %.18f\n' % max_diff, log_fileout)
    DNN_tools.log_string('The relative error: %.18f\n' % rel_err[0], log_fileout)
    DNN_tools.log_string('The running time: %.18f s\n' % run_time, log_fileout)

    plot_data.plot_scatter_solu(abs_diff, points2mesh, color='m', actName='diff', outPath=FolderName)
    plot_data.plot_scatter_solu(exact_solu, points2mesh, color='r', actName='exact', outPath=FolderName)
    plot_data.plot_scatter_solu(numeri_solu, points2mesh, color='b', actName='numerical', outPath=FolderName)

    saveData.save_testData_or_solus2mat(exact_solu, dataName='true', outPath=FolderName)
    saveData.save_testData_or_solus2mat(numeri_solu, dataName='numeri', outPath=FolderName)
    saveData.save_testData_or_solus2mat(abs_diff, dataName='perr', outPath=FolderName)

    saveData.save_run_time2mat(run_time=run_time, num2hidden=Rdic['rfn_hidden'], outPath=FolderName)
    saveData.save_max_rel2mat(max_err=max_diff, rel_err=rel_err, num2hidden=Rdic['rfn_hidden'], outPath=FolderName)


if __name__ == "__main__":
    R = {}
    # 文件保存路径设置
    file2results = 'Results'
    store_file = 'Laplace3D'
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
    # R['Equa_name'] = 'PDE3'
    # R['Equa_name'] = 'PDE4'

    R['point_num2inner'] = 10000
    R['point_num2boundary'] = 1000
    R['rfn_hidden'] = 2000
    R['input_dim'] = 3
    R['out_dim'] = 1

    R['name2model'] = 'PIELM'

    R['sigma'] = 20

    # R['act_name'] = 'tanh'
    R['act_name'] = 'sin'
    # R['act_name'] = 'gauss'
    # R['act_name'] = 'sinAddcos'

    # R['opt2initW'] = 'normal'
    R['opt2initW'] = 'scale_uniform'

    # R['opt2initB'] = 'normal'
    # R['opt2initB'] = 'uniform'
    R['opt2initB'] = 'scale_uniform'
    solve_possion(Rdic=R)