# !python3
# -*- coding: utf-8 -*-
# author: flag

import numpy as np
import scipy.io
import torch


# load the data from matlab of .mat
def load_Matlab_data(filename=None):
    data = scipy.io.loadmat(filename, mat_dtype=True, struct_as_record=True)  # variable_names='CATC'
    return data


def get_randomData2mat(dim=2, data_path=None, to_torch=False, to_float=True, float_type='float64', to_cuda=False, gpu_no=0, use_grad2x=False):
    if dim == 2:
        file_name2data = str(data_path) + '/' + str('testXY') + str('.mat')
        data2matlab = load_Matlab_data(filename=file_name2data)
        data2points = data2matlab['XY']
    elif dim == 3:
        file_name2data = str(data_path) + '/' + str('testXYZ') + str('.mat')
        data2matlab = load_Matlab_data(filename=file_name2data)
        data2points = data2matlab['XYZ']
    elif dim == 4:
        file_name2data = str(data_path) + '/' + str('testXYZS') + str('.mat')
        data2matlab = load_Matlab_data(filename=file_name2data)
        data2points = data2matlab['XYZS']
    elif dim == 5:
        file_name2data = str(data_path) + '/' + str('testXYZST') + str('.mat')
        data2matlab = load_Matlab_data(filename=file_name2data)
        data2points = data2matlab['XYZST']
    elif dim == 8:
        file_name2data = str(data_path) + '/' + str('testXYZRSTVW') + str('.mat')
        data2matlab = load_Matlab_data(filename=file_name2data)
        data2points = data2matlab['XYZRSTVW']
    if to_float:
        data2points = data2points.astype(np.float32)

    shape2data = np.shape(data2points)
    assert (len(shape2data) == 2)
    if shape2data[0] == 3 or shape2data[0] == 4 or shape2data[0] == 4 or shape2data[0] == 5 or shape2data[0] == 8:
        data2points = np.transpose(data2points, (1, 0))

    if to_torch:
        torch_data = torch.from_numpy(data2points)

        if to_cuda:
            torch_data = torch_data.cuda(device='cuda:' + str(gpu_no))

        torch_data.requires_grad = use_grad2x
    return torch_data


def get_meshData2Laplace(equation_name=None, mesh_number=2, to_torch=False, to_float=True, to_cuda=False, gpu_no=0,
                         use_grad2x=False):
    if equation_name == 'multi_scale2D_1':
        test_meshXY_file = '../dataMat2pLaplace/E1/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        test_meshXY_file = '../dataMat2pLaplace/E2/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        test_meshXY_file = '../dataMat2pLaplace/E3/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        test_meshXY_file = '../dataMat2pLaplace/E4/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        test_meshXY_file = '../dataMat2pLaplace/E5/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_6':
        assert (mesh_number == 6)
        test_meshXY_file = '../dataMat2pLaplace/E6/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_7':
        assert(mesh_number == 6)
        test_meshXY_file = '../dataMat2pLaplace/E7/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_points = load_Matlab_data(test_meshXY_file)
    XY_points = mesh_points['meshXY']
    shape2XY = np.shape(XY_points)
    assert(len(shape2XY) == 2)
    if shape2XY[0] == 2:
        xy_data = np.transpose(XY_points, (1, 0))
    else:
        xy_data = XY_points

    if to_float:
        xy_data = xy_data.astype(np.float32)

    if to_torch:
        xy_data = torch.from_numpy(xy_data)

        if to_cuda:
            xy_data = xy_data.cuda(device='cuda:' + str(gpu_no))

        xy_data.requires_grad = use_grad2x
    return xy_data


def get_data2pLaplace(equation_name=None, mesh_number=2, to_torch=False, to_float=True, to_cuda=False, gpu_no=0,
                      use_grad2x=False):
    if equation_name == 'multi_scale2D_1':
        test_meshXY_file = '../dataMat2pLaplace/E1/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        test_meshXY_file = '../dataMat2pLaplace/E2/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        test_meshXY_file = '../dataMat2pLaplace/E3/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        test_meshXY_file = '../dataMat2pLaplace/E4/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        test_meshXY_file = '../dataMat2pLaplace/E5/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_6':
        test_meshXY_file = '../dataMat2pLaplace/E6/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_9':
        test_meshXY_file = '../dataMat2pLaplace/E9/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_10':
        test_meshXY_file = '../dataMat2pLaplace/E10/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_7':
        assert(mesh_number == 6)
        test_meshXY_file = '../dataMat2pLaplace/E7/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_XY = load_Matlab_data(test_meshXY_file)
    XY_points = mesh_XY['meshXY']
    shape2XY = np.shape(XY_points)
    assert (len(shape2XY) == 2)
    if shape2XY[0] == 2:
        test_xy_data = np.transpose(XY_points, (1, 0))
    else:
        test_xy_data = XY_points

    if to_float:
        test_xy_data = test_xy_data.astype(np.float32)

    if to_torch:
        test_xy_data = torch.from_numpy(test_xy_data)

        if to_cuda:
            test_xy_data = test_xy_data.cuda(device='cuda:' + str(gpu_no))

        test_xy_data.requires_grad = use_grad2x
    return test_xy_data


def get_meshData2Boltzmann(equation_name=None, domain_lr='01', mesh_number=2, to_torch=False, to_float=True,
                           to_cuda=False, gpu_no=0, use_grad2x=False):
    if domain_lr == '01':
        meshXY_file = '../dataMat2Boltz/meshData_01/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif domain_lr == '11':
        meshXY_file = '../dataMat2Boltz/meshData_11/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_points = load_Matlab_data(meshXY_file)
    XY_points = mesh_points['meshXY']
    shape2XY = np.shape(XY_points)
    assert (len(shape2XY) == 2)
    if shape2XY[0] == 2:
        xy_data = np.transpose(XY_points, (1, 0))
    else:
        xy_data = XY_points

    if to_float:
        xy_data = xy_data.astype(np.float32)

    if to_torch:
        xy_data = torch.from_numpy(xy_data)

        if to_cuda:
            xy_data = xy_data.cuda(device='cuda:' + str(gpu_no))

        xy_data.requires_grad = use_grad2x
    return xy_data


def get_meshdata2Convection(equation_name=None, mesh_number=2, to_torch=False, to_float=True, to_cuda=False,
                            gpu_no=0, use_grad2x=False):
    if equation_name == 'multi_scale2D_1':
        meshXY_file = '../dataMat2pLaplace/E1/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        meshXY_file = '../dataMat2pLaplace/E2/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        meshXY_file = '../dataMat2pLaplace/E3/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        meshXY_file = '../dataMat2pLaplace/E4/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        meshXY_file = '../dataMat2pLaplace/E5/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_6':
        meshXY_file = '../dataMat2pLaplace/E6/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_points = load_Matlab_data(meshXY_file)
    XY_points = mesh_points['meshXY']
    shape2XY = np.shape(XY_points)
    assert (len(shape2XY) == 2)
    if shape2XY[0] == 2:
        xy_data = np.transpose(XY_points, (1, 0))
    else:
        xy_data = XY_points

    if to_float:
        xy_data = xy_data.astype(np.float32)

    if to_torch:
        xy_data = torch.from_numpy(xy_data)

        if to_cuda:
            xy_data = xy_data.cuda(device='cuda:' + str(gpu_no))

        xy_data.requires_grad = use_grad2x
    return xy_data


if __name__ == '__main__':
    mat_data_path = '../dataMat_highDim'
    mat_data = get_randomData2mat(dim=2, data_path=mat_data_path)
    print('end!!!!')