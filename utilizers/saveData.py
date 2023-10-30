"""
@author: LXA
 Date: 2020 年 5 月 31 日
"""
import scipy.io as scio


def save_trainLoss2mat_1actFunc(loss_it, loss_bd, loss, actName=None, outPath=None):
    # if actName == 's2ReLU':
    #     outFile2data = '%s/Loss2s2ReLU.mat' % (outPath)
    # if actName == 'sReLU':
    #     outFile2data = '%s/Loss2sReLU.mat' % (outPath)
    # if actName == 'ReLU':
    #     outFile2data = '%s/Loss2ReLU.mat' % (outPath)
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_it'
    key2mat_2 = 'loss_bd'
    key2mat_3 = 'loss'
    scio.savemat(outFile2data, {key2mat_1: loss_it, key2mat_2: loss_bd, key2mat_3: loss})


def save_trainLoss2mat_1act_Func(loss_it, loss_bd, loss_bdd, loss, actName=None, outPath=None):
    # if actName == 's2ReLU':
    #     outFile2data = '%s/Loss2s2ReLU.mat' % (outPath)
    # if actName == 'sReLU':
    #     outFile2data = '%s/Loss2sReLU.mat' % (outPath)
    # if actName == 'ReLU':
    #     outFile2data = '%s/Loss2ReLU.mat' % (outPath)
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_it'
    key2mat_2 = 'loss_bd'
    key2mat_3 = 'loss_bdd'
    key2mat_4 = 'loss'
    scio.savemat(outFile2data, {key2mat_1: loss_it, key2mat_2: loss_bd, key2mat_3: loss_bdd, key2mat_4: loss})


def save_trainLoss2mat_1actFunc_Dirichlet(loss_it, loss_bd, loss_bd2, loss_all, actName=None, outPath=None):
    # if actName == 's2ReLU':
    #     outFile2data = '%s/Loss2s2ReLU.mat' % (outPath)
    # if actName == 'sReLU':
    #     outFile2data = '%s/Loss2sReLU.mat' % (outPath)
    # if actName == 'ReLU':
    #     outFile2data = '%s/Loss2ReLU.mat' % (outPath)
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_it'
    key2mat_2 = 'loss_bd0'
    key2mat_4 = 'loss_bd2'
    key2mat_5 = 'loss'
    scio.savemat(outFile2data, {key2mat_1: loss_it, key2mat_2: loss_bd, key2mat_4: loss_bd2, key2mat_5: loss_all})


def save_trainLoss2mat_1actFunc_Navier(loss_U, loss_bd, loss_Psi, loss_bdd, loss, actName=None, outPath=None):
    # print('actName:', actName)
    # print('id of loss_it:', id(loss_U))
    # print('values of loss_it:', loss_U)
    if str.lower(actName) == 's2relu':
        outFile2data = '%s/Loss_s2ReLU.mat' % (outPath)
    elif str.lower(actName) == 'srelu':
        outFile2data = '%s/Loss_sReLU.mat' % (outPath)
    elif str.lower(actName) == 'relu':
        outFile2data = '%s/Loss_ReLU.mat' % (outPath)
    else:
        outFile2data = '%s/Loss_%s.mat' % (outPath, str(actName))

    # print('outFile2data:', outFile2data)

    key2mat_0 = 'lossU_%s' % (str(actName))
    key2mat_1 = 'lossBD_%s' % (str(actName))
    key2mat_2 = 'lossPsi_%s' % (str(actName))
    key2mat_3 = 'lossBDD_%s' % (str(actName))
    key2mat_4 = 'loss_%s' % (str(actName))
    scio.savemat(outFile2data, {key2mat_0: loss_U, key2mat_1: loss_bd, key2mat_2: loss_Psi, key2mat_3: loss_bdd, key2mat_4: loss})


def save_train_MSE_REL2mat(Mse_data, Rel_data, actName=None, outPath=None):
    # if actName == 's2ReLU':
    #     outFile2data = '%s/train_Err2s2ReLU.mat' % (outPath)
    # if actName == 'sReLU':
    #     outFile2data = '%s/train_Err2sReLU.mat' % (outPath)
    # if actName == 'ReLU':
    #     outFile2data = '%s/train_Err2ReLU.mat' % (outPath)
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'mse'
    key2mat_2 = 'rel'
    scio.savemat(outFile2data, {key2mat_1: Mse_data, key2mat_2: Rel_data})


def save_meshData2mat(data, dataName=None, mesh_number=4, outPath=None):
    outFile2data = '%s/%s%s.mat' % (outPath, dataName, mesh_number)
    key2mat = 'U%s' % (str.upper(dataName))
    scio.savemat(outFile2data, {key2mat: data})


# 一个mat文件保存一种数据
def save_testData_or_solus2mat(data, dataName=None, outPath=None):
    if str.lower(dataName) == 'testxy':
        outFile2data = '%s/testData2XY.mat' % (outPath)
        key2mat = 'Points2XY'
    elif str.lower(dataName) == 'testxyz':
        outFile2data = '%s/testData2XYZ.mat' % (outPath)
        key2mat = 'Points2XYZ'
    elif str.lower(dataName) == 'testxyzs':
        outFile2data = '%s/testData2XYZS.mat' % (outPath)
        key2mat = 'Points2XYZS'
    elif str.lower(dataName) == 'testxyzst':
        outFile2data = '%s/testData2XYZST.mat' % (outPath)
        key2mat = 'Points2XYZST'
    elif str.lower(dataName) == 'utrue':
        outFile2data = '%s/Utrue.mat' % (outPath)
        key2mat = 'Utrue'
    else:
        outFile2data = '%s/U%s.mat' % (outPath, dataName)
        key2mat = 'U%s' % (str.upper(dataName))

    scio.savemat(outFile2data, {key2mat: data})


# 合并保存数据
def save_2testSolus2mat(exact_solution, dnn_solution, actName=None, actName1=None, outPath=None):
    outFile2data = '%s/test_solus.mat' % (outPath)
    if str.lower(actName) == 'utrue':
        key2mat_1 = 'Utrue'
    key2mat_2 = 'U%s' % (actName1)
    scio.savemat(outFile2data, {key2mat_1: exact_solution, key2mat_2: dnn_solution})


# 合并保存数据
def save_3testSolus2mat(exact_solution, solution2act1, solution2act2, actName='Utrue', actName1=None, actName2=None,
                        outPath=None):
    outFile2data = '%s/solutions.mat' % (outPath)
    if str.lower(actName) == 'utrue':
        key2mat_1 = 'Utrue'
    key2mat_3 = 'U%s' % (actName1)
    key2mat_4 = 'U%s' % (actName2)
    scio.savemat(outFile2data, {key2mat_1: exact_solution, key2mat_3: solution2act1, key2mat_4: solution2act2})


# 合并保存数据
def save_4testSolus2mat(exact_solution, solution2act1, solution2act2, solution2act3, actName='Utrue', actName1=None,
                        actName2=None, actName3=None, outPath=None):
    outFile2data = '%s/solutions.mat' % (outPath)
    if str.lower(actName) == 'utrue':
        key2mat_1 = 'Utrue'
    key2mat_2 = 'U%s' % (actName1)
    key2mat_3 = 'U%s' % (actName2)
    key2mat_4 = 'U%s' % (actName3)
    scio.savemat(outFile2data, {key2mat_1: exact_solution, key2mat_2: solution2act1, key2mat_3: solution2act2,
                                key2mat_4: solution2act3})


def save_testLoss2mat_1act_Func(loss_it, loss_bd, loss_bd2, loss, actName=None, outPath=None):
    # if actName == 's2ReLU':
    #     outFile2data = '%s/Loss2s2ReLU.mat' % (outPath)
    # if actName == 'sReLU':
    #     outFile2data = '%s/Loss2sReLU.mat' % (outPath)
    # if actName == 'ReLU':
    #     outFile2data = '%s/Loss2ReLU.mat' % (outPath)
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_it'
    key2mat_2 = 'loss_bd0'
    key2mat_3 = 'loss_bd2'
    key2mat_4 = 'loss'
    scio.savemat(outFile2data, {key2mat_1: loss_it, key2mat_2: loss_bd, key2mat_3: loss_bd2, key2mat_4: loss})


def save_testMSE_REL2mat(Mse_data, Rel_data, actName=None, outPath=None):
    # if actName == 's2ReLU':
    #     outFile2data = '%s/test_Err2s2ReLU.mat' % (outPath)
    # if actName == 'sReLU':
    #     outFile2data = '%s/test_Err2sReLU.mat' % (outPath)
    # if actName == 'ReLU':
    #     outFile2data = '%s/test_Err2ReLU.mat' % (outPath)
    outFile2data = '%s/test_Err2%s.mat' % (outPath, actName)
    key2mat_1 = 'mse'
    key2mat_2 = 'rel'
    scio.savemat(outFile2data, {key2mat_1: Mse_data, key2mat_2: Rel_data})


# 按误差类别保存，MSE和REL
def save_testErrors2mat(err_sReLU, err_s2ReLU, errName=None, outPath=None):
    if str.upper(errName) == 'MSE':
        outFile2data = '%s/MSE.mat' % (outPath)
        key2mat_1 = 'mse2sReLU'
        key2mat_2 = 'mse2s2ReLU'
        scio.savemat(outFile2data, {key2mat_1: err_sReLU, key2mat_2: err_s2ReLU})
    elif str.upper(errName) == 'REL':
        outFile2data = '%s/REL.mat' % (outPath)
        key2mat_1 = 'rel2sReLU'
        key2mat_2 = 'rel2s2ReLU'
        scio.savemat(outFile2data, {key2mat_1: err_sReLU, key2mat_2: err_s2ReLU})


def save_test_point_wise_err2mat(data2point_wise_err, actName=None, outPath=None):
    if str.lower(actName) == 'srelu':
        outFile2data = '%s/pERR2sReLU.mat' % (outPath)
        key2mat = 'pERR2sReLU'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 's2relu':
        outFile2data = '%s/pERR2s2ReLU.mat' % (outPath)
        key2mat = 'pERR2s2ReLU'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 's3relu':
        outFile2data = '%s/pERR2s3ReLU.mat' % (outPath)
        key2mat = 'pERR2smReLU'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'csrelu':
        outFile2data = '%s/pERR2CsReLU.mat' % (outPath)
        key2mat = 'pERR2CsReLU'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'relu':
        outFile2data = '%s/pERR2ReLU.mat' % (outPath)
        key2mat = 'pERR2ReLU'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'sin':
        outFile2data = '%s/pERR2Sin.mat' % (outPath)
        key2mat = 'pERR2Sin'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'powsin_srelu':
        outFile2data = '%s/pERR2p2SinSrelu.mat' % (outPath)
        key2mat = 'pERR2p2SinSrelu'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'tanh':
        outFile2data = '%s/pERR2tanh.mat' % (outPath)
        key2mat = 'pERR2tanh'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'elu':
        outFile2data = '%s/pERR2elu.mat' % (outPath)
        key2mat = 'pERR2elu'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'singauss':
        outFile2data = '%s/pERR2sgauss.mat' % (outPath)
        key2mat = 'pERR2sgauss'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'modify_mexican':
        outFile2data = '%s/pERR2mmexican.mat' % (outPath)
        key2mat = 'pERR2sgauss'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'sin_modify_mexican':
        outFile2data = '%s/pERR2sm-mexican.mat' % (outPath)
        key2mat = 'pERR2sgauss'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})


def save_run_time2mat(run_time=0.01, num2hidden=1000, outPath=None):
    outFile2data = '%s/runTime_%s.mat' % (outPath, str(num2hidden))
    key2mat = 'time'
    key_hidden = 'basis'
    scio.savemat(outFile2data, {key2mat: run_time, key_hidden: num2hidden})


def save_max_rel2mat(max_err=0.01, rel_err=0.01, num2hidden=1000, outPath=None):
    outFile2data = '%s/Errors_%s.mat' % (outPath, str(num2hidden))
    key2max = 'max'
    key2rel = 'rel'
    key2hidden = 'basis'
    scio.savemat(outFile2data, {key2max: max_err, key2rel: rel_err, key2hidden: num2hidden})