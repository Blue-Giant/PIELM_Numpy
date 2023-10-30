from utilizers import DNN_tools


def dictionary_out2file(R_dic, log_fileout):
    # -----------------------------------------------------------------------------------------------------------------
    DNN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['Equa_name']), log_fileout)

    # -----------------------------------------------------------------------------------------------------------------
    DNN_tools.log_string('The model:%s\n' % str(R_dic['name2model']), log_fileout)
    DNN_tools.log_string('num to the unit of rfn basis:%s\n' % str(R_dic['rfn_hidden']), log_fileout)
    DNN_tools.log_string('Activate function for RFN basis: %s\n' % str(R_dic['act_name']), log_fileout)

    DNN_tools.log_string('Option for initializing the weights of RFN basis: %s\n' % str(R_dic['opt2initW']), log_fileout)
    DNN_tools.log_string('Option for initializing the weights of RFN basis: %s\n' % str(R_dic['opt2initB']), log_fileout)

    # -----------------------------------------------------------------------------------------------------------------
    DNN_tools.log_string('Batch-size 2 interior: %s\n' % str(R_dic['point_num2inner']), log_fileout)
    DNN_tools.log_string('Batch-size 2 boundary: %s\n' % str(R_dic['point_num2boundary']), log_fileout)