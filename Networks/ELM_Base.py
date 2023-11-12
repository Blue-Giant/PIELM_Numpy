import torch
import torch.nn as tn
import numpy as np


def ReLUFunc(x):
    y = (np.abs(x) + x) / 2.0
    return y


def SigmoidFunc(x):
    y = 1.0/(1+np.exp(-x))
    return y


# The defined activation function
class my_actFunc(object):
    def __init__(self, actName='linear'):
        """
        Args:
            actName: the name of activation function
        """
        super(my_actFunc, self).__init__()
        self.actName = actName

    def __call__(self, x_input):
        """
        Args:
             x_input: the input data
        return:
             the output after activate
        """
        if str.lower(self.actName) == 'relu':
            out_x = ReLUFunc(x_input)
        elif str.lower(self.actName) == 'tanh':
            out_x = np.tanh(x_input)
        elif str.lower(self.actName) == 'enhance_tanh' or str.lower(self.actName) == 'enh_tanh':  # Enhance Tanh
            out_x = np.tanh(0.5*np.pi*x_input)
        elif str.lower(self.actName) == 'srelu':
            out_x = ReLUFunc(x_input)*ReLUFunc(1-x_input)
        elif str.lower(self.actName) == 's2relu':
            out_x = ReLUFunc(x_input)*ReLUFunc(1-x_input)*np.sin(2*np.pi*x_input)
        elif str.lower(self.actName) == 'sin':
            out_x = np.sin(x_input)
        elif str.lower(self.actName) == 'sinaddcos':
            out_x = 0.5*np.sin(x_input) + 0.5*np.cos(x_input)
            # out_x = 0.75*torch.sin(x_input) + 0.75*torch.cos(x_input)
            # out_x = torch.sin(x_input) + torch.cos(x_input)
        elif str.lower(self.actName) == 'fourier':
            out_x = np.concatenate([np.sin(x_input), np.cos(x_input)], axis=-1)
        elif str.lower(self.actName) == 'sigmoid':
            out_x = SigmoidFunc(x_input)
        elif str.lower(self.actName) == 'gcu':
            out_x = x_input*np.cos(x_input)
        elif str.lower(self.actName) == 'gauss':
            out_x = np.exp(-1.0 * x_input * x_input)
        elif str.lower(self.actName) == 'requ':
            out_x = ReLUFunc(x_input)*ReLUFunc(x_input)
        elif str.lower(self.actName) == 'recu':
            out_x = ReLUFunc(x_input)*ReLUFunc(x_input)*ReLUFunc(x_input)
        else:
            out_x = x_input
        return out_x


# The defined activation function for 1st derivative
class ActFunc2Derivate(object):
    def __init__(self, actName='linear'):
        super(ActFunc2Derivate, self).__init__()
        self.actName = actName

    def __call__(self, x_input):
        if str.lower(self.actName) == 'tanh':
            out_x = 1.0 - np.tanh(x_input)*np.tanh(x_input)
        elif str.lower(self.actName) == 'enhance_tanh' or str.lower(self.actName) == 'enh_tanh':  # Enhance Tanh
            out_x = 1.0 - 0.5*np.pi * np.tanh(0.5*np.pi*x_input)*np.tanh(0.5*np.pi*x_input)
        elif str.lower(self.actName) == 'sin':
            out_x = np.cos(x_input)
        elif str.lower(self.actName) == 'sinaddcos':
            out_x = 0.5*np.cos(x_input) - 0.5*np.sin(x_input)
        elif str.lower(self.actName) == 'fourier':
            out_x = np.concatenate([np.cos(x_input), -np.sin(x_input)], axis=-1)
        elif str.lower(self.actName) == 'gauss':
            out_x = -2.0 * x_input * np.exp(-1.0 * x_input * x_input)
        elif str.lower(self.actName) == 'sigmoid':
            out_x = SigmoidFunc(x_input) - SigmoidFunc(x_input) * SigmoidFunc(x_input)
        return out_x


# The defined activation function for 2nd derivative
class ActFunc2DDerivate(object):
    def __init__(self, actName='linear'):
        super(ActFunc2DDerivate, self).__init__()
        self.actName = actName

    def __call__(self, x_input):
        if str.lower(self.actName) == 'tanh':
            out_x = -2.0 * np.tanh(x_input) + 2.0 * np.tanh(x_input) * np.tanh(x_input)*np.tanh(x_input)
        elif str.lower(self.actName) == 'enhance_tanh' or str.lower(self.actName) == 'enh_tanh':  # Enhance Tanh
            out_x = np.pi*np.tanh(0.5*np.pi*x_input)*(1.0 - 0.5*np.pi * np.tanh(0.5*np.pi*x_input)*np.tanh(0.5*np.pi*x_input))
        elif str.lower(self.actName) == 'sin':
            out_x = -np.sin(x_input)
        elif str.lower(self.actName) == 'sinaddcos':
            out_x = -0.5*np.sin(x_input) - 0.5*np.cos(x_input)
        elif str.lower(self.actName) == 'fourier':
            out_x = np.concatenate([-np.sin(x_input), -np.cos(x_input)], axis=-1)
        elif str.lower(self.actName) == 'gauss':
            out_x = -2.0 * np.exp(-1.0 * x_input * x_input) + 4.0 * x_input * x_input * np.exp(-1.0 * x_input * x_input)
        return out_x


# The defined activation function for 4th derivative
class ActFunc2DDDDerivate(object):
    def __init__(self, actName='linear'):
        super(ActFunc2DDDDerivate, self).__init__()
        self.actName = actName

    def __call__(self, x_input):
        if str.lower(self.actName) == 'tanh':
            out_x = (16.0 * np.tanh(x_input) - 24.0 * np.tanh(x_input) * np.tanh(x_input)*np.tanh(x_input))*\
                    (1.0-np.tanh(x_input)*np.tanh(x_input))
        elif str.lower(self.actName) == 'enhance_tanh' or str.lower(self.actName) == 'enh_tanh':  # Enhance Tanh
            out_x = (-4.0*((np.pi)**2)*np.tanh(0.5*np.pi*x_input) + 3.0*((np.pi*np.tanh(0.5*np.pi*x_input))**3))*\
                    (1 - 0.5*np.pi * np.tanh(0.5*np.pi*x_input)*np.tanh(0.5*np.pi*x_input))
        elif str.lower(self.actName) == 'sin':
            out_x = np.sin(x_input)
        elif str.lower(self.actName) == 'sinaddcos':
            out_x = 0.5*np.sin(x_input) + 0.5*np.cos(x_input)
        elif str.lower(self.actName) == 'fourier':
            out_x = np.concatenate([np.sin(x_input), np.cos(x_input)], axis=-1)
        return out_x


class PIELM(object):
    def __init__(self, dim2in=1, dim2out=1, num2hidden_units=None, name2Model='DNN', actName2hidden='tanh',
                 actName2Deri='tanh', actName2DDeri='tanh', type2float='float32', opt2init_hiddenW='xavier_normal',
                 opt2init_hiddenB='xavier_uniform', sigma2W=5.0, sigma2B=5.0):
        """
        The basis modules for physics informed extreme learning machine
        Args:
              dim2in: the dim of variable for solving problem, or the input dim for extreme learning machine network
              dim2out: the dimension for output
              num2hidden_units: the number of hidden units for ELM
              name2Model:  the name of model
              actName2hidden: the activation function for hidden nodes
              actName2Deri: the 1stn order derivative activation function for hidden nodes
              actName2DDeri: the 2nd order derivative activation function for hidden nodes
              type2float:  the type for numerical
              opt2init_hiddenW: the initialization method for initialing the weights of hidden nodes
              opt2init_hiddenB: the initialization method for initialing the biases of hidden nodes
              sigma2W: normal initialization, it is N(0, sigma2W^2) ; uniform initialization, it is U[-sigma2W, sigma2W]
              sigma2B: normal initialization, it is N(0, sigma2B^2) ; uniform initialization, it is U[-sigma2B, sigma2B]
        """
        super(PIELM, self).__init__()
        self.indim = dim2in
        self.outdim = dim2out
        self.num2hidden = num2hidden_units
        self.name2Model = name2Model
        self.actName2bases = actName2hidden
        self.actName2Deri = actName2Deri
        self.actName2DDeri = actName2DDeri
        self.act_func2hidden = my_actFunc(actName=actName2hidden)       # the original activation function
        self.act_func2Deri = ActFunc2Derivate(actName=actName2Deri)     # the 1st order for activation function
        self.act_func2DDeri = ActFunc2DDerivate(actName=actName2DDeri)  # the 2nd order for activation function
        self.act_func2DDDDeri = ActFunc2DDDDerivate(actName=actName2DDeri)

        if type2float == 'float32':
            self.float_type = np.float32
        elif type2float == 'float64':
            self.float_type = np.float64
        elif type2float == 'float16':
            self.float_type = np.float16

        # randomly initialize the weight matrix of hidden units
        if opt2init_hiddenW == 'xavier_normal':
            stddev2W = (2.0 / (dim2in + num2hidden_units)) ** 0.5
            self.W2Hidden = stddev2W*np.random.randn(dim2in, num2hidden_units)
        elif opt2init_hiddenW == 'normal':
            self.W2Hidden = sigma2W*np.random.randn(dim2in, num2hidden_units)
        elif opt2init_hiddenW == 'uniform':
            self.W2Hidden = np.random.uniform(low=-1.0, high=1.0, size=(dim2in, num2hidden_units))
        elif opt2init_hiddenW == 'scale_uniform':
            self.W2Hidden = np.random.uniform(low=-1.0*sigma2W, high=1.0*sigma2W, size=(dim2in, num2hidden_units))

        # randomly initialize the bias vector of hidden units
        if opt2init_hiddenB == 'xavier_normal':
            stddev2B = (2.0 / (1 + num2hidden_units)) ** 0.5
            self.B2Hidden = stddev2B*np.random.randn(1, num2hidden_units)
        elif opt2init_hiddenB == 'normal':
            self.B2Hidden = sigma2B*np.random.randn(1, num2hidden_units)
        elif opt2init_hiddenB == 'uniform':
            self.B2Hidden = np.random.uniform(low=-1.0, high=1.0, size=(1, num2hidden_units))
        elif opt2init_hiddenB == 'scale_uniform':
            self.B2Hidden = np.random.uniform(low=-1.0 * sigma2B, high=sigma2B, size=(1, num2hidden_units))

        self.W2Hidden.astype(dtype=self.float_type)
        self.B2Hidden.astype(dtype=self.float_type)

    def get_linear_out2Hidden(self, input_data=None):
        """
        the linear transformation for input by operator x·W+B
        Args:
             input_data: the input points
        return:
             out2hidden: the output of linear transformation
        """
        shape2input = input_data.shape
        lenght2input_shape = len(shape2input)
        assert (lenght2input_shape == 2)
        out2hidden = np.add(np.matmul(input_data, self.W2Hidden), self.B2Hidden)   # Y = x·W+B
        return out2hidden

    def get_out2Hidden(self, input_data=None):
        """
        the linear transformation for input by operator x·W+B
        Args:
             input_data: the input points
        return:
             out2hidden: the output of linear transformation
        """
        shape2input = input_data.shape
        lenght2input_shape = len(shape2input)
        assert (lenght2input_shape == 2)
        hidden_linear = np.add(np.matmul(input_data, self.W2Hidden), self.B2Hidden)   # Y = x·W+B
        act_out2hidden = self.act_func2hidden(hidden_linear)
        return act_out2hidden

    def assemble_matrix2interior_1D(self, X_input=None):
        """
        the output for hidden nodes by operator sigma(x·W+B)
        Args:
             X_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=X_input)    # Y = x·W+B
        act_out2hidden = self.act_func2hidden(hidden_linear)              # Z = sigma(Y)
        return act_out2hidden

    def assemble_matrix2boundary_1D(self, X_input=None):
        """
        the output for hidden nodes by operator sigma(x·W+B)
        Args:
             X_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=X_input)   # Y = x·W+B
        act_out2hidden = self.act_func2hidden(hidden_linear)             # Z = sigma(Y)
        return act_out2hidden

    def assemble_matrix2First_Derivative_1D(self, X_input=None, type2boundary='left'):
        """
        the output for hidden nodes with 1st order  Derivative
        Args:
             X_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=X_input)   # Y = x·W+B
        act_out2hidden = self.act_func2Deri(hidden_linear)               # Z = sigma'(Y), this is a matrix

        W = np.reshape(self.W2Hidden, newshape=[1, -1])            # this is a vector

        if type2boundary == 'left':
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)    # the hadamard product of weight and Z
        else:
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)     # the hadamard product of weight and Z
        return coef_Matrix

    def assemble_matrix2Second_Derivative_1D(self, X_input=None):
        """
        the output for hidden nodes with 2nd order  Derivative
        Args:
             X_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=X_input)  # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)             # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)

        coef_Matrix = np.multiply(act_out2hidden, squareW)  # the hadamard product for column_sum of weight and  Z

        return coef_Matrix

    def assemble_matrix2Laplace_1D(self, X_input=None):
        """
        the output for hidden nodes with 2nd order  Derivative
        Args:
             X_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=X_input)  # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)             # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)
        sum2W_column = np.reshape(np.sum(squareW, axis=0), newshape=[1, -1])   # the sum of column for weight matrix

        coef_Matrix = np.multiply(act_out2hidden, sum2W_column)  # the hadamard product for column_sum of weight and  Z

        return coef_Matrix

    def assemble_matrix2BiLaplace_1D(self, X_input=None):
        """
        the output for hidden nodes with 2nd order  Derivative
        Args:
             X_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=X_input)  # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)             # Z = sigma''(Y), this is a matrix
        fourOrder_out2hidden = self.act_func2DDDDeri(hidden_linear)  # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)

        coef_Matrix2SecondOrder_X = np.multiply(act_out2hidden, squareW)

        square_squareW = np.square(np.square(self.W2Hidden))
        sum2W_column = np.reshape(np.sum(square_squareW, axis=0), newshape=[1, -1])   # the sum of column for weight matrix

        coef_Matrix2fourOrder = np.multiply(fourOrder_out2hidden, sum2W_column)  # the hadamard product for column_sum of weight and  Z

        coef_Matrix = coef_Matrix2fourOrder + coef_Matrix2SecondOrder_X

        return coef_Matrix

    def assemble_matrix2interior_2D(self, XY_input=None):
        """
        the output for hidden nodes by operator sigma(x·W+B)
        Args:
             XY_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XY_input)    # Y = x·W+B
        act_out2hidden = self.act_func2hidden(hidden_linear)               # Z = sigma(Y), this is a matrix
        return act_out2hidden

    def assemble_matrix2boundary_2D(self, XY_input=None):
        """
        the output for hidden nodes by operator sigma(x·W+B)
        Args:
             XY_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XY_input)   # Y = x·W+B
        act_out2hidden = self.act_func2hidden(hidden_linear)              # Z = sigma(Y), this is a matrix
        return act_out2hidden

    def assemble_matrix2First_Derivative_2D_BD(self, XY_input=None, type2boundary='left'):
        """
        the output for hidden nodes with 1st order  Derivative
        Args:
             XY_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XY_input)    # Y = x·W+B
        act_out2hidden = self.act_func2Deri(hidden_linear)                 # Z = sigma'(Y), this is a matrix

        if type2boundary == 'left':
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])          # this is a vector
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)            # the hadamard product of weight and Z
        elif type2boundary == 'right':
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == 'bottom':
            W = np.reshape(self.W2Hidden[1, :], newshape=[1, -1])
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == 'top':
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        return coef_Matrix

    def assemble_matrix2First_Derivative_2D(self, XY_input=None, axis=0):
        """
        the output for hidden nodes with 1st order  Derivative
        Args:
             XY_input: the input points
             axis: direct to the derivative for inviable (x=0 or y=1)
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XY_input)    # Y = x·W+B
        act_out2hidden = self.act_func2Deri(hidden_linear)                 # Z = sigma'(Y), this is a matrix

        if axis == 0:
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])          # this is a vector
        else:
            W = np.reshape(self.W2Hidden[1, :], newshape=[1, -1])  # this is a vector

        coef_Matrix = np.multiply(W, act_out2hidden)            # the hadamard product of weight and Z
        return coef_Matrix

    def assemble_matrix2Second_Derivative_2D(self, XY_input=None, axis=0):
        """
        the output for hidden nodes with 2nd order  Derivative
        Args:
             XY_input: the input points
             axis: direct to the derivative for inviable (x=0 or y=1)
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XY_input)         # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)                     # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)

        if axis == 0:
            W = np.reshape(squareW[0, :], newshape=[1, -1])
        else:
            W = np.reshape(squareW[1, :], newshape=[1, -1])

        coef_Matrix = np.multiply(act_out2hidden, W)    # the hadamard product for column_sum of weight and Z

        return coef_Matrix

    def assemble_matrix2Laplace_2D(self, XY_input=None):
        """
        the output for hidden nodes with 2nd order  Derivative
        Args:
             XY_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XY_input)         # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)                     # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)
        sum2W_column = np.reshape(np.sum(squareW, axis=0), newshape=[1, -1])    # the sum of column for weight matrix

        coef_Matrix = np.multiply(act_out2hidden, sum2W_column)    # the hadamard product for column_sum of weight and Z

        return coef_Matrix

    def assemble_matrix2BiLaplace_2D(self, XY_input=None):
        """
        the output for hidden nodes with 2nd order  Derivative
        Args:
             XY_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XY_input)         # Y = x·W+B
        SecondOrder_out2hidden = self.act_func2DDeri(hidden_linear)
        FourOrder_out2hidden = self.act_func2DDDDeri(hidden_linear)                     # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)
        W2X = np.reshape(squareW[0, :], newshape=[1, -1])
        W2Y = np.reshape(squareW[1, :], newshape=[1, -1])

        coef_Matrix2X = np.multiply(SecondOrder_out2hidden, W2X)
        coef_Matrix2Y = np.multiply(SecondOrder_out2hidden, W2Y)

        square_squareW = np.square(np.square(self.W2Hidden))
        sum2W_column = np.reshape(np.sum(square_squareW, axis=0), newshape=[1, -1])    # the sum of column for weight matrix

        coef_Matrix2FourOrder = np.multiply(FourOrder_out2hidden, sum2W_column)    # the hadamard product for column_sum of weight and Z

        coef_Matrix = coef_Matrix2FourOrder + 2.0*np.multiply(coef_Matrix2X, coef_Matrix2Y)

        return coef_Matrix

    def assemble_matrix2interior_3D(self, XYZ_input=None):
        """
        the output for hidden nodes by operator sigma(x·W+B)
        Args:
             XYZ_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZ_input)   # Y = x·W+B
        act_out2hidden = self.act_func2hidden(hidden_linear)               # Z = sigma(Y), this is a matrix
        return act_out2hidden

    def assemble_matrix2boundary_3D(self, XYZ_input=None):
        """
        the output for hidden nodes by operator sigma(x·W+B)
        Args:
             XYZ_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZ_input)
        act_out2hidden = self.act_func2hidden(hidden_linear)
        return act_out2hidden

    def assemble_matrix2First_Derivative_3D_BD(self, XYZ_input=None, type2boundary='left'):
        """
        the output for hidden nodes with 1st order  Derivative
        Args:
             XYZ_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZ_input)   # Y = x·W+B
        act_out2hidden = self.act_func2Deri(hidden_linear)                 # Z = sigma'(Y), this is a matrix

        if type2boundary == 'left':
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])          # this is a vector
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)            # the hadamard product of weight and Z
        elif type2boundary == 'right':
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == 'bottom':
            W = np.reshape(self.W2Hidden[2, :], newshape=[1, -1])
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == 'top':
            W = np.reshape(self.W2Hidden[2, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == 'front':
            W = np.reshape(self.W2Hidden[1, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == 'behind':
            W = np.reshape(self.W2Hidden[1, :], newshape=[1, -1])
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)
        return coef_Matrix

    def assemble_matrix2First_Derivative_3D(self, XYZ_input=None, axis=0):
        """
        the output for hidden nodes with 1st order  Derivative
        Args:
             XYZ_input: the input points
             axis: direct to the derivative for inviable (x=0 or y=1 or z=2)
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZ_input)   # Y = x·W+B
        act_out2hidden = self.act_func2Deri(hidden_linear)                 # Z = sigma'(Y), this is a matrix

        if axis == 0:
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])          # this is a vector
        elif axis == 1:
            W = np.reshape(self.W2Hidden[1, :], newshape=[1, -1])
        else:
            W = np.reshape(self.W2Hidden[2, :], newshape=[1, -1])
        coef_Matrix = np.multiply(W, act_out2hidden)            # the hadamard product of weight and Z
        return coef_Matrix

    def assemble_matrix2Second_Derivative_3D(self, XYZ_input=None, axis=0):
        """
        the output for hidden nodes with 2nd order  Derivative
        Args:
             XYZ_input: the input points
             axis: direct to the derivative for inviable (x=0 or y=1 or z=2)
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZ_input)    # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)                 # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)
        if axis == 0:
            W = np.reshape(squareW[0, :], newshape=[1, -1])
        elif axis == 1:
            W = np.reshape(squareW[1, :], newshape=[1, -1])
        else:
            W = np.reshape(squareW[2, :], newshape=[1, -1])

        coef_Matrix = np.multiply(act_out2hidden, W)   # the hadamard product for column_sum of weight and Z

        return coef_Matrix

    def assemble_matrix2Laplace_3D(self, XYZ_input=None):
        """
        the output for hidden nodes with 2nd order Derivative
        Args:
             XYZ_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZ_input)    # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)                 # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)
        sum2W_column = np.reshape(np.sum(squareW, axis=0), newshape=[1, -1])   # the sum of column for weight matrix

        coef_Matrix = np.multiply(act_out2hidden, sum2W_column)   # the hadamard product for column_sum of weight and Z

        return coef_Matrix

    def assemble_matrix2BiLaplace_3D(self, XYZ_input=None):
        """
        the output for hidden nodes with 2nd order Derivative
        Args:
             XYZ_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZ_input)    # Y = x·W+B
        SecondOrder_out2hidden = self.act_func2DDeri(hidden_linear)
        FourOrder_out2hidden = self.act_func2DDDDeri(hidden_linear)  # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)
        W2X = np.reshape(squareW[0, :], newshape=[1, -1])
        W2Y = np.reshape(squareW[1, :], newshape=[1, -1])
        W2Z = np.reshape(squareW[2, :], newshape=[1, -1])

        coef_Matrix2X = np.multiply(SecondOrder_out2hidden, W2X)
        coef_Matrix2Y = np.multiply(SecondOrder_out2hidden, W2Y)
        coef_Matrix2Z = np.multiply(SecondOrder_out2hidden, W2Z)

        square_squareW = np.square(np.square(self.W2Hidden))
        sum2W_column = np.reshape(np.sum(square_squareW, axis=0), newshape=[1, -1])   # the sum of column for weight matrix

        coef_Matrix2FourOrder = np.multiply(FourOrder_out2hidden, sum2W_column)    # the hadamard product for column_sum of weight and Z

        coef_Matrix = coef_Matrix2FourOrder + 2.0 * np.multiply(coef_Matrix2X, coef_Matrix2Y) + \
                      2.0 * np.multiply(coef_Matrix2X, coef_Matrix2Z) + \
                      2.0 * np.multiply(coef_Matrix2Y, coef_Matrix2Z)

        return coef_Matrix

    def assemble_matrix2interior_4D(self, XYZS_input=None):
        """
        the output for hidden nodes by operator sigma(x·W+B)
        Args:
             XYZS_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZS_input)   # Y = x·W+B
        act_out2hidden = self.act_func2hidden(hidden_linear)               # Z = sigma(Y), this is a matrix
        return act_out2hidden

    def assemble_matrix2boundary_4D(self, XYZS_input=None):
        """
        the output for hidden nodes by operator sigma(x·W+B)
        Args:
             XYZS_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZS_input)
        act_out2hidden = self.act_func2hidden(hidden_linear)
        return act_out2hidden

    def assemble_matrix2First_Derivative_4D_BD(self, XYZS_input=None, type2boundary='left'):
        """
        the output for hidden nodes with 1st order  Derivative
        Args:
             XYZS_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZS_input)   # Y = x·W+B
        act_out2hidden = self.act_func2Deri(hidden_linear)                 # Z = sigma'(Y), this is a matrix

        if type2boundary == '00':
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])          # this is a vector
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)            # the hadamard product of weight and Z
        elif type2boundary == '01':
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '10':
            W = np.reshape(self.W2Hidden[1, :], newshape=[1, -1])
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '11':
            W = np.reshape(self.W2Hidden[1, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '20':
            W = np.reshape(self.W2Hidden[2, :], newshape=[1, -1])
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '21':
            W = np.reshape(self.W2Hidden[2, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '30':
            W = np.reshape(self.W2Hidden[3, :], newshape=[1, -1])
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '31':
            W = np.reshape(self.W2Hidden[3, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        return coef_Matrix

    def assemble_matrix2First_Derivative_4D(self, XYZS_input=None, axis=0):
        """
        the output for hidden nodes with 1st order  Derivative
        Args:
             XYZS_input: the input points
             axis: direct to the derivative for inviable (x=0 or y=1 or z=2)
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZS_input)   # Y = x·W+B
        act_out2hidden = self.act_func2Deri(hidden_linear)                 # Z = sigma'(Y), this is a matrix

        if axis == 0:
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])          # this is a vector
        elif axis == 1:
            W = np.reshape(self.W2Hidden[1, :], newshape=[1, -1])
        elif axis == 2:
            W = np.reshape(self.W2Hidden[2, :], newshape=[1, -1])
        elif axis == 3:
            W = np.reshape(self.W2Hidden[3, :], newshape=[1, -1])
        coef_Matrix = np.multiply(W, act_out2hidden)            # the hadamard product of weight and Z
        return coef_Matrix

    def assemble_matrix2Second_Derivative_4D(self, XYZS_input=None, axis=0):
        """
        the output for hidden nodes with 2nd order  Derivative
        Args:
             XYZS_input: the input points
             axis: direct to the derivative for inviable (x=0 or y=1 or z=2)
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZS_input)    # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)                 # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)
        if axis == 0:
            W = np.reshape(squareW[0, :], newshape=[1, -1])
        elif axis == 1:
            W = np.reshape(squareW[1, :], newshape=[1, -1])
        elif axis == 2:
            W = np.reshape(squareW[2, :], newshape=[1, -1])
        elif axis == 3:
            W = np.reshape(squareW[3, :], newshape=[1, -1])

        coef_Matrix = np.multiply(act_out2hidden, W)   # the hadamard product for column_sum of weight and Z

        return coef_Matrix

    def assemble_matrix2Laplace_4D(self, XYZS_input=None):
        """
        the output for hidden nodes with 2nd order Derivative
        Args:
             XYZS_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZS_input)    # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)                 # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)
        sum2W_column = np.reshape(np.sum(squareW, axis=0), newshape=[1, -1])   # the sum of column for weight matrix

        coef_Matrix = np.multiply(act_out2hidden, sum2W_column)   # the hadamard product for column_sum of weight and Z

        return coef_Matrix

    def assemble_matrix2BiLaplace_4D(self, XYZS_input=None):
        """
        the output for hidden nodes with 2nd order Derivative
        Args:
             XYZS_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZS_input)    # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)                 # Z = sigma''(Y), this is a matrix

        square_squareW = np.square(np.square(self.W2Hidden))
        sum2W_column = np.reshape(np.sum(square_squareW, axis=0), newshape=[1, -1])   # the sum of column for weight matrix

        coef_Matrix = np.multiply(act_out2hidden, sum2W_column)   # the hadamard product for column_sum of weight and Z

        return coef_Matrix

    def assemble_matrix2interior_5D(self, XYZST_input=None):
        """
        the output for hidden nodes by operator sigma(x·W+B)
        Args:
             XYZST_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZST_input)   # Y = x·W+B
        act_out2hidden = self.act_func2hidden(hidden_linear)               # Z = sigma(Y), this is a matrix
        return act_out2hidden

    def assemble_matrix2boundary_5D(self, XYZST_input=None):
        """
        the output for hidden nodes by operator sigma(x·W+B)
        Args:
             XYZST_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZST_input)
        act_out2hidden = self.act_func2hidden(hidden_linear)
        return act_out2hidden

    def assemble_matrix2First_Derivative_5D_BD(self, XYZST_input=None, type2boundary='left'):
        """
        the output for hidden nodes with 1st order  Derivative
        Args:
             XYZST_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZST_input)   # Y = x·W+B
        act_out2hidden = self.act_func2Deri(hidden_linear)                 # Z = sigma'(Y), this is a matrix

        if type2boundary == '00':
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])          # this is a vector
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)            # the hadamard product of weight and Z
        elif type2boundary == '01':
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '10':
            W = np.reshape(self.W2Hidden[1, :], newshape=[1, -1])
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '11':
            W = np.reshape(self.W2Hidden[1, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '20':
            W = np.reshape(self.W2Hidden[2, :], newshape=[1, -1])
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '21':
            W = np.reshape(self.W2Hidden[2, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '30':
            W = np.reshape(self.W2Hidden[3, :], newshape=[1, -1])
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '31':
            W = np.reshape(self.W2Hidden[3, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '40':
            W = np.reshape(self.W2Hidden[4, :], newshape=[1, -1])
            coef_Matrix = -1.0 * np.multiply(W, act_out2hidden)
        elif type2boundary == '41':
            W = np.reshape(self.W2Hidden[4, :], newshape=[1, -1])
            coef_Matrix = 1.0 * np.multiply(W, act_out2hidden)
        return coef_Matrix

    def assemble_matrix2First_Derivative_5D(self, XYZST_input=None, axis=0):
        """
        the output for hidden nodes with 1st order  Derivative
        Args:
             XYZST_input: the input points
             axis: direct to the derivative for inviable (x=0 or y=1 or z=2)
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZST_input)   # Y = x·W+B
        act_out2hidden = self.act_func2Deri(hidden_linear)                 # Z = sigma'(Y), this is a matrix

        if axis == 0:
            W = np.reshape(self.W2Hidden[0, :], newshape=[1, -1])          # this is a vector
        elif axis == 1:
            W = np.reshape(self.W2Hidden[1, :], newshape=[1, -1])
        elif axis == 2:
            W = np.reshape(self.W2Hidden[2, :], newshape=[1, -1])
        elif axis == 3:
            W = np.reshape(self.W2Hidden[3, :], newshape=[1, -1])
        elif axis == 4:
            W = np.reshape(self.W2Hidden[4, :], newshape=[1, -1])
        coef_Matrix = np.multiply(W, act_out2hidden)            # the hadamard product of weight and Z
        return coef_Matrix

    def assemble_matrix2Second_Derivative_5D(self, XYZST_input=None, axis=0):
        """
        the output for hidden nodes with 2nd order  Derivative
        Args:
             XYZST_input: the input points
             axis: direct to the derivative for inviable (x=0 or y=1 or z=2)
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZST_input)    # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)                 # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)
        if axis == 0:
            W = np.reshape(squareW[0, :], newshape=[1, -1])
        elif axis == 1:
            W = np.reshape(squareW[1, :], newshape=[1, -1])
        elif axis == 2:
            W = np.reshape(squareW[2, :], newshape=[1, -1])
        elif axis == 3:
            W = np.reshape(squareW[3, :], newshape=[1, -1])
        elif axis == 4:
            W = np.reshape(squareW[4, :], newshape=[1, -1])

        coef_Matrix = np.multiply(act_out2hidden, W)   # the hadamard product for column_sum of weight and Z

        return coef_Matrix

    def assemble_matrix2Laplace_5D(self, XYZST_input=None):
        """
        the output for hidden nodes with 2nd order Derivative
        Args:
             XYZST_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZST_input)    # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)                 # Z = sigma''(Y), this is a matrix

        squareW = np.square(self.W2Hidden)
        sum2W_column = np.reshape(np.sum(squareW, axis=0), newshape=[1, -1])   # the sum of column for weight matrix

        coef_Matrix = np.multiply(act_out2hidden, sum2W_column)   # the hadamard product for column_sum of weight and Z

        return coef_Matrix

    def assemble_matrix2BiLaplace_5D(self, XYZST_input=None):
        """
        the output for hidden nodes with 2nd order Derivative
        Args:
             XYZST_input: the input points
        return:
             out2hidden: the output
        """
        hidden_linear = self.get_linear_out2Hidden(input_data=XYZST_input)    # Y = x·W+B
        act_out2hidden = self.act_func2DDeri(hidden_linear)                 # Z = sigma''(Y), this is a matrix

        square_squareW = np.square(np.square(self.W2Hidden))
        sum2W_column = np.reshape(np.sum(square_squareW, axis=0), newshape=[1, -1])   # the sum of column for weight matrix

        coef_Matrix = np.multiply(act_out2hidden, sum2W_column)   # the hadamard product for column_sum of weight and Z

        return coef_Matrix
