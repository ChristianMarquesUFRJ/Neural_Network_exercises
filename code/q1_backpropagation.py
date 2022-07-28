from q1_foward_propagation import *
from math import sqrt, pow, exp

eta = 0.2
x = [0.3,-0.5]
y_target = [-0.5, 0.2]
tgh_a = 1.0
linear_a = 1.0
linear_b = 0.0

def sigmoid(y, a):
    return 1/(1+exp(-a*y))

def tg_hip(y, a):
    return (exp(2*a*y)-1)/(exp(2*a*y)+1)

def linear(y):
    return (linear_a*y)+linear_b

def get_error(real, target):
    size = len(real)
    error = [0]*size
    for i in range(size):
        error[i] = target[i] - real[i]
    return error

def get_derivate_f(y, id, depth_function):
    if (depth_function == functions.index('sigmoide')):
        sig = sigmoid(y, id)
        return sig*(1-sig)
    elif (depth_function == functions.index('tg_hip')):
        return 1-pow(tg_hip(y),2)
    elif (depth_function == functions.index('linear')):
        return linear_a
    

def get_layer_gradient(y, error, depth_function, start_id):
    size = len(y)
    layer_gradient = [0]*size
    for i in range(size):
        layer_gradient[i] = get_derivate_f(y[i], start_id+i, depth_function)*error[i]
    return layer_gradient

##########################################
# CODE
##########################################
if __name__ == "__main__":
    ##########################################
    # CONFIGURACAO DA REDE
    ##########################################
    # Bias
    # ---
    bias=[1, 1, 1]

    # Entradas
    # ---
    x=[0.3, -0.5]

    # Pesos
    # ---
    w1=[[0.3,   0.2,    0.0],
        [-0.2,  0.1,    -0.4]]
    w2=[[0.3,   0.0,    0.1],
        [0.4,   0.0,    0.4],
        [0.1,   -0.3,   0.0]]  
    w3=[[-0.3,  0.5,    0.2,    0.1],
        [0.4,   0.3,    0.2,    0.5]]
    w_real = [w1, w2, w3]

    # Funcoes de ativacao de cada camada
    # ---
    depth_functions_1a = [functions.index('retificadora'), functions.index('retificadora'), functions.index('retificadora')]
    depth_functions_1b = [functions.index('sigmoide'), functions.index('tg_hip'), functions.index('linear')]

    y = get_NN_output(bias, x, w_real, depth_functions_1b)
    y_real = [y[5], y[6]]
    errors = get_error(y_real, y_target)
    layer_gradient = get_layer_gradient(y_real, errors, functions.index('linear'), 6)

    print("Y_real: ", str(y_real))
    print("Erros: ", str(errors))
    print("Gradiente: ", str(layer_gradient))