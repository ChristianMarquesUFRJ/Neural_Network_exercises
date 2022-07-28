from math import exp
import numpy as np

##########################################
# FUNCOES DE ATIVACAO POSSIVEIS
##########################################
functions = ['limiar', 'degrau', 'sigmoide', 'tg_hip', 'linear', 'linear_partes', 'retificadora', 'soft_plus', 'leaky', 'swish']

##########################################
# FUNCOES AUXILIARES
##########################################
def get_rows(mat):
    return len(mat);

def get_cols(mat):
    return len(mat[0]);

def format_vector(v):
    return ['%.4f' % elem for elem in v]

def sum(x_, w_):
    return np.dot(x_, w_)

def apply_function(y, depth_functions_, depth, k):
    if (depth_functions_[depth] == functions.index('limiar')):
        if (y>=k): return 1
        return 0

    elif (depth_functions_[depth] == functions.index('degrau')):
        if (y<k): return 10
        elif (y==k): return 20
        return 30

    elif (depth_functions_[depth] == functions.index('sigmoide')):
        return 1/(1+exp(-k*y))

    elif (depth_functions_[depth] == functions.index('tg_hip')):
        return (exp(2*k*y)-1)/(exp(2*k*y)+1)

    elif (depth_functions_[depth] == functions.index('linear')):
        return k*y + 0

    elif (depth_functions_[depth] == functions.index('linear_partes')):
        l = 10
        if (y<=k): return 10
        elif ((y<k) and (y<l)): return (y-k)/(l-k)
        return 20

    elif (depth_functions_[depth] == functions.index('retificadora')):
        if (y<0): return 0
        return y

    elif (depth_functions_[depth] == functions.index('soft_plus')):
        return np.ln(1+exp(k*y))/k

    elif (depth_functions_[depth] == functions.index('leaky')):
        if (y<0): return k*y
        return y

    elif (depth_functions_[depth] == functions.index('swish')):
        return y/(1+exp(-k*y))
    
    return y

def get_NN_output(bias_, x_, w_, depth_functions_):
    x = x_
    y=[]
    n=1
    depth_layers = get_rows(bias_) # Obtem a quantidade de camada profundas
    complete_out = []

    for d in range(depth_layers):
        x = np.insert(x, 0, bias_[d]) # insere os bias no comeco da entrada
        layer_neurons_number = get_rows(w_[d]) # Obtem a quantidade de neuronios por camada
        y = [0]*layer_neurons_number # Cria o vetor de saida dos neuronios
        for i in range(layer_neurons_number):
            y_ = sum(x, w_[d][i]) # faz o somatorio ponderado
            y[i] = apply_function(y_, depth_functions_, d, n)
            complete_out.append(y[i])
            # print("Yo[", n, "]={:.4f}".format(y_),"Y[", n, "]={:.4f}".format(y[i]), " | Entrada: ", format_vector(x), " | Pesos: ", format_vector(w[d][i]))
            n=n+1
        # print("")
        x = y # Joga a saida dos neuronio no vetor de entrada

    return complete_out

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
    w = [w1, w2, w3]

    # Funcoes de ativacao de cada camada
    # ---
    # QUESTÃO 1)A)
    depth_functions = [functions.index('retificadora'), functions.index('retificadora'), functions.index('retificadora')]
    # QUESTÃO 1)B)
    # depth_functions = [functions.index('sigmoide'), functions.index('tg_hip'), functions.index('linear')]

    ##########################################
    # GERAL
    ##########################################
    # Obtencao da saida da RN
    # ---
    y = get_NN_output(bias, x, w, depth_functions)

    # Mostra o resultado da saida formatado
    # ---
    out = y[5], y[6]
    print("Resposta final: ", format_vector(out))