from q1_foward_propagation import *
from math import sqrt, pow, exp
from sklearn.metrics import mean_squared_error, accuracy_score

# Bias
# ---
bias=[1, 1, 1]

# Entradas
# ---
x=[0.3, -0.5]


# Pesos Iniciais
# ---
w2=[[0.3,   0.2,    0.0],
    [-0.2,  0.1,    -0.4]]
w3=[[0.3,   0.0,    0.1],
    [0.4,   0.0,    0.4],
    [0.1,   -0.3,   0.0]]  
w4=[[-0.3,  0.5,    0.2,    0.1],
    [0.4,   0.3,    0.2,    0.5]]
w = [w2, w3, w4]

# Pesos Apos o BackPropagation
# ---
w2_=[[0.3358,   0.2108,    -0.0179],
    [-0.1916,   0.1025,    -0.4042]]
w3_=[[-0.0299,      -0.1943,    -0.0699],
    [0.2634,        -0.0805,    0.4704],
    [-0.4204,       -0.6025,    -0.2680]]  
w4_=[[-0.8031,  0.1058,     -0.2953,    0.2840],
    [-0.6859,   -0.5507,    -0.8689,    0.8972]]
w_ = [w2_, w3_, w4_]

if __name__ == "__main__":
    rmse_2 = mean_squared_error(w2, w2_) 
    rmse_3 = mean_squared_error(w3, w3_) 
    rmse_4 = mean_squared_error(w4, w4_) 
    print("MSE_w2: {:.4f}".format(rmse_2))
    print("MSE_w3: {:.4f}".format(rmse_3))
    print("MSE_w4: {:.4f}".format(rmse_4))