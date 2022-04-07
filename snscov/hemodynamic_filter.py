import numpy as np
import matplotlib.pyplot as plt
import math


def Poisson_function(lam, t):
    x = np.arange(t)
    d = np.array([math.factorial(i) for i in x])
    n = np.array([np.power(lam, i)  for i in x])
    y =  (n / d) * np.exp(-lam)
    return y

def Gamma_function(alpha, beta, t):
    x = np.arange(t)
    x = np.power(x, alpha-1) * np.exp(-beta * x)
    
    Ga = math.factorial(alpha-1)
    
    y = np.power(beta, alpha) * x / Ga
    return y
    
if __name__ == "__main__":
    alpha = [1,2,3,5,9,7,]
    beta =  [1,1,1,1,2,1]
    for i in range(len(alpha)):
        plt.plot(Gamma_function(alpha[i], beta[i], 30),marker='o', label='$\\alpha={}, \\beta={}$'.format(alpha[i], beta[i]))

    plt.xlabel('t (sec)')
    plt.title('Gamma Filter')
    plt.legend()
    plt.show()

