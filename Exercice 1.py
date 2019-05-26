# Exercice 1

# Fonction calcul voisins:

def calcul_nb_voisins(Z):
    forme = len(Z), len(Z[0])
    N = [[0, ] * (forme[0]) for i in range(forme[1])]
    for x in range(1, forme[0] - 1):
        for y in range(1, forme[1] - 1):
            N[x][y] = Z[x-1][y-1]+Z[x][y-1]+Z[x+1][y-1] \
            + Z[x-1][y] + 0 +Z[x+1][y] \
            + Z[x-1][y+1]+Z[x][y+1]+Z[x+1][y+1]
    return N

# Fonction iteration jeu:

def iteration_jeu(Z):
    forme = len(Z), len(Z[0])
    N = calcul_nb_voisins(Z)
    for x in range(1,forme[0]-1):
        for y in range(1,forme[1]-1):
            if Z[x][y] == 1 and (N[x][y] < 2 or N[x][y] > 3):
                Z[x][y] = 0
            elif Z[x][y] == 0 and N[x][y] == 3:
                Z[x][y] = 1
    return Z

# 10 iteration:

%matplotlib inline
Z = [[0,0,0,0,0,0],
    [0,0,0,1,0,0],
    [0,1,0,1,0,0],
    [0,0,1,1,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0]]
import numpy as np
import matplotlib.pyplot as plt
import time
liste=list()
for i in range(10):
    plt.subplot(2,5,i+1)
    liste.append(iteration_jeu(Z))
    print('Z',i,'=', np.matrix(Z))
    plt.imshow(liste[i])

# numba

import numpy as np
import time
from numba import jit
@jit
def iteration_jeu_numba(Z):
    forme = len(Z), len(Z[0])
    N = calcul_nb_voisins(Z)
    for x in range(1,forme[0]-1):
        for y in range(1,forme[1]-1):
            if Z[x][y] == 1 and (N[x][y] < 2 or N[x][y] > 3):
                Z[x][y] = 0
            elif Z[x][y] == 0 and N[x][y] == 3:
                Z[x][y] = 1
    return Z

start = time.time()
iteration_jeu_numba(Z)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))
start = time.time()
iteration_jeu_numba(Z)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

#@jit
def calcul_nb_voisins_numba(Z):
    forme = len(Z), len(Z[0])
    N = [[0, ] * (forme[0]) for i in range(forme[1])]
    for x in range(1, forme[0] - 1):
        for y in range(1, forme[1] - 1):
            N[x][y] = Z[x-1][y-1]+Z[x][y-1]+Z[x+1][y-1] \
                + Z[x-1][y] + 0 +Z[x+1][y] \
                + Z[x-1][y+1]+Z[x][y+1]+Z[x+1][y+1]
    return N

start = time.time()
calcul_nb_voisins_numba(Z)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))
start = time.time()
calcul_nb_voisins_numba(Z)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

# Fonction iter:

def Iiter(n):
    plt.figure(figsize=(25,20))
    Zwidg=np.copy(Z_np)
    for i in range (n):
        plt.subplot(1,5,1)
        plt.imshow(np.array(Zwidg))
        Zbis = iteration_jeu(Zwidg)

# Widget
from ipywidgets import interact, fixed
interact(Iiter,n=(1,30,1));