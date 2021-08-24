import random
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
'''declaration---------------'''

'''declare parameters'''

N = 10
Niter = 1
spin = [-1, 1]
tempstep = 300
temp = np.linspace(1, 4, tempstep)
warmup = 1000
measureinter = 100
measureiter = 1000


Ebar, Esquare, Mbar, Msquare, C, X = np.zeros(tempstep), np.zeros(tempstep), np.zeros(
    tempstep), np.zeros(tempstep), np.zeros(tempstep), np.zeros(tempstep)

Etot = []
Mtot = []
Ctot = []
Xtot = []


'''spin lattice'''


def initialState(N):
    state = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            state[i, j] = random.choice(spin)
    Ebar, Esquare, Mbar, Msquare, C, X = np.zeros(tempstep), np.zeros(tempstep), np.zeros(
        tempstep), np.zeros(tempstep), np.zeros(tempstep), np.zeros(tempstep)
    return state


'''moving by Metropolis Algorithm'''


def mcmoving(state, N, tempstep):
    latx = random.randrange(0, N)
    laty = random.randrange(0, N)
    x1 = latx-1
    x2 = latx+1
    x3 = latx
    x4 = latx
    y1 = laty
    y2 = laty
    y3 = laty-1
    y4 = laty+1
    if latx == N-1:
        x2 = 0
    elif latx == 0:
        x1 = N-1
    if laty == N-1:
        y4 = 0
    elif laty == 0:
        y3 = N-1
    origin = state[latx, laty]
    fliped = -1*origin
    deltaE = -(state[x1, y1]+state[x2, y2] +
               state[x3, y3]+state[x4, y4])*(fliped-origin)

    if deltaE < 0:
        state[latx, laty] = fliped
    else:
        if random.random() < np.exp(-deltaE/temp[tempstep]):
            state[latx, laty] = fliped
    return state


'''calculate energy of a given configuration'''


def calcE(state, N):
    energy = 0
    for i in range(N-1):
        for j in range(N-1):
            energy += -state[i, j]*state[i+1, j] - state[i, j]*state[i, j+1]
    return energy


'''calculate magnetization of a given config'''


def calcM(state, N):
    magnet = 0
    for i in range(N):
        for j in range(N):
            magnet += state[i, j]
    return magnet


'''measurement per 100 move'''


def measure(lattice, N, t):
    for i in range(measureinter):
        lattice = mcmoving(lattice, N, t)
    Ebar[t] += calcE(lattice, N)
    Esquare[t] += calcE(lattice, N)*calcE(lattice, N)
    Mbar[t] += np.absolute(calcM(lattice, N))
    Msquare[t] += calcM(lattice, N)*calcM(lattice, N)


'''------------------------'''


'''main part'''
for b in range(Niter):
    N = 2*N
    lattice = initialState(N)
    for t in range(tempstep-1, -1, -1):
        for j in range(warmup):
            lattice = mcmoving(lattice, N, t)
        for k in range(measureiter):
            measure(lattice, N, t)
        C[t] = ((Esquare[t]/measureiter)-(Ebar[t]/measureiter)**2) / \
            (temp[t]*temp[t]*N*N)
        X[t] = ((Msquare[t]/measureiter) -
                (Mbar[t]/measureiter)**2)/(temp[t]*N*N)
        Ebar[t] = Ebar[t]/((N*N)*measureiter)
        Mbar[t] = np.absolute(Mbar[t])/((N*N)*measureiter)
    Etot.append(copy(Ebar))
    Mtot.append(copy(Mbar))
    Ctot.append(copy(C))
    Xtot.append(copy(X))


'''
print("Energy per lattice ",Etot)
print("Magnetization per lattice",Mtot)
print("Heat Capacity per lattice",Ctot)
print("Magnetic Susceptibility",Xtot)
print(temp)
print(lattice)
'''

f = plt.figure(figsize=(13, 7))  # plot the calculated values

sp = f.add_subplot(2, 2, 1)
for b in range(Niter):
    plt.plot(temp, Etot[b], label=f"lattice size: {N*2**(b-1)}")
plt.xlabel("Temperature", fontsize=10)
plt.ylabel("Energy", fontsize=10)
plt.legend(loc='upper left')

sp = f.add_subplot(2, 2, 2)
for b in range(Niter):
    plt.plot(temp, Mtot[b], label=f"lattice size: {N*2**(b-1)}")
plt.xlabel("Temperature", fontsize=10)
plt.ylabel("Magnetization ", fontsize=10)
plt.legend(loc='upper right')

sp = f.add_subplot(2, 2, 3)
for b in range(Niter):
    plt.plot(temp, Ctot[b], label=f"lattice size: {N*2**(b-1)}")
plt.xlabel("Temperature", fontsize=10)
plt.ylabel("Heat Capacity", fontsize=10)
plt.legend(loc='upper left')

sp = f.add_subplot(2, 2, 4)
for b in range(Niter):
    plt.plot(temp, Xtot[b], label=f"lattice size: {N*2**(b-1)}")
plt.xlabel("Temperature", fontsize=10)
plt.ylabel("Magnetic Susceptibility", fontsize=10)
plt.legend(loc='upper left')

plt.show()
