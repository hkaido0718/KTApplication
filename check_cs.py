
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib 
import time
from scipy import optimize
from sklearn.linear_model import LogisticRegression
from numpy.linalg import inv
import ray
import sys
from numba import jit
import ctypes as ct
from timeit import default_timer as timer 

#pts = np.load("app_conf_set.npz")
pts = np.load("app_conf_set_disc.npz")
thetagrid = pts["thetagrid"]
ST = pts["ST"]

def get_CS(thetagrid,T,cv):
    idx = (T <= cv) 
    theta_selected = thetagrid[idx,:]
    return theta_selected,idx

def Contour_L_Delta(X, Y, Z, selected_delta,filename):
    fig, ax = plt.subplots()
    levels = [30,40,50,60,80,100,150,200,300,400]
    # levels = [1.286,1.29,1.3,1.35,1.4,1.5,1.6,1.7]
    # if dgp == 1:
    #     levels = [1.211,1.215,1.23,1.26,1.29,1.32, 1.35,1.4,1.45]
    # elif dgp == 2:
    #     levels = [1.154,1.16,1.18,1.2,1.23,1.26,1.29,1.32] # Levels for DGP2
    # elif dgp == 3:
    #     levels = [1.16,1.164,1.18,1.2,1.23,1.26,1.29,1.33, 1.36,1.39]
    # elif dgp == 4:
    #     levels = [1.09,1.1,1.12,1.14,1.16,1.18,1.2,1.22,1.24]
    CS = ax.contour(X, Y, Z,levels)
    #ax.plot(selected_delta[:,0], selected_delta[:,1]) # identified set
    mkstyle = matplotlib.markers.MarkerStyle(marker="h", fillstyle="full")
    ax.scatter(selected_delta[:,0], selected_delta[:,1],s=2.5,c="tab:red",marker=mkstyle)  # identified set (if it is a single point)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Contour Plot of Concentrated Objective Function')
    ax.set_xlabel(r"$\Delta^{(1)}$")
    ax.set_ylabel(r"$\Delta^{(2)}$", rotation="horizontal")
    ax.set_ylim(-2, 0)
    ax.set_xlim(-2, 0)
    # ax.set_ylim(-2, 0)
    # ax.set_xlim(0,2)

    plt.savefig(filename)

cv = chi2.ppf(.95,9)
numgrid = np.int(np.sqrt(ST.shape[0]))
T = np.array([ST[i][9] for i in range(numgrid**2)])
thetasel,idx=get_CS(thetagrid,T,cv)

# numgrid = 25
xx = np.linspace(-2, 0, numgrid) # grid for Delta1 
yy = np.linspace(-2, 0, numgrid) # grid for Delta2
# xx = np.linspace(0,2, numgrid) # grid for Delta1 
# yy = np.linspace(-2, 0, numgrid) # grid for Delta2
Delta1, Delta2 = np.meshgrid(xx, yy)
Z = T.reshape((numgrid,numgrid))
Contour_L_Delta(Delta1, Delta2, Z, thetasel,"temp")

fig, ax = plt.subplots()
mkstyle = matplotlib.markers.MarkerStyle(marker="o", fillstyle="full")
ax.scatter(thetasel[:,0],thetasel[:,1],s=10,c="tab:blue",marker=mkstyle)  
ax.set_title('Confidence Set')
ax.set_xlabel(r"$\Delta^{(1)}$")
ax.set_ylabel(r"$\Delta^{(2)}$", rotation="horizontal")
# ax.set_ylim(-2.5, 0)
# ax.set_xlim(-2.5, 0)
ax.set_ylim(-2, 0)
ax.set_xlim(0,2)

figfilename = "confset_posneg.pdf"
plt.savefig(figfilename)