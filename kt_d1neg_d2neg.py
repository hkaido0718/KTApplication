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
from numpy.linalg import inv, pinv
import ray
import sys
from numba import jit, cuda
import ctypes as ct
from timeit import default_timer as timer 
from numpy.random import default_rng
rng = default_rng()
ind = int(sys.argv[1])
nprocs = int(sys.argv[2])
ray.init(num_cpus=nprocs)
# ray.init()

# so_file = "./TOMS462/toms462.so"
# lib = ct.CDLL(so_file)
# lib.bivnor.restype = ct.c_double
# lib.bivnor.argtypes = [ct.c_double, ct.c_double, ct.c_double]

def normalize_data(x):
    M = np.amax(x,axis=0)
    m = np.amin(x,axis=0)
    y = (x-m)/(M-m) # normalize to [0,1]
    y = 2*y-1   # normalize to [-1,1]
    return y


# This needs to be extended to accommodate 3 variables
def get_3Dtensorseries(x1,x2,x3,J):
    n = x1.shape[0]
    X = np.zeros((n,J**3))
    for j in range(J):
        for k in range(J):
            for l in range(J):
                c = np.zeros((J,J,J))
                c[j,k,l] = 1
                X[:,(j*J+k)*J+l] = np.polynomial.hermite_e.hermeval3d(x1, x2, x3, c)
    return X
        
def get_Xfrequency(X,xsupp):
    n = X.shape[0]
    K = xsupp.shape[0]
    freq = np.zeros(K)
    for i in range(n):
        for k in range(K):
            freq[k] = freq[k] + np.array_equal(X[i,:],xsupp[k])
    return freq

def get_XYfrequency(Y,X,ysupp,xsupp):
    n = X.shape[0]
    K = xsupp.shape[0]
    H = ysupp.shape[0]
    freq = np.zeros((K,H))
    for i in range(n):
        for k in range(K):
            for h in range(H):
                freq[k,h] = freq[k,h] + np.array_equal(X[i,:],xsupp[k])*np.array_equal(Y[i],ysupp[h])
    return freq

def get_ccp_pred(X,xsupp,ccpmat):
    n = X.shape[0]
    K = xsupp.shape[0]
    H = ccpmat.shape[1]
    ccp = np.zeros((n,H))
    for i in range(n):
        for k in range(K):
            if np.array_equal(X[i,:],xsupp[k]):
                ccp[i,:] = ccpmat[k,:]
    return ccp

def estimate_ccp(Y,X):
    # check DGP
    xfreq  = get_Xfrequency(X,xsupp)
    xfreqmat  = np.transpose(np.tile(xfreq,(4,1))) 
    xyfreq = get_XYfrequency(Y,X,ysupp,xsupp)
    ccpmat = np.divide(xyfreq,xfreqmat)
    ccp =  get_ccp_pred(X,xsupp,ccpmat)
    return ccp





# Load data
Data = np.loadtxt(open("./KlineTamerData/airlinedata.dat","rb"),delimiter=",")
Xlcc_pres = Data[:,0]
Xoa_pres  = Data[:,1]
Xsize     = Data[:,2]
y1 = Data[:,3]
y2 = Data[:,4]
n = Data.shape[0]
y = np.zeros(n)
y[(y1==1)*(y2==1)] = 11
y[(y1==0)*(y2==1)] = 1
y[(y1==1)*(y2==0)] = 10

# Normalize data
xlcc_pres = normalize_data(Xlcc_pres)
xoa_pres  = normalize_data(Xoa_pres)
xsize     = normalize_data(Xsize)

# Calculate median
med_lcc_pres = np.median(Xlcc_pres)
med_oa_pres  = np.median(Xoa_pres)
med_size     = np.median(Xsize)

# Y and X's support
ysupp = np.array([00,1,10,11])
xsupp = np.array([[1, 0, 0, 1, 0, 0],[1, 0, 0, 1, 1, 0],[1, 1, 0, 1, 0, 0],[1, 1, 0, 1, 1, 0], \
    [1, 0, 1, 1, 0, 1],[1, 0, 1, 1, 1, 1],[1, 1, 1, 1, 0, 1],[1, 1, 1, 1, 1, 1]]) 

# Estimate CCP
discretize = 1
if discretize == 0:
    xseries = get_3Dtensorseries(xlcc_pres,xoa_pres,xsize,2)
    clf = LogisticRegression(penalty='l2').fit(xseries, y)
    ccp = clf.predict_proba(xseries)
    x1 = np.column_stack((np.ones(n),xlcc_pres,xsize))
    x2 = np.column_stack((np.ones(n),xoa_pres,xsize))
    resfile = "app_conf_set_cts.npz" 
elif discretize ==1:
    x1 = np.column_stack((np.ones(n),1*(Xlcc_pres > med_lcc_pres),1*(Xsize>med_size)))
    x2 = np.column_stack((np.ones(n),1*(Xoa_pres > med_oa_pres),1*(Xsize>med_size)))
    x = np.column_stack((x1,x2))
    ccp = estimate_ccp(y,x)
    resfile = "app_conf_set_disc.npz"
phat_marginal = np.ones(n)/n



#@cuda.jit
def G(v,w,rho):
    try:    
        rv = sp.stats.multivariate_normal(mean=[0,0],cov=[[1,rho],[rho,1]])
    except np.linalg.LinAlgError as err:
        if 'singular matrix' in str(err):
            if rho > 0:
                rho = rho - 1e-9
            else:
                rho =  rho + 1e-9
        rv = sp.stats.multivariate_normal(mean=[0,0],cov=[[1,rho],[rho,1]])
    arg = np.dstack((v,w))
    val = rv.cdf(arg)
    return val

def gpdf(X,mu,sigma_rho):
    try:    
        rv = sp.stats.multivariate_normal(mean=mu,cov=sigma_rho)
    except np.linalg.LinAlgError as err:
        if 'singular matrix' in str(err):
            rho = sigma_rho[0,1]
            if rho > 0:
                rho = rho - 1e-9
            else:
                rho =  rho + 1e-9
            sigma_rho = np.array([[1,rho],[rho,1]])
        rv = sp.stats.multivariate_normal(mean=mu,cov=sigma_rho)
    val = rv.pdf(X)
    return val

# @jit
# def G(v,w,rho):
#     ah = -v
#     ak = -w
#     n = ah.shape[0]
#     r = rho
#     val = np.zeros(n)
#     for i in range(n):
#         val[i] = lib.bivnor(ah[i],ak[i],r)
#     return val


def DGv(v,w,rho):
    arg = (v-rho*w)/np.sqrt(1-rho**2)
    return norm.pdf(v)*norm.cdf(arg)

def DGw(v,w,rho):
    arg = (w-rho*v)/np.sqrt(1-rho**2)
    return norm.pdf(w)*norm.cdf(arg)

def DGrho(v,w,rho):
    c1 = G(v,w,rho)/(1-rho**2)**2
    c2 = rho - rho**3
    c3 = (1+rho**2)*get_EXiXj(1,2,v,w,rho)
    c4 = -rho*(get_EXiXj(1,1,v,w,rho)+get_EXiXj(2,2,v,w,rho))
    drho = c1*(c2+c3+c4)
    return drho

def get_EXiXj(i,j,v,w,rho):
    upper = np.column_stack((v,w))
    mu = np.array([0,0])
    sigma_rho = np.array([[1,rho],[rho,1]])
    if i==1 and j==2:
        EXiXj = sigma_rho[0,1] + sigma_rho[1,0]*(-v*get_Fk(v,1,mu,sigma_rho,upper)) + sigma_rho[0,1]*(-w*get_Fk(w,2,mu,sigma_rho,upper)) + (sigma_rho[0,0]*sigma_rho[1,1]-sigma_rho[0,1]*sigma_rho[1,0])*get_Fqr(upper,mu,sigma_rho,upper)
    elif i==1 and j==1:
        EXiXj= sigma_rho[1,1] + sigma_rho[0,0]*(-v*get_Fk(v,1,mu,sigma_rho,upper)) + sigma_rho[0,1]**2/sigma_rho[1,1]*(-w*get_Fk(w,2,mu,sigma_rho,upper)) + (sigma_rho[0,1]*sigma_rho[0,0]-sigma_rho[0,1]**2*sigma_rho[1,0]/sigma_rho[1,1])*get_Fqr(upper,mu,sigma_rho,upper)
    elif i==2 and j==2:
        EXiXj= sigma_rho[1,1] + sigma_rho[1,1]*(-w*get_Fk(w,2,mu,sigma_rho,upper))+ sigma_rho[1,0]**2/sigma_rho[0,0]*(-v*get_Fk(v,1,mu,sigma_rho,upper))+ (sigma_rho[1,0]*sigma_rho[1,1]-sigma_rho[1,0]**2*sigma_rho[0,1]/sigma_rho[0,0])*get_Fqr(upper,mu,sigma_rho,upper)
    return EXiXj

def get_Fqr(X,mu,sigma_rho,upper):
    v = upper[:,0]
    w = upper[:,1]
    rho = sigma_rho[0,1]
    alpha = G(v,w,rho)
    Fqr = gpdf(X,mu,sigma_rho)/alpha
    return Fqr

def get_Fk(xn,i,mu,sigma_rho,upper):
    v = upper[:,0]
    w = upper[:,1]
    n = xn.shape[0]
    C = sigma_rho
    rho = sigma_rho[0,1]
    A = np.linalg.inv(sigma_rho)
    if i==1:
        j=1
        i=0
    elif i==2:
        j=0
        i=1
    A_1 = A[j,j]
    A_1_inv = 1/A_1
    C_1 = C[j,j]
    c_nn = C[i,i]
    c = C[j,i]
    mu_1 = mu[j]
    mu_n = mu[i]
    f_xn = np.zeros(n)
    p = G(v,w,rho)
    for l in range(n):
        m = mu_1 + (xn[l] - mu_n) * c/c_nn
        f_xn[l] = np.exp(-0.5 * (xn[l] - mu_n)**2/c_nn) * norm.cdf(upper[l,j],m,np.sqrt(A_1_inv))
    Fk = 1/p * 1/np.sqrt(2 * np.pi * c_nn) * f_xn    
    return Fk

# Tentative values
# Delta1 = -.7
# Delta2 = -.7
# beta11 = .5
# beta12 = .5
# beta13 = .5
# beta21 = 1
# beta22 = 1
# beta23 = 1
# rho    = 0.5
# beta1 = np.array([beta11,beta12,beta13])
# beta2 = np.array([beta21,beta22,beta23])
# theta0 = np.array([Delta1, Delta2, beta11, beta12, beta13, beta21, beta22, beta23, rho])
# ysupp = np.array([00,1,10,11])
dx = 9

def S_theta_x(theta, y,x1, x2,p0,smooth):
  # Function that returns derivative of the value function for fixed x=(x1,x2)
  #beta1, beta2, Delta1, Delta2 = theta
  Delta1 = theta[0]
  Delta2 = theta[1]
  beta1 = theta[2:5]
  beta2 = theta[5:8]
  rho   = theta[8]
  x1 = np.array(x1)
  x2 = np.array(x2)
  dx = x1.shape[1]
  n = x1.shape[0]
  p = p0

  eta1 = 1-G((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-G(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)
  eta2 = G(np.dot(x1,beta1),(-1)*np.dot(x2,beta2)-Delta2,-rho) 
  eta3 = G(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho) + G((-1)*np.dot(x1,beta1)-Delta1,(-1)*np.dot(x2,beta2),rho) - G((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)

  # Outcome (0,0)
  s_beta1_00 = -x1*np.repeat(DGv((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)/G((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho),dx).reshape(-1,dx)
  s_beta2_00 = -x2*np.repeat(DGw((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)/G((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho),dx).reshape(-1,dx)
  s_Delta1_00 = np.zeros(n)
  s_Delta2_00 = np.zeros(n)
  s_rho_00 =  DGrho((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)/G((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)

  # Outcome (1,1)
  s_beta1_11 = x1*np.repeat(DGv(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)/G(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho),dx).reshape(-1,dx)
  s_beta2_11 = x2*np.repeat(DGw(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)/G(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho),dx).reshape(-1,dx)
  s_Delta1_11 = DGv(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)/G(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)
  s_Delta2_11 = DGw(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)/G(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)
  s_rho_11 = DGrho(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)/G(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)

  # Outcome (0,1) and (1,0)
  idcase1 = (p[:,2]/(p[:,1]+p[:,2])*eta1 >= eta3) * (p[:,2]/(p[:,1]+p[:,2])*eta1 <= eta2)
  idcase2 = (p[:,2]/(p[:,1]+p[:,2])*eta1 > eta2)
  idcase3 = (p[:,2]/(p[:,1]+p[:,2])*eta1 < eta3)

  # Case 1
  s_beta1_01_case1 = x1*np.repeat((DGv((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-DGv(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho))/eta1,dx).reshape(-1,dx)
  s_beta2_01_case1 = x2*np.repeat((DGw((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-DGw(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho))/eta1,dx).reshape(-1,dx)
  s_Delta1_01_case1 = -DGv(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)/eta1
  s_Delta2_01_case1 = -DGw(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)/eta1
  s_rho_01_case1 = (-DGrho((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-DGrho(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho))/eta1 


  s_beta1_10_case1 = x1*np.repeat((DGv((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-DGv(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho))/eta1,dx).reshape(-1,dx)
  s_beta2_10_case1 = x2*np.repeat((DGw((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-DGw(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho))/eta1,dx).reshape(-1,dx)
  s_Delta1_10_case1 = -DGv(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)/eta1
  s_Delta2_10_case1 = -DGw(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)/eta1
  s_rho_10_case1 = (-DGrho((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-DGrho(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho))/eta1

  #Case 2 
  s_beta1_01_case2 = x1*np.repeat((DGv((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-DGv(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)-DGv(np.dot(x1,beta1),(-1)*np.dot(x2,beta2)-Delta2,-rho))/(eta1-eta2),dx).reshape(-1,dx)
  s_beta2_01_case2 = x2*np.repeat((DGw((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-DGw(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)+DGw(np.dot(x1,beta1),(-1)*np.dot(x2,beta2)-Delta2,-rho))/(eta1-eta2),dx).reshape(-1,dx) # Note the sign of the last term on the numerator due to the chain rule
  s_Delta1_01_case2 = -DGv(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)/(eta1-eta2)
  s_Delta2_01_case2 = (-DGw(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)+ \
                   DGw(np.dot(x1,beta1),(-1)*np.dot(x2,beta2)-Delta2,-rho))/(eta1-eta2)
  s_rho_01_case2 = (-DGrho((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho) - DGrho(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho) + DGrho(np.dot(x1,beta1),(-1)*np.dot(x2,beta2)-Delta2,-rho) )/(eta1-eta2)# Note the sign of the last term due to the chain rule 
   
  s_beta1_10_case2 = x1*np.repeat(DGv(np.dot(x1,beta1),(-1)*np.dot(x2,beta2)-Delta2,-rho)/eta2,dx).reshape(-1,dx)
  s_beta2_10_case2 = -x2*np.repeat(DGw(np.dot(x1,beta1),(-1)*np.dot(x2,beta2)-Delta2,-rho)/eta2,dx).reshape(-1,dx)
  s_Delta1_10_case2 = np.zeros(n)
  s_Delta2_10_case2 = -DGw(np.dot(x1,beta1),(-1)*np.dot(x2,beta2)-Delta2,-rho)/eta2
  s_rho_10_case2 = -DGrho(np.dot(x1,beta1),(-1)*np.dot(x2,beta2)-Delta2,-rho)/eta2

  #Case 3 # Check this case carefully again eta3 enters with a minus sign.
  s_beta1_01_case3 = x1*np.repeat((DGv((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-DGv(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho) \
      - DGv(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho)+DGv((-1)*np.dot(x1,beta1)-Delta1,(-1)*np.dot(x2,beta2),rho)-DGv((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho))/(eta1-eta3),dx).reshape(-1,dx) # Note the sign of the derivative of eta3 due to the chain rule. 
  s_beta2_01_case3 = x2*np.repeat((DGw((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-DGw(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)\
      +DGw(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho) + DGw((-1)*np.dot(x1,beta1)-Delta1,(-1)*np.dot(x2,beta2),rho) - DGw((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho))/(eta1-eta3),dx).reshape(-1,dx)
  s_Delta1_01_case3 = (-DGv(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)\
      - DGv(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho) + DGv((-1)*np.dot(x1,beta1)-Delta1,(-1)*np.dot(x2,beta2),rho))/(eta1-eta3)
  s_Delta2_01_case3 = (-DGw(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)\
      +DGw(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho))/(eta1-eta3) 
  s_rho_01_case3 = (-DGrho((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-DGrho(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)\
      +DGrho(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho)-DGrho((-1)*np.dot(x1,beta1)-Delta1,(-1)*np.dot(x2,beta2),rho)+DGrho((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho))/(eta1-eta3)# HK stopped here 1/10/21

  s_beta1_10_case3 = x1*np.repeat((DGv(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho) - DGv((-1)*np.dot(x1,beta1)-Delta1,(-1)*np.dot(x2,beta2),rho) + DGv((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho))/eta3,dx).reshape(-1,dx)
  s_beta2_10_case3 = x2*np.repeat((-DGw(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho) - DGw((-1)*np.dot(x1,beta1)-Delta1,(-1)*np.dot(x2,beta2),rho) + DGw((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho))/eta3,dx).reshape(-1,dx)
  s_Delta1_10_case3 = (DGv(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho) - DGv((-1)*np.dot(x1,beta1)-Delta1,(-1)*np.dot(x2,beta2),rho))/eta3
  s_Delta2_10_case3 = (-DGw(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho))/eta3
  s_rho_10_case3 = (-DGrho(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho)+ DGrho((-1)*np.dot(x1,beta1)-Delta1,(-1)*np.dot(x2,beta2),rho) - DGrho((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho))/eta3

  # aggregate across cases
  s_beta1_01 = s_beta1_01_case1 * np.repeat(idcase1,dx).reshape(-1,dx) + s_beta1_01_case2 * np.repeat(idcase2,dx).reshape(-1,dx) + s_beta1_01_case3 * np.repeat(idcase3,dx).reshape(-1,dx)
  s_beta2_01 = s_beta2_01_case1 * np.repeat(idcase1,dx).reshape(-1,dx) + s_beta2_01_case2 * np.repeat(idcase2,dx).reshape(-1,dx) + s_beta2_01_case3 * np.repeat(idcase3,dx).reshape(-1,dx)
  s_beta1_10 = s_beta1_10_case1 * np.repeat(idcase1,dx).reshape(-1,dx) + s_beta1_10_case2 * np.repeat(idcase2,dx).reshape(-1,dx) + s_beta1_10_case3 * np.repeat(idcase3,dx).reshape(-1,dx)
  s_beta2_10 = s_beta2_10_case1 * np.repeat(idcase1,dx).reshape(-1,dx) + s_beta2_10_case2 * np.repeat(idcase2,dx).reshape(-1,dx) + s_beta2_10_case3 * np.repeat(idcase3,dx).reshape(-1,dx)
  s_Delta1_01 = s_Delta1_01_case1 * idcase1 + s_Delta1_01_case2 * idcase2 + s_Delta1_01_case3 * idcase3
  s_Delta2_01 = s_Delta2_01_case1 * idcase1 + s_Delta2_01_case2 * idcase2 + s_Delta2_01_case3 * idcase3
  s_Delta1_10 = s_Delta1_10_case1 * idcase1 + s_Delta1_10_case2 * idcase2 + s_Delta1_10_case3 * idcase3
  s_Delta2_10 = s_Delta2_10_case1 * idcase1 + s_Delta2_10_case2 * idcase2 + s_Delta2_10_case3 * idcase3
  s_rho_01 = s_rho_01_case1 * idcase1 + s_rho_01_case2 * idcase2 + s_rho_01_case3 * idcase3
  s_rho_10 = s_rho_10_case1 * idcase1 + s_rho_10_case2 * idcase2 + s_rho_10_case3 * idcase3

  s_beta11 = np.array([s_beta1_00[:,0], s_beta1_01[:,0],s_beta1_10[:,0], s_beta1_11[:,0]]).T
  s_beta12 = np.array([s_beta1_00[:,1], s_beta1_01[:,1],s_beta1_10[:,1], s_beta1_11[:,1]]).T
  s_beta13 = np.array([s_beta1_00[:,2], s_beta1_01[:,2],s_beta1_10[:,2], s_beta1_11[:,2]]).T
  s_beta21 = np.array([s_beta2_00[:,0], s_beta2_01[:,0],s_beta2_10[:,0], s_beta2_11[:,0]]).T
  s_beta22 = np.array([s_beta2_00[:,1], s_beta2_01[:,1],s_beta2_10[:,1], s_beta2_11[:,1]]).T
  s_beta23 = np.array([s_beta2_00[:,2], s_beta2_01[:,2],s_beta2_10[:,2], s_beta2_11[:,2]]).T
  s_Delta1 = np.array([s_Delta1_00, s_Delta1_01,s_Delta1_10, s_Delta1_11]).T
  s_Delta2 = np.array([s_Delta2_00, s_Delta2_01,s_Delta2_10, s_Delta2_11]).T
  s_rho    = np.array([s_rho_00, s_rho_01, s_rho_10, s_rho_11]).T 
  
  selector = (np.tile(ysupp,(n,1))==np.repeat(y,4).reshape(n,-1))
  if smooth == 0:
    S_beta11 = np.sum(s_beta11*selector,axis=1)
    S_beta12 = np.sum(s_beta12*selector,axis=1)
    S_beta13 = np.sum(s_beta13*selector,axis=1)
    S_beta21 = np.sum(s_beta21*selector,axis=1)
    S_beta22 = np.sum(s_beta22*selector,axis=1)
    S_beta23 = np.sum(s_beta23*selector,axis=1)
    S_Delta1 = np.sum(s_Delta1*selector,axis=1)
    S_Delta2 = np.sum(s_Delta2*selector,axis=1)
    S_rho    = np.sum(s_rho*selector,axis=1)
  elif smooth == 1:
    S_beta11 = np.sum(s_beta11*p,axis=1)
    S_beta12 = np.sum(s_beta12*p,axis=1)
    S_beta13 = np.sum(s_beta13*p,axis=1)
    S_beta21 = np.sum(s_beta21*p,axis=1)
    S_beta22 = np.sum(s_beta22*p,axis=1)
    S_beta23 = np.sum(s_beta23*p,axis=1)
    S_Delta1 = np.sum(s_Delta1*p,axis=1)
    S_Delta2 = np.sum(s_Delta2*p,axis=1)
    S_rho    = np.sum(s_rho*p,axis=1)
  return np.array([S_Delta1,S_Delta2,S_beta11,S_beta12,S_beta13,S_beta21,S_beta22,S_beta23,S_rho]).T


@ray.remote
def get_Tn(theta_star,y,x1,x2,ccp):
    # Estimate CCP
    score = S_theta_x(theta_star,y,x1,x2,ccp,0)
    temp = np.sum(score,axis=1)
    score = score[~np.isnan(temp),:]
    nn = score.shape[0]
    sbar = np.sum(score,axis=0)/nn
    cscore = score - np.tile(sbar,(nn,1))
    W = np.einsum('ij,ik->jk', cscore, cscore)/nn
    S = np.sum(score,axis=0)/np.sqrt(nn)
    try:
        Tn = np.dot(S,np.matmul(pinv(W+0*np.eye(dx)),S))
    except np.linalg.LinAlgError as err:
        Tn = 1e+5
    STn =  np.append(S,Tn)
    return STn

def get_Tn_smooth(theta_star,y,x1,x2,ccp):
    score = S_theta_x(theta_star,y,x1,x2,ccp,1)
    temp = np.sum(score,axis=1)
    score = score[~np.isnan(temp),:]
    sbar = np.sum(score,axis=0)/n
    cscore = score - np.tile(sbar,(n,1))
    weight = rng.standard_normal((n,B)) # seed this properly
    sweighted = []
    for b in range(B):
        w = np.repeat(weight[:,b],dx).reshape((n,dx))
        sweighted.append(np.sum(cscore*w,axis=0)/np.sqrt(n))
    sweighted = np.array(sweighted)
    csweighted = sweighted.T - np.mean(sweighted,axis=0)[:,None]
    BW = (csweighted @ csweighted.T)/B
    try:
        Tn = np.dot(S,np.matmul(pinv(BW+0.1*np.eye(dx)),S))
    except np.linalg.LinAlgError as err:
        Tn = 1e+5
    STn =  np.append(S,Tn)


# 
def L_theta_x(theta,x1,x2,p):
    dy = p.shape[1]
    n =  x1.shape[0]
    Delta1 = theta[0]
    Delta2 = theta[1]
    beta1 = theta[2:5]
    beta2 = theta[5:8]
    rho   = theta[8]

    eta1 = 1-G((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)-G(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)
    eta2 = G(np.dot(x1,beta1),(-1)*np.dot(x2,beta2)-Delta2,-rho) 
    eta3 = G(np.dot(x1,beta1)+Delta1,(-1)*np.dot(x2,beta2)-Delta2,-rho) + G((-1)*np.dot(x1,beta1)-Delta1,(-1)*np.dot(x2,beta2),rho) - G((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho)

    idcase1 = (p[:,2]/(p[:,1]+p[:,2])*eta1 >= eta3) * (p[:,2]/(p[:,1]+p[:,2])*eta1 <= eta2)
    idcase2 = (p[:,2]/(p[:,1]+p[:,2])*eta1 > eta2)
    idcase3 = (p[:,2]/(p[:,1]+p[:,2])*eta1 < eta3)

    # Case 1 (Eq (22))
    qstar_case1 = np.column_stack((G((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho), p[:,1]/(p[:,1]+p[:,2])*eta1, p[:,2]/(p[:,1]+p[:,2])*eta1, G(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)))

    #Case 2 (Eq (23))
    qstar_case2 = np.column_stack((G((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho), eta1-eta2, eta2, G(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)))

    #Case 3 (Eq (24))
    qstar_case3 = np.column_stack((G((-1)*np.dot(x1,beta1),(-1)*np.dot(x2,beta2),rho), eta1-eta3, eta3, G(np.dot(x1,beta1)+Delta1,np.dot(x2,beta2)+Delta2,rho)))

    qstar = qstar_case1 * np.repeat(idcase1,dy).reshape(-1,dy) + qstar_case2 * np.repeat(idcase2,dy).reshape(-1,dy) + qstar_case3 * np.repeat(idcase3,dy).reshape(-1,dy)
    lnqstar = np.log(qstar)
    ind = (np.isfinite(np.sum(lnqstar,axis=1)))*(~np.isnan(np.sum(lnqstar,axis=1)))
    return np.einsum('ij,ij->i',lnqstar[ind,:],p[ind,:]), ind

# Function that returns expected profiled log-likelihood function (L(theta)= E[log of qstar]) 
def L_theta(theta):
    Elnqstar_x, ind = L_theta_x(theta, x1, x2, ccp)
    summation = np.dot(Elnqstar_x,phat_marginal[ind])  
    return summation

def rho_trans(b):
    rho = np.arctan(b)/(np.pi/2)
    return rho

def L_Delta(Delta1, Delta2,out):
    # The following function is used to concentrate out beta
    NNg = Delta1.shape[0]
    result_ids = []

    @ray.remote
    def prof_beta(D1,D2,out):
      L_beta = lambda beta: -L_theta(np.array([D1,D2,beta[0],beta[1],beta[2],beta[3], beta[4],beta[5],beta[6]])) 
      #L_beta = lambda beta: -L_theta(np.array([D1,D2,beta[0],beta[1],beta[2],beta[3], beta[4],beta[5],rho_trans(beta[6])])) # Note: beta[6] is rho
      bds = optimize.Bounds([-10,-2,-2,-10,-2,-2,0],[2,10,3,2,10,3,0.85])
      res = minimize(L_beta, np.array([0,0,0,0,0,0,0]),bounds=bds)#,method='slsqp')
      if out == "value":
        return res.fun
      elif out == "arg":
        # res.x[6] = rho_trans(res.x[6])
        return res.x

    for i in range(NNg):
      result_ids.append(prof_beta.remote(Delta1[i],Delta2[i],out))
    results = ray.get(result_ids)
    return results

def Function_L_Delta_root(numgrid=2):
    xx = np.linspace(-2, 0, numgrid) # grid for Delta1 
    yy = np.linspace(-2, 0, numgrid) # grid for Delta2
    Delta1, Delta2 = np.meshgrid(xx, yy)
    Delta1 = Delta1.ravel()
    Delta2 = Delta2.ravel()
    beta = np.array(L_Delta(Delta1,Delta2,"arg"))
    beta11 = beta[:,0]
    beta12 = beta[:,1]
    beta13 = beta[:,2]
    beta21 = beta[:,3]
    beta22 = beta[:,4]
    beta23 = beta[:,5]
    rho = beta[:,6]
    return Delta1, Delta2, beta11, beta12,beta13, beta21, beta22, beta23, rho

def get_ST(num_grid): 
    Delta1, Delta2, beta11, beta12, beta13, beta21, beta22, beta23, rho = Function_L_Delta_root(num_grid)
    nd = beta11.shape[0]
    thetagrid = np.column_stack((Delta1, Delta2, beta11, beta12, beta13, beta21, beta22, beta23, rho))
    # Rewrite below to calculate Tn and check the value below cv
    STtemp = [get_Tn.remote(thetagrid[i,:],y,x1,x2,ccp) for i in range(nd)]
    ST = ray.get(STtemp)

    # idx = (T <= cv)
    return thetagrid, ST

def get_CS(thetagrid,T,cv):
    idx = (T <= cv) 
    theta_selected = thetagrid[idx,:]
    return theta_selected,idx

numgrid = 30
cv = chi2.ppf(.95,9)
thetagrid, ST = get_ST(numgrid)
np.savez(resfile,thetagrid=thetagrid,ST=ST)
T = np.array([ST[i][9] for i in range(numgrid**2)])
thetasel,idx=get_CS(thetagrid,T,cv)
plt.scatter(thetasel[:,0],thetasel[:,1])

# Delta1 = thetagrid[:,0]
# Delta2 = thetagrid[:,1]
# Delta1cs = Delta1[idx]
# Delta2cs = Delta2[idx]
np.savez(resfile,thetagrid=thetagrid,ST=ST)