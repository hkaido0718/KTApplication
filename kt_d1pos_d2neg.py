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

# Estimate CCP
xseries = get_3Dtensorseries(xlcc_pres,xoa_pres,xsize,2)
Lreg = LogisticRegression(penalty='l2',max_iter=500,verbose=1)
clf = Lreg.fit(xseries, y)
ccp = clf.predict_proba(xseries)
x1 = np.column_stack((np.ones(n),xlcc_pres,xsize))
x2 = np.column_stack((np.ones(n),xoa_pres,xsize))
phat_marginal = np.ones(n)/n

# Check CCP
ccpbdry =  np.maximum((np.amin(ccp,axis=1) < 1e-4),(np.amin(1-ccp,axis=1)<1e-4))
y = y[~ccpbdry]
x1 = x1[~ccpbdry,:]
x2 = x2[~ccpbdry,:]
ccp = ccp[~ccpbdry,:]
phat_marginal = phat_marginal[~ccpbdry]

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

def get_eta1(x1,x2,theta):
    Delta1 = theta[0]
    Delta2 = theta[1]
    beta1 = theta[2:5]
    beta2 = theta[5:8]
    rho   = theta[8]
    v = (-1)*np.dot(x1,beta1)
    w = (-1)*np.dot(x2,beta2)
    val = G(v,w,rho)
    return val

def get_eta2(x1,x2,theta):
    Delta1 = theta[0]
    Delta2 = theta[1]
    beta1 = theta[2:5]
    beta2 = theta[5:8]
    rho   = theta[8]
    v = (-1)*np.dot(x1,beta1) - Delta1
    w = np.dot(x2,beta2)        
    val = G(v,w,-rho)
    return val

def get_eta3(x1,x2,theta):
    Delta1 = theta[0]
    Delta2 = theta[1]
    beta1 = theta[2:5]
    beta2 = theta[5:8]
    rho   = theta[8]
    v = np.dot(x1,beta1)
    w = (-1)*np.dot(x2,beta2) - Delta2
    val  =  G(v,w,-rho)
    return val

def Deta1(x1,x2,theta):
    dx = x1.shape[1]
    n = x1.shape[0]
    Delta1 = theta[0]
    Delta2 = theta[1]
    beta1 = theta[2:5]
    beta2 = theta[5:8]
    rho   = theta[8]
    v = (-1)*np.dot(x1,beta1)
    w = (-1)*np.dot(x2,beta2)
    Deta_Delta1 = np.zeros((n,1))
    Deta_Delta2 = np.zeros((n,1))
    Deta_beta1 = -x1*np.repeat(DGv(v,w,rho),dx).reshape(-1,dx)
    Deta_beta2 = -x2*np.repeat(DGw(v,w,rho),dx).reshape(-1,dx)
    Deta_rho   = DGrho(v,w,rho)
    return np.column_stack((Deta_Delta1,Deta_Delta2,Deta_beta1,Deta_beta2,Deta_rho))

def Deta2(x1,x2,theta):
    dx = x1.shape[1]
    n = x1.shape[0]
    Delta1 = theta[0]
    Delta2 = theta[1]
    beta1 = theta[2:5]
    beta2 = theta[5:8]
    rho   = theta[8]
    v = (-1)*np.dot(x1,beta1)-Delta1
    w = np.dot(x2,beta2)
    Deta_Delta1 = -DGv(v,w,-rho)
    Deta_Delta2 = np.zeros((n,1))
    Deta_beta1 = -x1*np.repeat(DGv(v,w,-rho),dx).reshape(-1,dx)
    Deta_beta2 = x2*np.repeat(DGw(v,w,-rho),dx).reshape(-1,dx)
    Deta_rho   = -DGrho(v,w,-rho)
    return np.column_stack((Deta_Delta1,Deta_Delta2,Deta_beta1,Deta_beta2,Deta_rho))

def Deta3(x1,x2,theta):
    dx = x1.shape[1]
    n = x1.shape[0]
    Delta1 = theta[0]
    Delta2 = theta[1]
    beta1 = theta[2:5]
    beta2 = theta[5:8]
    rho   = theta[8]
    v = np.dot(x1,beta1)
    w = (-1)*np.dot(x2,beta2) -Delta2
    Deta_Delta1 = np.zeros((n,1))
    Deta_Delta2 = -DGw(v,w,-rho)
    Deta_beta1 = x1*np.repeat(DGv(v,w,-rho),dx).reshape(-1,dx)
    Deta_beta2 = -x2*np.repeat(DGw(v,w,-rho),dx).reshape(-1,dx)
    Deta_rho   = -DGrho(v,w,-rho)
    return np.column_stack((Deta_Delta1,Deta_Delta2,Deta_beta1,Deta_beta2,Deta_rho))


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

# # Tentative values
# Delta1 = 7
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
ysupp = np.array([00,1,10,11])

def S_theta_x(theta,y,x1,x2,ccp):
  # Function that returns derivative of the value function for fixed x=(x1,x2)
  #beta1, beta2, Delta1, Delta2 = theta

  dx = x1.shape[1]
  n = x1.shape[0]
  p = ccp
  dtheta = theta.shape[0]

  eta1 = get_eta1(x1,x2,theta)
  eta2 = get_eta2(x1,x2,theta)
  eta3 = get_eta3(x1,x2,theta)

  # Outcome (0,0)
  s_theta_00_case1 = np.zeros((n,dtheta)) 
  s_theta_00_case3 = -Deta2(x1,x2,theta)/np.repeat(1-get_eta2(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_00_case4 = -Deta3(x1,x2,theta)/np.repeat(1-get_eta3(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_00_case7 = (-Deta2(x1,x2,theta)-Deta3(x1,x2,theta))/np.repeat(1-get_eta2(x1,x2,theta)-get_eta3(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_00_other = Deta1(x1,x2,theta)/np.repeat(get_eta1(x1,x2,theta),dtheta).reshape(-1,dtheta)

  # Outcome (0,1)
  s_theta_01_case1 = np.zeros((n,dtheta)) 
  s_theta_01_case2 = -Deta1(x1,x2,theta)/np.repeat(1-get_eta1(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_01_case4 = -Deta3(x1,x2,theta)/np.repeat(1-get_eta3(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_01_case6 = (-Deta1(x1,x2,theta)-Deta3(x1,x2,theta))/np.repeat(1-get_eta1(x1,x2,theta)-get_eta3(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_01_other = Deta2(x1,x2,theta)/np.repeat(get_eta2(x1,x2,theta),dtheta).reshape(-1,dtheta)

  # Outome (1,0)
  s_theta_10_case1 = np.zeros((n,dtheta)) 
  s_theta_10_case2 = -Deta1(x1,x2,theta)/np.repeat(1-get_eta1(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_10_case3 = -Deta2(x1,x2,theta)/np.repeat(1-get_eta2(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_10_case5 = (-Deta1(x1,x2,theta)-Deta2(x1,x2,theta))/np.repeat(1-get_eta1(x1,x2,theta)-get_eta2(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_10_other = Deta3(x1,x2,theta)/np.repeat(get_eta3(x1,x2,theta),dtheta).reshape(-1,dtheta)

  # Outcome (1,1)
  s_theta_11_case1 = np.zeros((n,dtheta)) 
  s_theta_11_case2 = -Deta1(x1,x2,theta)/np.repeat(1-get_eta1(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_11_case3 = -Deta2(x1,x2,theta)/np.repeat(1-get_eta2(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_11_case4 = -Deta3(x1,x2,theta)/np.repeat(1-get_eta3(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_11_case5 = (-Deta1(x1,x2,theta)-Deta2(x1,x2,theta))/np.repeat(1-get_eta1(x1,x2,theta)-get_eta2(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_11_case6 = (-Deta1(x1,x2,theta)-Deta3(x1,x2,theta))/np.repeat(1-get_eta1(x1,x2,theta)-get_eta3(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_11_case7 = (-Deta2(x1,x2,theta)-Deta3(x1,x2,theta))/np.repeat(1-get_eta2(x1,x2,theta)-get_eta3(x1,x2,theta),dtheta).reshape(-1,dtheta)
  s_theta_11_case8 = (-Deta1(x1,x2,theta)-Deta2(x1,x2,theta)-Deta3(x1,x2,theta))/np.repeat(1-get_eta1(x1,x2,theta)-get_eta2(x1,x2,theta)-get_eta3(x1,x2,theta),dtheta).reshape(-1,dtheta)

  # id indicators
  idcase1 = (p[:,0]>=eta1)*(p[:,1]>=eta2)*(p[:,2]>=eta3)
  idcase2 = (p[:,0]< eta1)*(p[:,1]/(1-p[:,0])>=eta2/(1-eta1))*(p[:,2]/(1-p[:,0])>=eta3/(1-eta1))
  idcase3 = (p[:,1]< eta2)*(p[:,0]/(1-p[:,1])>=eta1/(1-eta2))*(p[:,2]/(1-p[:,1])>=eta3/(1-eta2))
  idcase4 = (p[:,2]< eta3)*(p[:,0]/(1-p[:,2])>=eta1/(1-eta3))*(p[:,1]/(1-p[:,2])>=eta2/(1-eta3))
  idcase5 = (p[:,2]/(1-p[:,0]-p[:,1])>=eta3/(1-eta1-eta2))*(p[:,0]/(1-p[:,0]-p[:,1])<eta1/(1-eta1-eta2))*(p[:,1]/(1-p[:,0]-p[:,1])<eta2/(1-eta1-eta2))
  idcase6 = (p[:,1]/(1-p[:,0]-p[:,2])>=eta2/(1-eta1-eta3))*(p[:,0]/(1-p[:,0]-p[:,2])<eta1/(1-eta1-eta3))*(p[:,2]/(1-p[:,0]-p[:,2])<eta3/(1-eta1-eta3))
  idcase7 = (p[:,0]/(1-p[:,1]-p[:,2])>=eta1/(1-eta2-eta3))*(p[:,1]/(1-p[:,1]-p[:,2])<eta2/(1-eta2-eta3))*(p[:,2]/(1-p[:,1]-p[:,2])<eta3/(1-eta2-eta3))
  idcase8 = (p[:,0]/p[:,3]<eta1/(1-eta1-eta2-eta3))*(p[:,1]/p[:,3]<eta2/(1-eta1-eta2-eta3))*((p[:,2]/p[:,3]<eta3/(1-eta1-eta2-eta3)))
  idfreq = np.array([np.sum(idcase1),np.sum(idcase2),np.sum(idcase3),np.sum(idcase4),np.sum(idcase5),np.sum(idcase6),np.sum(idcase7),np.sum(idcase8)])

  # aggregate across cases
  s_theta_00 = s_theta_00_case1 * np.repeat(idcase1,dtheta).reshape(-1,dtheta) + s_theta_00_other * np.repeat(idcase2,dtheta).reshape(-1,dtheta) + s_theta_00_case3 * np.repeat(idcase3,dtheta).reshape(-1,dtheta) + s_theta_00_case4 * np.repeat(idcase4,dtheta).reshape(-1,dtheta) + s_theta_00_other * np.repeat(idcase5,dtheta).reshape(-1,dtheta) +  s_theta_00_other * np.repeat(idcase6,dtheta).reshape(-1,dtheta) + s_theta_00_case7 * np.repeat(idcase7,dtheta).reshape(-1,dtheta) + s_theta_00_other * np.repeat(idcase8,dtheta).reshape(-1,dtheta)

  s_theta_01 = s_theta_01_case1 * np.repeat(idcase1,dtheta).reshape(-1,dtheta) + s_theta_01_case2 * np.repeat(idcase2,dtheta).reshape(-1,dtheta) + s_theta_01_other * np.repeat(idcase3,dtheta).reshape(-1,dtheta) + s_theta_01_case4 * np.repeat(idcase4,dtheta).reshape(-1,dtheta) + s_theta_01_other * np.repeat(idcase5,dtheta).reshape(-1,dtheta) +  s_theta_01_case6 * np.repeat(idcase6,dtheta).reshape(-1,dtheta) + s_theta_01_other * np.repeat(idcase7,dtheta).reshape(-1,dtheta) + s_theta_01_other * np.repeat(idcase8,dtheta).reshape(-1,dtheta)

  s_theta_10 = s_theta_10_case1 * np.repeat(idcase1,dtheta).reshape(-1,dtheta) + s_theta_10_case2 * np.repeat(idcase2,dtheta).reshape(-1,dtheta) + s_theta_10_case3 * np.repeat(idcase3,dtheta).reshape(-1,dtheta) + s_theta_10_other * np.repeat(idcase4,dtheta).reshape(-1,dtheta) + s_theta_10_case5 * np.repeat(idcase5,dtheta).reshape(-1,dtheta) +  s_theta_10_other * np.repeat(idcase6,dtheta).reshape(-1,dtheta) + s_theta_10_other * np.repeat(idcase7,dtheta).reshape(-1,dtheta) + s_theta_00_other * np.repeat(idcase8,dtheta).reshape(-1,dtheta) 

  s_theta_11 = s_theta_11_case1 * np.repeat(idcase1,dtheta).reshape(-1,dtheta) + s_theta_11_case2 * np.repeat(idcase2,dtheta).reshape(-1,dtheta) + s_theta_11_case3 * np.repeat(idcase3,dtheta).reshape(-1,dtheta) + s_theta_11_case4 * np.repeat(idcase4,dtheta).reshape(-1,dtheta) + s_theta_11_case5 * np.repeat(idcase5,dtheta).reshape(-1,dtheta) +  s_theta_11_case6 * np.repeat(idcase6,dtheta).reshape(-1,dtheta) + s_theta_11_case7 * np.repeat(idcase7,dtheta).reshape(-1,dtheta) + s_theta_11_case8 * np.repeat(idcase8,dtheta).reshape(-1,dtheta) 
  
  selector = (np.tile(ysupp,(n,1))==np.repeat(y,4).reshape(n,-1))
  s_theta  = s_theta_00 * np.repeat(selector[:,0],dtheta).reshape(-1,dtheta) + s_theta_01 * np.repeat(selector[:,1],dtheta).reshape(-1,dtheta) + s_theta_10 * np.repeat(selector[:,2],dtheta).reshape(-1,dtheta) + s_theta_11 * np.repeat(selector[:,3],dtheta).reshape(-1,dtheta)
  
  return s_theta, idfreq


@ray.remote
def get_Tn(theta_star,y,x1,x2,ccp):
    # Estimate CCP
    score,idfreq = S_theta_x(theta_star,y,x1,x2,ccp)
    temp = np.sum(score,axis=1)
    scoremax = np.amax(score,axis=1)
    score = score[scoremax>0,:]
    if np.sum(np.isnan(temp))>0:
        S = np.zeros(9)
        Tn = 1e+4
    else:  
    # score = score[~np.isnan(temp),:]
        nn = score.shape[0]
        sbar = np.sum(score,axis=0)/nn
        cscore = score - np.tile(sbar,(nn,1))
        W = np.einsum('ij,ik->jk', cscore, cscore)/nn
        S = np.sum(score,axis=0)/np.sqrt(nn)
        try:
            Tn = np.dot(S,np.matmul(pinv(W),S))
        except np.linalg.LinAlgError as err:
            Tn = 1e+5
    STn = np.append(S,Tn)
    STn = np.append(STn,idfreq)
    return STn

# Calculates log likelihood for each x
def L_theta_x(theta,x1,x2,p):
    dx = x1.shape[1]
    dy = p.shape[1]
    n = x1.shape[0]
    Delta1 = theta[0]
    Delta2 = theta[1]
    beta1 = theta[2:5]
    beta2 = theta[5:8]
    rho   = theta[8]

    # Calculate etas
    eta1 = get_eta1(x1,x2,theta)
    eta2 = get_eta2(x1,x2,theta)
    eta3 = get_eta3(x1,x2,theta)

    # id indicators
    idcase1 = (p[:,0]>=eta1)*(p[:,1]>=eta2)*(p[:,2]>=eta3)
    idcase2 = (p[:,0]< eta1)*(p[:,1]/(1-p[:,0])>=eta2/(1-eta1))*(p[:,2]/(1-p[:,0])>=eta3/(1-eta1))
    idcase3 = (p[:,1]< eta2)*(p[:,0]/(1-p[:,1])>=eta1/(1-eta2))*(p[:,2]/(1-p[:,1])>=eta3/(1-eta2))
    idcase4 = (p[:,2]< eta3)*(p[:,0]/(1-p[:,2])>=eta1/(1-eta3))*(p[:,1]/(1-p[:,2])>=eta2/(1-eta3))
    idcase5 = (p[:,2]/(1-p[:,0]-p[:,1])>=eta3/(1-eta1-eta2))*(p[:,0]/(1-p[:,0]-p[:,1])<eta1/(1-eta1-eta2))*(p[:,1]/(1-p[:,0]-p[:,1])<eta2/(1-eta1-eta2))
    idcase6 = (p[:,1]/(1-p[:,0]-p[:,2])>=eta2/(1-eta1-eta3))*(p[:,0]/(1-p[:,0]-p[:,2])<eta1/(1-eta1-eta3))*(p[:,2]/(1-p[:,0]-p[:,2])<eta3/(1-eta1-eta3))
    idcase7 = (p[:,0]/(1-p[:,1]-p[:,2])>=eta1/(1-eta2-eta3))*(p[:,1]/(1-p[:,1]-p[:,2])<eta2/(1-eta2-eta3))*(p[:,2]/(1-p[:,1]-p[:,2])<eta3/(1-eta2-eta3))
    idcase8 = (p[:,0]/p[:,3]<eta1/(1-eta1-eta2-eta3))*(p[:,1]/p[:,3]<eta2/(1-eta1-eta2-eta3))*((p[:,2]/p[:,3]<eta3/(1-eta1-eta2-eta3)))

    # qstar
    qstar_case1 = p
    qstar_case2 = np.column_stack((eta1,p[:,1]*(1-eta1)/(1-p[:,0]),p[:,2]*(1-eta1)/(1-p[:,0]),p[:,3]*(1-eta1)/(1-p[:,0])))
    qstar_case3 = np.column_stack((p[:,0]*(1-eta2)/(1-p[:,1]),eta2,p[:,2]*(1-eta2)/(1-p[:,1]),p[:,3]*(1-eta2)/(1-p[:,1])))
    qstar_case4 = np.column_stack((p[:,0]*(1-eta3)/(1-p[:,2]),p[:,1]*(1-eta3)/(1-p[:,2]),eta3,p[:,3]*(1-eta3)/(1-p[:,2])))
    qstar_case5 = np.column_stack((eta1,eta2,p[:,2]*(1-eta1-eta2)/(1-p[:,0]-p[:,1]),p[:,3]*(1-eta1-eta2)/(1-p[:,0]-p[:,1])))
    qstar_case6 = np.column_stack((eta1,p[:,1]*(1-eta1-eta3)/(1-p[:,0]-p[:,2]),eta3,p[:,3]*(1-eta1-eta3)/(1-p[:,0]-p[:,2])))
    qstar_case7 = np.column_stack((p[:,0]*(1-eta2-eta3)/(1-p[:,1]-p[:,2]),eta2,eta3,p[:,3]*(1-eta2-eta3)/(1-p[:,1]-p[:,2])))
    qstar_case8 = np.column_stack((eta1,eta2,eta3,1-eta1-eta2-eta3))

    qstar = qstar_case1 * np.repeat(idcase1,dy).reshape(-1,dy) + qstar_case2 * np.repeat(idcase2,dy).reshape(-1,dy) + qstar_case3 * np.repeat(idcase3,dy).reshape(-1,dy) + qstar_case4 * np.repeat(idcase4,dy).reshape(-1,dy) + qstar_case5 * np.repeat(idcase5,dy).reshape(-1,dy) + qstar_case6 * np.repeat(idcase6,dy).reshape(-1,dy) + qstar_case7 * np.repeat(idcase7,dy).reshape(-1,dy) + qstar_case8 * np.repeat(idcase8,dy).reshape(-1,dy)
    
    lnqstar = np.log(qstar)
    ind = (np.isfinite(np.sum(lnqstar,axis=1)))*(~np.isnan(np.sum(lnqstar,axis=1)))
    return np.einsum('ij,ij->i',lnqstar[ind,:],p[ind,:]),ind

# Function that returns expected profiled log-likelihood function (L(theta)= E[log of qstar]) 
def L_theta(theta):
    Elnqstar_x, ind = L_theta_x(theta, x1, x2, ccp)
    # if np.sum(ind) == 0:
    summation = np.dot(Elnqstar_x,phat_marginal[ind])  
    # else:
    #     summation = 1e+3
    return summation

def rho_trans(b):
    rho = np.arctan(b)/(np.pi/2)
    return rho

def rho_invtrans(rho):
    b = np.tan(rho*(np.pi/2))
    return b

def L_Delta(Delta1, Delta2,out):
    # The following function is used to concentrate out beta
    NNg = Delta1.shape[0]
    result_ids = []

    @ray.remote
    def prof_beta(D1,D2,out):
      L_beta = lambda beta: -L_theta(np.array([D1,D2,beta[0],beta[1],beta[2],beta[3], beta[4],beta[5],rho_trans(beta[6])])) # Note: beta[6] is rho
      bds = optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,0,np.inf,np.inf,np.inf])
      res = minimize(L_beta, np.array([0,0,0,0,0,0,0]),bounds=bds)
      if out == "value":
        return res.fun
      elif out == "arg":
        res.x[6] = rho_trans(res.x[6])
        return res.x

    for i in range(NNg):
      result_ids.append(prof_beta.remote(Delta1[i],Delta2[i],out))
    results = ray.get(result_ids)
    return results

def Function_L_Delta_root(num_grid=4):
    x = np.linspace(0, 2, num_grid) # grid for Delta1 
    y = np.linspace(-2, 0, num_grid) # grid for Delta2
    Delta1, Delta2 = np.meshgrid(x, y)
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
    idx = (T <= cv) + (T==1e+5)
    theta_selected = thetagrid[idx,:]
    return theta_selected,idx

numgrid = 100
cv = chi2.ppf(.95,9)
thetagrid, ST = get_ST(numgrid)
T = np.array([ST[i][9] for i in range(numgrid**2)])
thetasel,idx=get_CS(thetagrid,T,cv)
plt.scatter(thetasel[:,0],thetasel[:,1])
resfile = "app_conf_set_posneg_ml.npz"
# Delta1 = thetagrid[:,0]
# Delta2 = thetagrid[:,1]
# Delta1cs = Delta1[idx]
# Delta2cs = Delta2[idx]
np.savez(resfile,thetagrid=thetagrid,ST=ST)