import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize

class MarcenkoPasturCovariance:
    def __init__(self, returns):
        q = returns.shape[0]/returns.shape[1]
        bandwidth = returns.shape[0]**(-1/5)
        covariance = returns.cov()
        self.covariance = deNoiseCov(covariance, q, bandwidth)

def fitKDE(obs, bandwidth=.25, kernel='gaussian', x=None):
    """
    Fit kernel to a series of observations, and derive the probability of observation. 
        x is the array of values on which the fit KDE will be evaluated. It is the empirical PDF.
    Args:
        obs (np.ndarray): observations to fit. Commonly is the diagonal of Eigenvalues
        bandwidth (float): The bandwidth of the kernel. Default is .25
        kernel (str): The kernel to use. Valid kernels are ['gaussian'|'tophat'|
            'epanechnikov'|'exponential'|'linear'|'cosine'] Default is 'gaussian'.
        x (np.ndarray): x is the array of values on which the fit KDE will be evaluated
    Returns:
        pd.Series: Empirical PDF
    """
    if len(obs.shape)==1:
        obs = obs.reshape(-1,1) 

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(obs) 

    if x is None:
        x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:
        x=x.reshape(-1,1) 

    logProb = kde.score_samples(x) # log(density) 
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf

def mpPDF(var, q, points=1000):
    """
    Creates a Marchenko-Pastur Probability Density Function
    Args:
        var (float): Variance
        q (float): T/N where T is the number of rows (observations) and N the number of columns (variables)
        pts (int): Number of points used to construct the PDF
    Returns:
        pd.Series: Marchenko-Pastur PDF
    """
    eMin, eMax = var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2 
    eVal = np.linspace(eMin,eMax,points).flatten()
    pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5 
    pdf = pd.Series(pdf,index=eVal)
    return pdf

def errPDFs(var, eVal, q, bandwidth, pts=1000):
    """
    Calculates sum of squared errors between the theoretical and empirical pdf
    Args:
        var (float): Variance
        eVal (np.ndarray): Eigen-values of the correlation matrix
        q (float): T/N where T is the number of rows (observations) and N the number of columns (variables)
    Returns:
        pd.Series: Sum of Squared Errors
    """
    pdf0 = mpPDF(var, q, pts) # theoretical pdf 
    pdf1 = fitKDE(eVal, bandwidth, x=pdf0.index.values) # empirical pdf 
    sse = np.sum((pdf1-pdf0)**2)
    return sse

def findMaxEval(eVal,q,bandwidth):
    """
    Calculates max random eigen-value by fitting Marcenko's dist to the empirical one
    Args:
        var (float): Variance
        eVal (np.ndarray): Eigen-values of the correlation matrix
        q (float): T/N where T is the number of rows (observations) and N the number of columns (variables)
    Returns:
        pd.Series: Sum of Squared Errors
    """
    out = minimize(lambda *x : errPDFs(*x), .5, args=(eVal, q, bandwidth), bounds=((1E-5,1-1E-5),))
    if out.success:
        var=out['x'][0] 
    else:
        var=1 
    eMax = var*(1+(1./q)**.5)**2 
    return eMax, var

def corr2cov(corr, std):
    cov = corr*np.outer(std,std)
    return cov 

def cov2corr(cov):
    """
    Derive the correlation matrix from a covariance matrix
    """
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std,std) 
    corr[corr<-1],corr[corr>1] = -1, 1 # numerical error 
    return corr

def getPCA(matrix):
    """
    Gets the Eigenvalues and Eigenvector values from a Hermitian Matrix
    Args:
        matrix pd.DataFrame: Correlation matrix
    Returns:
         (tuple): tuple containing:
            np.ndarray: Eigenvalues of correlation matrix
            np.ndarray: Eigenvectors of correlation matrix
    """
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1] # arguments for sorting eVal desc 
    eVal, eVec = eVal[indices], eVec[:,indices] 
    eVal = np.diagflat(eVal)
    return eVal, eVec

def denoisedCorr(eVal, eVec, nFacts):
    """
    Remove noise from corr by fixing random eigenvalues
    Args:
        eVal (np.ndarray): Eigen-values of the correlation matrix
        eVal (np.ndarray): Eigen-values of the correlation matrix
        nFacts (int): number of rows of random normal
    Returns:
        np.ndarray: denoised correlation matrix
    """
    eVal_ = np.diag(eVal).copy() 
    eVal_[nFacts:] = eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts) 
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return corr1

def deNoiseCov(covariance, q, bandwidth):
    """
    Remove noise from covariance by fixing random eigenvalues
    Args:
        covariance (pd.DataFrame): covariance matrix
        q (float): T/N where T is the number of rows (observations) and N the number of columns (variables)
        bandwidth (float): The bandwidth of the kernel. Default is .25
    Returns:
        np.ndarray: denoised covariance matrix
    """
    corr0 = cov2corr(covariance)
    eVal, eVec = getPCA(corr0) 
    eMax, _ = findMaxEval(np.diag(eVal), q, bandwidth) 
    nFacts = eVal.shape[0] - np.diag(eVal)[::-1].searchsorted(eMax) 
    corr1 = denoisedCorr(eVal, eVec, nFacts) 
    cov1 = corr2cov(corr1, np.diag(covariance)**.5)
    return cov1

def optPort(cov, mu=None):
    """
    Gives the weights of the portfolio.
    If mu is None, gives the min variance portfolio, else gives the max Sharpe portfolio.
    """
    inv = np.linalg.inv(cov) 
    ones = np.ones(shape = (inv.shape[0], 1)) 
    if mu is None:
        mu = ones 
    w = (inv @ mu) / (ones.T @ inv @ mu)
    return w