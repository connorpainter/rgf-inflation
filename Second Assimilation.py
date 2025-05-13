import numpy as np
import scipy as sp
import scipy.linalg
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
import math
import time
import itertools
from mpl_toolkits.mplot3d import Axes3D
from tempfile import TemporaryFile
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from colorama import Fore, Style
from numba import jit

###############################################################################
## ----------------------------GLOBAL CONSTANTS----------------------------- ##
###############################################################################

G = 6.674e-11
hbar = 1.055e-34
c = 2.998e8

###############################################################################
## ----------------------------GLOBAL VARIABLES----------------------------- ##
###############################################################################

noise = 1e-13
ddphi0 = 0
tPrime0 = 0
ddphiPrev = 0
dummy = True

###############################################################################
## -----------------------------MATRIX CREATION----------------------------- ##
###############################################################################



def VCOV(s, phiVec, infoVec):
    
    l = len(infoVec[0])
    vcov = np.zeros((l,l))

    for i in range(l):
        for j in np.arange(i,l):
            
            vcov[i, j] = popVCOV(s, phiVec, infoVec, i, j)
            vcov[j, i] = vcov[i, j]
    
    return vcov



def popVCOV(s, phiVec, infoVec, i, j):
    
    alpha = int(infoVec[0][i])
    beta = int(infoVec[0][j])
    t1 = int(infoVec[1][i])
    t2 = int(infoVec[1][j])
    
    diffVec = np.array(phiVec[t1]) - np.array(phiVec[t2]) 
    base = np.exp(-1*np.dot(diffVec, diffVec)/(2*s**2))   
    multiplier = 1
    
    if (alpha==-1 or beta==-1) and alpha!=beta:
        if alpha>beta:
            multiplier = 1/(s**2)*(phiVec[t2][alpha] - phiVec[t1][alpha]) 
        else:
            multiplier = -1/(s**2)*(phiVec[t2][beta] - phiVec[t1][beta])
    
    elif alpha!=-1 and beta!=-1:
        if alpha==beta:
            multiplier = -1/(s**4)*(phiVec[t1][alpha] - phiVec[t2][alpha])**2 + 1/(s**2)
        else:
            multiplier = -1/(s**4)*(phiVec[t1][beta] - phiVec[t2][beta])*(phiVec[t1][alpha] - phiVec[t2][alpha])
    
    return multiplier*base



def tetrisVCOV(s, phiVec, upLeftCov, infoVec):
    ## Tacks on one "row" (could be many) and "column" onto V covariance matrix.
    
    l = len(infoVec[0])
    
    vcov = np.zeros((l,l))
    
    vcov[0:len(upLeftCov), 0:len(upLeftCov)] = upLeftCov
    
    for i in np.arange(len(upLeftCov), l):
        for j in range(l):
            vcov[i, j] = popVCOV(s, phiVec, infoVec, i, j)
            vcov[j, i] = vcov[i, j]
    
    return vcov



def tetrisCompressedVCOV(s, phiVec, upLeftCov, infoVec, A, alsoNormal = False, OO = [], annotate = False):
    
    fakeL = len(infoVec[0])
    change = (fakeL-len(A[0]))-(len(upLeftCov)-len(A)) ## How much has been added, including new info, since compression minus how much had been added before new info.
    l = len(upLeftCov)+change
    
    fakevcov = np.empty((fakeL, fakeL))
    vcov = np.empty((l,l))
    
    for i in np.arange(fakeL - change, fakeL):
        for j in range(fakeL):
            fakevcov[i, j] = popVCOV(s, phiVec, infoVec, i, j)
    
    lA = len(A[0])
    
    vcov[0:len(upLeftCov), 0:len(upLeftCov)] = upLeftCov
    
    NA = fakevcov[-change:, 0:lA] @ A.T
    NO = fakevcov[-change:, lA:-change]
    NN = fakevcov[-change:, -change:]
    
    lULC, lNA, lNO, lNN = len(upLeftCov), len(NA[0]), len(NO[0]), len(NN)
    
    vcov[lULC:, 0:lNA] = NA
    vcov[0:lNA, lULC:] = NA.T
    vcov[lULC:, lNA:lNA+lNO] = NO
    vcov[lNA:lNA+lNO, lULC:] = NO.T
    vcov[-lNN:, -lNN:] = NN
    
    if alsoNormal:
        if len(OO)==0: print("Warning: full upper-left V covariance matrix not given as 'OO'!")
        fakevcov[0:len(OO), 0:len(OO)] = OO
        fakevcov[:, -change:] = fakevcov[-change:, :].T
        
        return vcov, fakevcov
    
    return vcov
 
    

def XTX(s, phiVec, infoVec):
    l = len(infoVec[0])
    xtx = np.zeros((l,l))

    for i in range(l):
        for j in np.arange(i,l):
            
            xtx[i, j] = popXTX(s, phiVec, infoVec, i, j)
            xtx[j, i] = xtx[i, j]
    
    return xtx



def popXTX(s, phiVec, infoVec, i, j):
    
    dimPhi = len(phiVec[0])
    alpha, beta, t1, t2 = int(infoVec[0][i]), int(infoVec[0][j]), int(infoVec[1][i]), int(infoVec[1][j])
    diffVec = np.array(phiVec[t1]) - np.array(phiVec[t2]) 
    magVec = np.dot(diffVec, diffVec)
    base = (np.pi*s**2)**(dimPhi/2)*np.exp(-1*magVec/(4*s**2))
    multiplier = 1 + dimPhi/(2*s**2) - magVec/(4*s**4)
    
    if (alpha==-1 or beta==-1) and alpha!=beta:
        if alpha>beta:
            multiplier = (-1*(phiVec[t2][alpha]-phiVec[t1][alpha])/(2*s**2))*(1 + (dimPhi+2)/(2*s**2) - magVec/(4*s**4)) 
        else:
            multiplier = ((phiVec[t2][beta]-phiVec[t1][beta])/(2*s**2))*(1 + (dimPhi+2)/(2*s**2) - magVec/(4*s**4))
    
    elif alpha!=-1 and beta!=-1:
        if alpha==beta:
            multiplier = ((-1*(phiVec[t2][alpha]-phiVec[t1][alpha])**2/(4*s**4))*(1 + (dimPhi+4)/(2*s**2) - magVec/(4*s**4)) + 1/(2*s**2)*(1 + (dimPhi+2)/(2*s**2) - magVec/(4*s**4)))
        else:
            multiplier = (-1*(phiVec[t2][alpha]-phiVec[t1][alpha])*(phiVec[t2][beta]-phiVec[t1][beta])/(4*s**4))*(1 + (dimPhi+4)/(2*s**2) - magVec/(4*s**4))
    
    return multiplier*base



def tetrisXTX(s, phiVec, upLeftXTX, infoVec):
    
    l = len(infoVec[0])

    xtx = np.zeros((l,l))
    
    xtx[0:len(upLeftXTX), 0:len(upLeftXTX)] = upLeftXTX
    
    for i in np.arange(len(upLeftXTX), l):
        for j in range(l):
            xtx[i, j] = popXTX(s, phiVec, infoVec, i, j)
            xtx[j, i] = xtx[i, j]
    
    return xtx


###############################################################################
## -----------------------SPEED OPTIMIZATION FUNCTIONS---------------------- ##
###############################################################################
    


def smartVCOV(s, phiPoints, infoVec):
    l = len(infoVec[0])
    vcov = np.zeros((l,l))
    infoVec = infoVec.astype(int)
    vIndices, gIndices = np.where(infoVec[0]==-1)[0], np.where(infoVec[0]!=-1)[0]
    
    ## VV
    vLocations = np.array(list(itertools.combinations_with_replacement(vIndices, 2)))
    diffVecs = phiPoints[infoVec[1, vLocations[:,0]]] - phiPoints[infoVec[1, vLocations[:,1]]]
    
    vcov[vLocations[:,0], vLocations[:,1]] = np.exp(-1*np.linalg.norm(diffVecs, axis=1)**2/(2*s**2))
    vcov[vLocations[:,1], vLocations[:,0]] = vcov[vLocations[:,0], vLocations[:,1]]
    
    ## VG and GV
    gLocations = np.array(list(itertools.product(vIndices, gIndices)))
    gvLocations = gLocations[gLocations[:,0] > gLocations[:,1]]
    vgLocations = gLocations[gLocations[:,0] < gLocations[:,1]]
    
    gvt1s, gvt2s = infoVec[1, gvLocations[:,0]], infoVec[1, gvLocations[:,1]]
    gvDiffVecs = phiPoints[gvt1s] - phiPoints[gvt2s]
    gvBase = np.exp(-1*np.linalg.norm(gvDiffVecs, axis=1)**2/(2*s**2))
    
    vgt1s, vgt2s = infoVec[1, vgLocations[:,0]], infoVec[1, vgLocations[:,1]]
    vgDiffVecs = phiPoints[infoVec[1, vgLocations[:,0]]] - phiPoints[infoVec[1, vgLocations[:,1]]]
    vgBase = np.exp(-1*np.linalg.norm(vgDiffVecs, axis=1)**2/(2*s**2))
    
    alphas, betas = infoVec[0, gvLocations[:,1]], infoVec[0, vgLocations[:,1]]
    
    vcov[gvLocations[:,0], gvLocations[:,1]] = 1/(s**2)*(phiPoints[gvt1s, alphas] - phiPoints[gvt2s, alphas])*gvBase ## order could be flipped
    vcov[vgLocations[:,0], vgLocations[:,1]] = 1/(s**2)*(phiPoints[vgt1s, betas] - phiPoints[vgt2s, betas])*vgBase
    vcov[gLocations[:,1], gLocations[:,0]] = vcov[gLocations[:,0], gLocations[:,1]]
    
    ## GG (same and different)
    ggLocations = np.array(list(itertools.combinations_with_replacement(gIndices, 2)))
    ggSameLocations = ggLocations[infoVec[0, ggLocations[:,0]] == infoVec[0, ggLocations[:,1]]]
    ggDiffLocations = ggLocations[infoVec[0, ggLocations[:,0]] != infoVec[0, ggLocations[:,1]]]
    
    ggSamet1s, ggSamet2s = infoVec[1, ggSameLocations[:,0]], infoVec[1, ggSameLocations[:,1]]
    ggSameDiffVecs = phiPoints[ggSamet1s] - phiPoints[ggSamet2s]
    ggSameBase = np.exp(-1*np.linalg.norm(ggSameDiffVecs, axis=1)**2/(2*s**2))
    
    ggDifft1s, ggDifft2s = infoVec[1, ggDiffLocations[:,0]], infoVec[1, ggDiffLocations[:,1]]
    ggDiffDiffVecs = phiPoints[ggDifft1s] - phiPoints[ggDifft2s]
    ggDiffBase = np.exp(-1*np.linalg.norm(ggDiffDiffVecs, axis=1)**2/(2*s**2))
    
    alphas = infoVec[0, ggSameLocations[:,0]]
    betas, gammas = infoVec[0, ggDiffLocations[:,0]], infoVec[0, ggDiffLocations[:,1]]
    
    ggSameMultiplier = -1/(s**4)*(phiPoints[ggSamet1s, alphas] - phiPoints[ggSamet2s, alphas])**2 + 1/(s**2)
    ggDiffMultiplier = -1/(s**4)*(phiPoints[ggDifft1s, betas] - phiPoints[ggDifft2s, betas])*(phiPoints[ggDifft1s, gammas] - phiPoints[ggDifft2s, gammas])
    vcov[ggSameLocations[:,0], ggSameLocations[:,1]] = ggSameMultiplier*ggSameBase
    vcov[ggDiffLocations[:,0], ggDiffLocations[:,1]] = ggDiffMultiplier*ggDiffBase
    vcov[ggLocations[:,1], ggLocations[:,0]] = vcov[ggLocations[:,0], ggLocations[:,1]]
    
    return vcov



def tetrisSmartVCOV(s, phiPoints, infoVec, upLeftCov):
    
    infoVec = infoVec.astype(int)
    l, lU = len(infoVec[0]), len(upLeftCov)
    vcov = np.zeros((l,l))
    vcov[0:lU, 0:lU] = upLeftCov
    
    vIndices, gIndices = np.where(infoVec[0]==-1)[0], np.where(infoVec[0]!=-1)[0]
    
    ## VV
    vvLocations = np.array(list(itertools.product(vIndices, [lU])))
    diffVecs = phiPoints[infoVec[1, vvLocations[:,0]]] - phiPoints[infoVec[1, vvLocations[:,1]]]
    
    vcov[vvLocations[:,0], vvLocations[:,1]] = np.exp(-1*np.linalg.norm(diffVecs, axis=1)**2/(2*s**2))
    vcov[vvLocations[:,1], vvLocations[:,0]] = vcov[vvLocations[:,0], vvLocations[:,1]]
    
    ## VG and GV
    gvLocations = np.array(list(itertools.product([lU], gIndices)))
    vgLocations = np.array(list(itertools.product(vIndices, np.arange(lU+1, l).astype(int))))
    
    gvt1s, gvt2s = infoVec[1, gvLocations[:,0]], infoVec[1, gvLocations[:,1]]
    gvDiffVecs = phiPoints[gvt1s] - phiPoints[gvt2s]
    gvBase = np.exp(-1*np.linalg.norm(gvDiffVecs, axis=1)**2/(2*s**2))
    
    vgt1s, vgt2s = infoVec[1, vgLocations[:,0]], infoVec[1, vgLocations[:,1]]
    vgDiffVecs = phiPoints[infoVec[1, vgLocations[:,0]]] - phiPoints[infoVec[1, vgLocations[:,1]]]
    vgBase = np.exp(-1*np.linalg.norm(vgDiffVecs, axis=1)**2/(2*s**2))
    
    alphas, betas = infoVec[0, gvLocations[:,1]], infoVec[0, vgLocations[:,1]]
    
    vcov[gvLocations[:,0], gvLocations[:,1]] = 1/(s**2)*(phiPoints[gvt1s, alphas] - phiPoints[gvt2s, alphas])*gvBase ## order could be flipped
    vcov[vgLocations[:,0], vgLocations[:,1]] = 1/(s**2)*(phiPoints[vgt1s, betas] - phiPoints[vgt2s, betas])*vgBase
    vcov[gvLocations[:,1], gvLocations[:,0]] = vcov[gvLocations[:,0], gvLocations[:,1]]
    vcov[vgLocations[:,1], vgLocations[:,0]] = vcov[vgLocations[:,0], vgLocations[:,1]]
    
    ## GG (same and different)
    ggLocations = np.array(list(itertools.product(gIndices, np.arange(lU+1, l).astype(int))))
    ggSameLocations = ggLocations[infoVec[0, ggLocations[:,0]] == infoVec[0, ggLocations[:,1]]]
    ggDiffLocations = ggLocations[infoVec[0, ggLocations[:,0]] != infoVec[0, ggLocations[:,1]]]
    
    ggSamet1s, ggSamet2s = infoVec[1, ggSameLocations[:,0]], infoVec[1, ggSameLocations[:,1]]
    ggSameDiffVecs = phiPoints[ggSamet1s] - phiPoints[ggSamet2s]
    ggSameBase = np.exp(-1*np.linalg.norm(ggSameDiffVecs, axis=1)**2/(2*s**2))
    
    ggDifft1s, ggDifft2s = infoVec[1, ggDiffLocations[:,0]], infoVec[1, ggDiffLocations[:,1]]
    ggDiffDiffVecs = phiPoints[ggDifft1s] - phiPoints[ggDifft2s]
    ggDiffBase = np.exp(-1*np.linalg.norm(ggDiffDiffVecs, axis=1)**2/(2*s**2))
    
    alphas = infoVec[0, ggSameLocations[:,0]]
    betas, gammas = infoVec[0, ggDiffLocations[:,0]], infoVec[0, ggDiffLocations[:,1]]
    
    ggSameMultiplier = -1/(s**4)*(phiPoints[ggSamet1s, alphas] - phiPoints[ggSamet2s, alphas])**2 + 1/(s**2)
    ggDiffMultiplier = -1/(s**4)*(phiPoints[ggDifft1s, betas] - phiPoints[ggDifft2s, betas])*(phiPoints[ggDifft1s, gammas] - phiPoints[ggDifft2s, gammas])
    vcov[ggSameLocations[:,0], ggSameLocations[:,1]] = ggSameMultiplier*ggSameBase
    vcov[ggDiffLocations[:,0], ggDiffLocations[:,1]] = ggDiffMultiplier*ggDiffBase
    vcov[ggLocations[:,1], ggLocations[:,0]] = vcov[ggLocations[:,0], ggLocations[:,1]]
    
    return vcov



def smartXTX():
    return None



def popBlockXTX():
    return None



def tetrisSmartXTX():
    return None



###############################################################################
## ----------------------MATRIX TOOLS AND MANIPULATION---------------------- ##
###############################################################################



def quarter(m, definedCut = 0, annotate = False):
    ## If definedCut == 0, slices down the middle into four equal quadrants.
    ## Otherwise slices slice down whatever index is given. Negative values supported.
    
    am = np.array(m)
    l = len(m)
    
    if definedCut==0:
        if annotate: print("Quartering matrix at center...")
        cut = int(l/2)
        return (am[0:cut, 0:cut], am[0:cut, cut:l], am[cut:l, 0:cut], am[cut:l, cut:l])
    else:
        if annotate: print("Quartering matrix at index", definedCut, "...")
        return (am[0:definedCut, 0:definedCut], am[0:definedCut, definedCut:l], am[definedCut:l, 0:definedCut], am[definedCut:l, definedCut:l])



def choleskyUpdate(oldL, s, newPhiVec, vcov = np.zeros(1)):
    ## Given a cholesky decomposition 'oldL' and the new phi vector, will perform a rank 1 update.
    
    if (vcov == np.zeros(1)).all():
        vcov = VCOV(s, newPhiVec)
    
    quartered = quarter(vcov, len(oldL))
    ON = quartered[1]
    NN = quartered[3]

    n = sp.linalg.solve_triangular(oldL, ON, lower = True)

    toChol = NN - np.transpose(n) @ n
    l = sp.linalg.cholesky(toChol + noise*np.identity(len(toChol)), lower = True)    
    
    newL = np.zeros((len(vcov), len(vcov)))
    newL[0:len(oldL), 0:len(oldL)] = oldL
    newL[len(oldL):len(newL), 0:len(oldL)] = np.transpose(n)
    newL[len(oldL):len(newL), len(oldL):len(newL)] = l
    
    return newL



###############################################################################
## ----------------------EIGENSTUFF AND FORGETTING-------------------------- ##
###############################################################################



def minEig(m):
    ## Returns the lowest eigenvalue of a matrix
    
    return np.real(min(sp.linalg.eigvalsh(m)))



def compressVCOV(s, oldPhiPoints, infoVec, cap = 20, percent = -1, OO = [], xtx = [], returnCompressed = False, printTraceRatio = False, plotTraceRatio = False, annotate = False):
    
    if len(OO)==0:
        if annotate: print("Populating Gamma_OO...")
        OO = VCOV(s, oldPhiPoints)
    if len(xtx)==0:
        if annotate: print("Populating XTX...")
        xtx = XTX(s, oldPhiPoints)

    eigProblem = sp.linalg.eigh(xtx, OO + noise*np.identity(len(OO)))
    
    if percent != -1:
        if annotate: print("Calculating number of eigenvalues necessary to encompass", percent, "% of information...")
        percent = percent/100
        total = np.sum(eigProblem[0])
        k = 1
        while np.sum(eigProblem[0][-k:])/total < percent and k<len(xtx):
            k = k+1  
        cap = k
        if annotate: print(cap, "/", len(eigProblem[0]), "eigenvectors to be included.")
    
    A = eigProblem[1][-cap:]
    if printTraceRatio: print("Trace ratio:", traceRatio(s, oldPhiPoints, infoVec, A, OO, annotate = annotate))
    if plotTraceRatio: traceRatioPlot(s, oldPhiPoints, infoVec, eigProblem, OO)
        
    if returnCompressed:
        return A, A @ OO @ A.T
    else:    
        return A
  
    

###############################################################################
## -----------------------PATH CREATION AND ANALYSIS------------------------ ##
###############################################################################
    


def eulerCosmo(phiStuff, vStuff, tStep, s, V0, mean):
    ## np.shape(phiStuff) = (2,dimPhi). phiStuff[0] is position of phi. phiStuff[1] is time derivative.
    ## np.shape(vStuff) = (N+1,). vStuff[0] is V at phiStuff[0]. vStuff[i+1] is dV/d(phi_i).
    
    dphi, V, dV = phiStuff[1], vStuff[0], vStuff[1:]
    
    return np.array(phiStuff) + np.array((dphi, -np.sqrt(24*np.pi*(s**2/2*np.dot(dphi, dphi) + V0*V + mean))*dphi - dV*V0/(s**2)))*tStep


## Differential equation solver with adaptive timesteps. Inverse: tPrime = 0.001 for (s, V0) = (1, 1).
## Created: 5-16-20
## Updated: 5-21-20
def eulerAdapt(phiStuff, vStuff, tPrime, s, V0, mean, i, accelList):
    
    dphi, V, dV = phiStuff[1]*s, vStuff[0]*V0, vStuff[1:]*V0/s ## Scaling to Planck units to use in eqs.
    f = np.sqrt(24*np.pi*(1/2*np.dot(dphi, dphi) + V + mean))
    ddphi = -f*dphi - dV
    
    factor, tick, epsilon, delta, alpha, beta = 1.05, 1/5, 1/10, 1/4, 1/4, 1
    magdphi, magddphi = np.linalg.norm(dphi), np.linalg.norm(ddphi)
    ratio = np.linalg.norm(ddphiPrev)/magddphi
    magDeltaPhi, magDeltaDPhi = magdphi*tPrime, magddphi*tPrime
    
    deltaDDPhiCondition, highAccelCondition, accelDiff = False, False, []
    if i>5: 
        accelDiff = accelList[-5:] - accelList[-6:-1]
        deltaDDPhiCondition = max(np.linalg.norm(accelDiff, axis=1)) < alpha*magddphi
        highAccelCondition = max(np.linalg.norm(accelDiff, axis=1)) > beta*magddphi
    
    ratioCondition = np.abs(np.log10(ratio)) > tick
    deltaPhiCondition = magDeltaPhi < epsilon*s
    deltaDPhiCondition = (magDeltaDPhi < delta*magdphi) or i<3
    
    if ratioCondition and deltaPhiCondition and deltaDPhiCondition and deltaDDPhiCondition:
        tPrime = tPrime*factor
        print(Fore.GREEN + f"Time step increased to: {tPrime}.", Style.RESET_ALL)
    
    if not deltaPhiCondition: 
        while tPrime > epsilon*s/magdphi:
            tPrime = tPrime/factor
            print(Fore.RED + f"FLAG: Big jump... Time step decreased to: {tPrime}.", Style.RESET_ALL)
    if not deltaDPhiCondition: 
        while (tPrime >= delta*magdphi/magddphi):
            tPrime = tPrime/factor
            print(Fore.RED + f"FLAG: Change in speed... Time step decreased to: {tPrime}.", Style.RESET_ALL)
    if highAccelCondition:
        tPrime = tPrime/factor
        print(Fore.RED + f"FLAG: High acceleration... Time step decreased to: {tPrime}.", Style.RESET_ALL)
    phiNew = np.array(phiStuff) + np.array((dphi, ddphi))*tPrime/s ## Dividing by s scales back to "units of s" regime. 
    
    return phiNew, tPrime, ddphi



def eulerDetect(phiStuff, vStuff, tPrime, s, V0, mean, tValues):
    phiLast = phiStuff[-1]
    dphi = phiLast[1]*s
    dimPhi = len(dphi)
    V, dV = vStuff[-dimPhi-1]*V0, vStuff[-dimPhi:]*V0/s
    ddphi = -np.sqrt(24*np.pi*(1/2*np.dot(dphi, dphi) + V + mean))*dphi - dV
    phiNew = np.array(phiLast) + np.array((dphi, ddphi))*tPrime/s
    phiStuff = np.concatenate((phiStuff, [phiNew])) ## Tacking it onto phiStuff early so it will be considered in the fit.
    tValues = np.append(tValues, tValues[-1] + tPrime)
    
    ## Detect signs that time step should be lowered/raised.
    lowerFlag, raiseFlag = False, False
    c = [1/10, 4, 0.999, 0.99999, 1.1]
    
    if np.linalg.norm(dphi*tPrime) > c[0]: lowerFlag = True
    elif len(tValues) >= c[1] + 5: 
        r2 = quadraticR2(tValues[-c[1]:], np.linalg.norm(phiStuff[-c[1]:,1]*s, axis=1))
        if r2 < c[2]: 
            lowerFlag = True
            print(f"Deviant?! R^2 = {r2}, t = {tValues[-1]}")
        elif r2 > c[3]: 
            raiseFlag = True
            print(f"Increase! R^2 = {r2}")
    
    if lowerFlag: tPrime = tPrime/c[4]
    if raiseFlag: tPrime = tPrime*c[4]
    
    return phiStuff, tPrime



def quadratic(x, a, b, c):
    x = np.array(x)
    return a*x**2 + b*x + c



def quadraticR2(tValues, points):
    
    pOpt, pCov = sp.optimize.curve_fit(quadratic, tValues, points)
    residuals = points - quadratic(tValues, *pOpt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((points - np.mean(points))**2)
    R2 = 1 - ss_res/ss_tot
    
    return R2



def wienerFilter(OO, LOONoisy, vNoisy):
    Y = sp.linalg.solve_triangular(LOONoisy, OO, lower = True)
    x = sp.linalg.solve_triangular(LOONoisy, vNoisy, lower = True)
    return np.transpose(Y) @ x



def getZeta(bStuff, plot = True):
    
    phiStuff, tValues, dimPhi = bStuff[0], bStuff[5], bStuff[7]
    magdphi = np.linalg.norm(phiStuff[:,1], axis=1)
    V = extractV(bStuff)
    dV = np.zeros((len(phiStuff), dimPhi))
    for i in range(dimPhi):
        dV[:,i] = extractV(bStuff, index = i)
    magdV = np.linalg.norm(dV, axis=1)
    zeta = np.abs(magdphi**2 + V - np.sqrt(V**2 + 1/(12*np.pi)*magdV**2))
    
    if plot: standardPlot(tValues, zeta, r"Time [$t_{Pl}$]", r"$\zeta$", "Zeta (Measure of Acceleration) VS. Time")
    
    return zeta
    
    

def getArg(bStuff, cos = False, plot = True):
    
    phiStuff, tValues, dimPhi = bStuff[0], bStuff[5], bStuff[7]
    dphi = phiStuff[:,1]
    dV = np.zeros((len(phiStuff), dimPhi))
    for i in range(dimPhi):
        dV[:,i] = extractV(bStuff, index = i)
    args = []
    for t in range(len(phiStuff)):
        args = np.append(args, np.dot(dphi[t], dV[t])/(np.linalg.norm(dphi[t])*np.linalg.norm(dV[t])) )
    if not cos: args = np.arccos(args)
    
    if plot:
        if not cos: standardPlot(tValues, args, r"Time [$t_{Pl}$]", r"$\theta$", r"$\dot{\phi}$, $\nabla V$ Argument VS. Time")
        else: standardPlot(tValues, args, r"Time [$t_{Pl}$]", r"$\cos\theta$", r"$\dot{\phi}$, $\nabla V$ Argument VS. Time")
    
    return args



def getVelocity(bStuff, norm = True, plot = True):
    
    phiStuff, tValues = bStuff[0], bStuff[5]
    velocity = np.linalg.norm(phiStuff[:,1], axis=1) if norm else phiStuff[:,1]
    
    if plot:
        if not norm: standardPlot(tValues, velocity, r"Time [$t_{Pl}$]", r"$\dot{\phi}$ [Planck units]", "Inflaton 'Velocity' VS. Time")
        else: standardPlot(tValues, velocity, r"Time [$t_{Pl}$]", r"$|\dot{\phi}|$ [Planck units]", "Inflaton 'Speed' VS. Time")
        
    return velocity



def getAcceleration(bStuff, norm = True, plot = True):
    
    phiStuff, tValues, mean, dimPhi = bStuff[0], bStuff[5], bStuff[6], bStuff[7]
    dphi = phiStuff[:,1]
    V = extractV(bStuff)
    dV = np.zeros((len(phiStuff), dimPhi))
    for i in range(dimPhi):
        dV[:,i] = extractV(bStuff, index = i)
    acceleration = []
    for t in range(len(phiStuff)):
        if t==0: acceleration = [-np.sqrt(24*np.pi*(1/2*np.dot(dphi[t], dphi[t]) + V[t] + mean))*dphi[t] - dV[t]]
        else: acceleration = np.concatenate((acceleration, np.array([-np.sqrt(24*np.pi*(1/2*np.dot(dphi[t], dphi[t]) + V[t] + mean))*dphi[t] - dV[t]])))
    if norm: acceleration = np.linalg.norm(acceleration, axis=1)
    
    if plot:
        if not norm: standardPlot(tValues, acceleration, r"Time [$t_{Pl}$]", r"$\ddot{\phi}$ [Planck units]", "Inflaton 'Acceleration' VS. Time")
        else: standardPlot(tValues, acceleration, r"Time [$t_{Pl}$]", r"$\ddot{\phi}$ [Planck units]", "Inflaton 'Acceleration' VS. Time")
    
    return acceleration



def getSlopes(bStuff, norm = True, plot = True):
    slopes = [extractV(bStuff, index=0)]
    for i in np.arange(1,bStuff[7]):
        slopes = np.concatenate((slopes, [extractV(bStuff, index = i)]))
    if norm: slopes = np.linalg.norm(slopes, axis=0)
    if plot and norm: standardPlot(bStuff[5], slopes, r"Time [$t_{Pl}$]", "Derivatives of V", "Gradient of Potential VS. Time")
    elif plot:
        plt.figure(figsize=(10,10))
        for i in np.arange(bStuff[7]):
            plt.plot(bStuff[5], slopes[i], label = r"$\frac{dV}{d\phi_{i}}$")
            plt.xlabel(r"Time [$t_{Pl}$]")
            plt.ylabel("Derivatives of V")
            plt.legend()
            plt.title("Gradient of Potential VS. Time")
    return slopes



def extractV(bStuff, index = -1, returnIndices = False, forceBlock = False, returnArrayLocs = False):
    ## Extracts V values from vStuff by default, but can extract any dV index alpha==0...dimPhi-1.
    
    vStuff, infoVec = bStuff[1], bStuff[2]
    V, indices, arrayLocs = [], [], []
    
    for i in range(len(infoVec[0])):
        if infoVec[0,i]==index:
            V = [vStuff[i]] if len(V)==0 else np.append(V, [vStuff[i]])
            indices = [int(infoVec[1,i])] if len(indices)==0 else np.append(indices, [int(infoVec[1,i])])
            arrayLocs = [i] if len(arrayLocs)==0 else np.append(arrayLocs, [i])
    
    if returnIndices:
        return V, indices
    
    if forceBlock:
        dimPhi = int(max(infoVec[0]))+1
        blockVStuff, indices = [], []
        
        for j in range(len(infoVec[0])-dimPhi):
            if (infoVec[0,j:j+dimPhi+1]==np.arange(-1,dimPhi)).all():
                blockVStuff = vStuff[j:j+dimPhi+1] if len(blockVStuff)==0 else np.append(blockVStuff, vStuff[j:j+dimPhi+1])
                indices = [infoVec[1,j]] if len(indices)==0 else np.append(indices, [infoVec[1,j]])
        
        return blockVStuff, indices
    
    if returnArrayLocs:
        return V, arrayLocs
    
    return V



## Finds evolution of the scale factor over the given path. Always ends at ln(a) \approx -73. Units correct.
## Updated: 5-14-20
def findScaleFactor(bStuff, atf = 1.9236e-32, nReheat = 1, log = False, plot = True):
    
    tValues = bStuff[5]
    tSteps = np.roll(tValues, -1) - tValues
    H = getH(bStuff, plot = False)
    ln_a = np.cumsum(H[:-1]*tSteps[:-1])
    
    ## Normalize the scale factor to a reasonable value at the end of inflation.
    atf = atf*nReheat
    ln_a = (ln_a - ln_a[-1]) + np.log(atf) 
    
    if plot: standardPlot(tValues[:-1], ln_a, r"Time Since Inflation Begins [$t_{Pl}$]", r"$\ln(a)$ [Unitless]", r"Scale Factor $a(t)$ VS. Time")
    
    if log: return ln_a
    
    return np.exp(ln_a)



## Computes the tensor perturbation spectrum from any path from the formula P_h(k) = 8*np.pi*H**2 / k**3. (G=hbar=c=1)
## Updated: 5-14-20
def tensorP_h(bStuff, dimensionless = True, wavelength = False, plot = True, SI = False): 
    
    a = findScaleFactor(bStuff, plot = False)
    H = getH(bStuff, plot = False)[:-1]
    ## We seek a range of k values representing tensor modes which "cross the horizon" during inflation.
    k = a * H  
    spectrum = 8*np.pi*H**2 if dimensionless else 8*np.pi*H**2/k**3
    
    ## Handling of optional arguments for how data is returned/plotted.
    xUnitsLabel, yUnitsLabel = "[l_Pl^-1]", "[l_Pl^3]"
    if dimensionless: yUnitsLabel = "[Unitless]"
    if SI: 
        k, xUnitsLabel = k*6.188e34, "[m^-1]" ## Conversion factor from l_Pl^-1 to m^-1. 
        if not dimensionless: spectrum, yUnitsLabel = spectrum*(G*hbar/c**5), "m^3" ## Conversion factor from l_Pl**3 to m**3
    if wavelength: 
        k, xUnitsLabel = 2*np.pi/k, "[l_Pl]"
        if SI: xUnitsLabel = "[m]"
    
    if plot:
        plt.figure(figsize = (8,8))
        plt.loglog(k, spectrum)
        plt.xlabel("k (Wavenumber) " + xUnitsLabel)
        if wavelength: plt.xlabel("Wavelength " + xUnitsLabel)
        plt.ylabel("P_h(k) " + yUnitsLabel)
        plt.title("Tensor Perturbation Power Spectrum")
    
    return k, spectrum



## Updated: 5-14-20
def getH(bStuff, plot = True):
    
    phiStuff, tValues, mean = bStuff[0], bStuff[5], bStuff[6]
    if mean!=0: 
        print(Fore.RED + f"Warning! This path was computed with a nonzero mean ({mean}).")
        print(Style.RESET_ALL)
    
    V, indices = extractV(bStuff, returnIndices = True)
    dphi = phiStuff[indices,1,:]
    magdphi = np.linalg.norm(dphi, axis=1)
    H = np.sqrt(8*np.pi/3*(1/2*magdphi**2 + V + mean))
                
    if plot: standardPlot(tValues, H, "Time Since Inflation Begins [Planck units]", "H(t) [Unitless]", "Hubble Parameter VS. Time")
    
    return H
    


## Evaluates whether a path is converging to a zero-potential minimum at the origin based on 3 indicators.
## Returns boolean array.
## Created: 5-10-2020
## Updated: 5-10-2020
def convergingToOrigin(bStuff, closerFactor = 10, lowerFactor = 10, slowerFactor = 10, annotate = False):
    
    phiStuff, V = bStuff[0], extractV(bStuff)
    phi0, v0 = phiStuff[0][0], V[0]
    phiF, vF = phiStuff[-1][0], V[-1]
    magdphiF = np.linalg.norm(phiStuff[-1][1])
    
    ## Did it get close enough to the origin?
    indicator1 = np.linalg.norm(phi0)/np.linalg.norm(phiF) > closerFactor
    ## Did the potential drop enough?
    indicator2 = False
    if vF>0: indicator2 = v0/vF > lowerFactor
    ## Is dphiF small enough?
    magdphiMax = np.max(np.linalg.norm(phiStuff[:,1], axis=1))
    indicator3 = magdphiMax/magdphiF > slowerFactor
    
    if annotate:
        print(f"Indicator 1: {indicator1}. Starting distance: {np.linalg.norm(phi0)}; Final distance: {np.linalg.norm(phiF)}; Ratio: {np.linalg.norm(phi0)/np.linalg.norm(phiF)}.")
        print(f"Indicator 2: {indicator2}. Starting potential: {v0}; Final potential: {vF}; Ratio: {v0/vF}.")
        print(f"Indicator 3: {indicator3}. Max |dphi|: {magdphiMax}; Final |dphi|: {magdphiF}; Ratio: {magdphiMax/magdphiF}.")
    
    return indicator1, indicator2, indicator3
    


## This function is built to numerically solve eq. (6.51) in Dodelson, verifying an appropriate solution
## for the power spectrum of tensor perturbations.
## Created: 5-11-2020
## Updated: 5-12-2020 (NOT UPDATED FOR NEW TVALUES OUTPUT)
def eulerQuantumV(bStuff, plot = True, soln = False):
    
    tStep = bStuff[5]
    a = findScaleFactor(bStuff, exp = True, plot = False)
    ln_a = np.log(a)
    a_t = sp.signal.lfilter([1.0/30]*30, 1, np.gradient(a)/tStep) ## Divide by tStep since np.gradient doesn't know what the x-axis is.
    a_tt = sp.signal.lfilter([1.0/30]*30, 1, np.gradient(a_t)/tStep)
    a_etaeta = a*a_t**2 + a**2*a_tt
    H = getH(bStuff, plot = False)
    eFoldHotSpot = (ln_a[-1]+ln_a[0])/2
    
    tHotSpot = 0
    while ln_a[tHotSpot] < eFoldHotSpot: ## More efficient way to do this in the line starting "t0 ="
        tHotSpot += 1
    
    k = a[tHotSpot]*H[tHotSpot]
    
    etaValues = getConformalTime(bStuff, tStep)
    etaSteps = tStep/a
    
    ## We need to find the time t0 at which k*etaSteps[t0] ~ 0.01 (else etaStep is too large - diff. eq. blows up)
    t0 = (np.where(etaSteps > 0.01/k))[0][-1]
    eta0 = etaValues[t0]
    quantumV = [np.array((np.exp(-1j*k*eta0), -1j*k*np.exp(-1j*k*eta0)))]*1/np.sqrt(2*k) ## Initial conditions.
    
    for i in np.arange(t0+1, len(a)):
        quantumVNew = quantumV[-1] + np.array((quantumV[-1,1], quantumV[-1,0]*(a_etaeta[i]/a[i] - k**2)))*etaSteps[i] ## Eq. (6.51)
        #quantumVNew = quantumV[-1] + np.array((quantumV[-1,1], quantumV[-1,0]*(2/etaValues[i]**2 - k**2)))*etaSteps[i] ## Eq. (6.56) approximation
        quantumV = np.concatenate((quantumV, [quantumVNew]))
    
    if plot:
        plt.figure(figsize=(10,10))
        plt.semilogy(np.arange(t0, len(a))*tStep, np.real(quantumV[:,0]), color='green', label='Re(v_mine) - \ddot{a}/a')
        plt.xlabel("Regular Time [Planck units]")
        plt.ylabel("v(k, eta(t))")
        plt.title(f"Quantum Coefficient v(k, eta(t)) VS. Regular Time with k = {k}")
        if soln:
            dodelSoln = np.exp(-1j*k*etaValues[t0:-1])/np.sqrt(2*k)*(1 - 1j/(k*etaValues[t0:-1]))
            plt.semilogy(np.arange(t0, len(a))*tStep, np.real(dodelSoln)*-1, color='red', label='Re(v_solution)*(-1)')
        plt.legend()
    
    return quantumV



## This function calculates discrete samples of the conformal time (as defined in Dodelson eq. (6.20)) from scale factor data.
## Created: 5-11-2020
## Updated: 5-13-2020
def getConformalTime(bStuff, plot = True):
    tValues = bStuff[5]
    tSteps = np.roll(tValues, -1) - tValues
    a = findScaleFactor(bStuff, plot=False)
    eta = [0]
    t = len(a)-1
    while t >= 0:
        eta = np.append(eta[0] - tSteps[t]/a[t], eta)
        t -= 1
    if plot: 
        plt.semilogy(tValues, -eta)
        plt.xlabel("Regular Time [t_Pl]")
        plt.ylabel("Conformal Time [t_Pl]")
    return eta



## Created: 5-14-2020
def getPhiStuff(bStuff):
    return bStuff[0]
def getVStuff(bStuff):
    return bStuff[1]
def getInfoVec(bStuff):
    return bStuff[2]
def getS(bStuff):
    return bStuff[3]
def getV0(bStuff):
    return bStuff[4]
def getTValues(bStuff):
    return bStuff[5]
def getTSteps(bStuff, plot = False):
    tSteps = (np.roll(bStuff[5], -1) - bStuff[5])[:-1]
    if plot: 
        plt.plot(tSteps)
        plt.xlabel(r"Iteration Number")
        plt.ylabel(r"Time Step [$t_{Pl}$]")
        plt.title("Time Steps Used in Diff. Eq. Solver")
    return tSteps
def getMean(bStuff):
    return bStuff[6]
def printInfo(bStuff):
    print(f"Shape of inflaton field data = {np.shape(bStuff[0])}")
    print(f"Shape of potential data = {np.shape(bStuff[1])}")
    print(f"Shape of information matrix = {np.shape(bStuff[2])}")
    print(f"s = {bStuff[3]}")
    print(f"V0 = {bStuff[4]}")
    print(f"First time value = {bStuff[5][0]}, Last time value = {bStuff[5][-1]}")
    print(f"mean = {bStuff[6]}")
    print(f"Dimensions of inflaton potential = {bStuff[7]}")
    print(f"Shape of covariance matrix = {np.shape(bStuff[8])}")
    print(f"Adaptive timesteps? {bStuff[9]}.")
    return       
        

###############################################################################
## ---------------------------COMPARISON TOOLS------------------------------ ##
###############################################################################


def compareStepSize(tStepValues, s = 1, V0 = 1, tMax = 150):
    b = bushwhack(tStep = tStepValues[0], s = s, V0 = V0, tMin = tMax, tMax = tMax)
    for tStep in tStepValues[1:]:
        b = bushwhack(bStuff = b, phiStart = np.zeros((2,2)), tStep = tStep, more = tMax)
    return b



def compareStepMethod(tStep, tPrime, s = 1, V0 = 1, tMax = 150, topo = True, diffPlot = True):
    
    bConstant = bushwhack(adapt = False, tStep = tStep, tMin = tMax, tMax = tMax)
    b = bushwhack(bStuff = bConstant, phiStart = np.zeros((2,2)), adapt = True, tPrime = tPrime, more = tMax)
    phiConstant, phiAdapt, tValuesConstant, tValuesAdapt = b[0][:tMax,0], b[0][tMax:,0], b[5][:tMax], b[5][tMax:]
    
    fConstant0 = sp.interpolate.interp1d(tValuesConstant, phiConstant[:,0], kind = 'cubic') ## IDK what I'm doing.
    fConstant1 = sp.interpolate.interp1d(tValuesConstant, phiConstant[:,1], kind = 'cubic')
    fAdapt0 = sp.interpolate.interp1d(tValuesAdapt, phiAdapt[:,0], kind = 'cubic')
    fAdapt1 = sp.interpolate.interp1d(tValuesAdapt, phiAdapt[:,1], kind = 'cubic')
    phiConstantCont = (fConstant0(tValuesConstant), fConstant1(tValuesConstant)) 
    phiAdaptCont = (fAdapt0(tValuesAdapt), fAdapt1(tValuesAdapt))
    
    infoAdjust = np.array((np.zeros(tMax*3), tMax*np.ones(tMax*3)), dtype = np.int64)
    bAdapt = b[0][tMax:], b[1][tMax*3:], b[2][:,tMax*3:] - infoAdjust, b[3], b[4], tValuesAdapt, b[6], b[7], b[8], b[9]
    
    if diffPlot: 
        plt.plot(tValuesConstant, phiConstantCont[0], 'r')
        plt.plot(tValuesConstant, phiConstantCont[1], 'r', label = f'Constant tStep = {tStep}')
        plt.plot(tValuesAdapt, phiAdaptCont[0], 'b')
        plt.plot(tValuesAdapt, phiAdaptCont[1], 'b', label = f'Adaptive tPrime = {tPrime}')
        plt.xlabel(r"Time [$t_{Pl}$]")
        plt.ylabel(r"$\phi$ Values")
        plt.title(f"Constant v. Adapted Time Steps")
        plt.legend()
        plt.grid()
        plt.show()
        #standardPlot(b[5][:tMax], np.sqrt((phiAdaptCont[0] - phiConstantCont[0])**2 + (phiAdaptCont[1] - phiConstantCont[1])**2), 
                     #r"Time [$t_{Pl}$]", "Difference", f"Constant (tStep = {tStep}) v. Adapted (tPrime = {tPrime}) Time Steps")
    if topo: topoPlot(b)
    
    return bConstant, bAdapt



def traceRatio(s, phiPoints, infoVec, A, OO, annotate = False):
    ## Only functions inside of compressVCOV: yields the ratio of the trace of the compressed V covariance matrix to that of the full one.
    
    blockLen = len(phiPoints[0])+1
    LOO = sp.linalg.cholesky(OO + noise*np.identity(len(OO)), lower = True)

    if annotate: print("Populating Gamma_NO...")
    
    r, c = blockLen, blockLen*len(phiPoints)
    NO = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            NO[i, j] = popVCOV(s, phiPoints, infoVec, i, j)
    
    if annotate: print("Compressing covariance matrix and finding trace...")
    
    OOc = A @ OO @ np.transpose(A)
    
    NOc = NO @ np.transpose(A)
    
    halfSandwich = sp.linalg.solve_triangular(LOO, np.transpose(NO), lower = True)
    sandwich = np.transpose(halfSandwich) @ halfSandwich ## Cleverly, this is NO @ OOI @ ON
    
    LOOc = sp.linalg.cholesky(OOc + noise*np.identity(len(OOc)),lower = True)#A @ LOO
    halfSandwichc = sp.linalg.solve_triangular(LOOc, np.transpose(NOc), lower = True )
    sandwichc = np.transpose(halfSandwichc) @ halfSandwichc ## Cleverly again, this is NOc @ OOcI @ ONc
    
    return np.trace(sandwichc)/np.trace(sandwich)



def deltaPhi(bStuff, plot = True):
    
    phi, s, tValues = bStuff[0][:,0], bStuff[3], bStuff[5]
    deltaPhi = np.linalg.norm(phi - np.roll(phi, 1, axis=0), axis=1)
    deltaPhi = deltaPhi[1:]/s
    
    if plot: standardPlot(tValues[1:], deltaPhi, r"Time [$t_{Pl}$]", r"Inflaton Change $\frac{\Delta\phi}{s}$", r"$\Delta \phi$ at each Timestep (As Fraction of $s$)")
    
    return deltaPhi


###############################################################################
## -------------------------PLOTTING FUNCTIONS------------------------------ ##
###############################################################################


def autoVisualize(bStuff):
    dimPhi = bStuff[7]
    if dimPhi==1 or dimPhi>3:
        manyPlot(bStuff)
        print("manyPlot identified as best plotting function and executed.")
    if dimPhi==2:
        topoPlot(bStuff)
        print("topoPlot identified as best plotting function and executed.")
    if dimPhi==3:
        phiphiphiPlot(bStuff)
        print("phiphiphiPlot identified as best plotting function and executed.")
    return



## Updated: 5-14-20
def topoPlot(bStuff, vSteps = None, position = True, velocity = False, topo = True, perp = True, downhill = False, evenVSteps = True, vColor = True, phiNums=(0,1), numberEvery = None):
    ## Note: this function takes only two dimensions of phi and its respective vStuff.
    
    phiStuff, V = bStuff[0], extractV(bStuff)
    
    plt.figure(figsize = (10,10))
    
    if vSteps == None: vSteps = (np.max(V) - np.min(V))/4 
    if not position: topo = False
    if vSteps < 0 and evenVSteps == True: 
        evenVSteps = False
        print("Include spacing in topoPlot argument. E.g.: 'vSteps = 0.25'")
    elif evenVSteps == False:
        evenVSteps = True
    colors = V if vColor else None
    
    if position: plt.scatter(phiStuff[:,0,0], phiStuff[:,0,1], marker='+', c = colors, linewidth = 1, cmap = plt.cm.get_cmap('coolwarm'), label='phi vector')
    if velocity: plt.scatter(phiStuff[:,1,0], phiStuff[:,1,1], marker='D', c = colors, linewidth = 1, cmap = plt.cm.get_cmap('coolwarm'), label = 'dphi/dt')
    if vColor: 
        cbar = plt.colorbar()
        cbar.set_label("Potential (V) [Planck units]")
    plt.legend()

    if topo:
        dV0 = extractV(bStuff, index = 0)
        dV1 = extractV(bStuff, index = 1)
    
        if evenVSteps:
            landmarkChunks, tSteps = findLandmarkChunks(bStuff, vSteps)
            dV0 = landmarkChunks[np.arange(1, len(landmarkChunks), 3)]
            dV1 = landmarkChunks[np.arange(2, len(landmarkChunks), 3)]
    
        dV0 = dV0/(np.sqrt(dV0**2 + dV1**2))
        dV1 = dV1/(np.sqrt(dV0**2 + dV1**2))

        if perp: plt.quiver(phiStuff[tSteps,0,0], phiStuff[tSteps,0,1], dV1, -dV0, angles = 'xy', pivot = 'mid', headwidth = 1, scale = 4)
        if downhill: plt.quiver(phiStuff[tSteps,0,0], phiStuff[tSteps,0,1], -dV0, -dV1, angles = 'xy')
    
    if numberEvery is None or numberEvery>0:
        if numberEvery is None: numberEvery = len(phiStuff)/10
        for x in np.arange(0,len(phiStuff),int(numberEvery)): plt.annotate(x, (phiStuff[x,0,0], phiStuff[x,0,1]))
    plt.axis('scaled')
    minX, maxX, minY, maxY = min(phiStuff[:,0,0]), max(phiStuff[:,0,0]), min(phiStuff[:,0,1]), max(phiStuff[:,0,1])
    plt.xlim(minX - (maxX-minX)/20, maxX + (maxX-minX)/20)
    plt.ylim(minY - (maxY-minY)/20, maxY + (maxY-minY)/20)
    plt.xlabel('phi'+str(phiNums[0]))
    plt.ylabel('phi'+str(phiNums[1]))
    title = "Topographic Plot" if position else "d(phi)/dt Visualization"
    plt.title(title, fontfamily = 'times new roman', fontsize = 30)
    
    return None



## Updated: 5-14-20
def waterfallPlot(bStuff, vSteps = 0.25, joltAngle = 30, pause = 1, phiNums = (1,2), figsize = 10):
    ## Note: this function takes only two dimensions of phi and its respective vStuff.
    phiStuff = bStuff[0]
    
    zs, tSteps = extractV(bStuff, returnIndices = True)
    xs = phiStuff[tSteps,0,0]
    ys = phiStuff[tSteps,0,1]
    

    landmarkChunks, tSteps = findLandmarkChunks(bStuff, vSteps) ## tSteps is reassigned.
    dV1 = landmarkChunks[np.arange(1, len(landmarkChunks), 3)]
    dV2 = landmarkChunks[np.arange(2, len(landmarkChunks), 3)]

    flat = np.zeros((len(dV1)))

    for angle in range(int(360/joltAngle)):
        fig = plt.figure(figsize = (figsize,figsize))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("phi" + str(phiNums[0]))
        ax.set_ylabel("phi" + str(phiNums[1]))
        ax.set_zlabel("V")
        ax.scatter(xs, ys, zs, label='phi vector')
        
        xsq, ysq = phiStuff[tSteps,0,0], phiStuff[tSteps,0,1]
        print(np.shape(dV2))
        if vSteps>0: ax.quiver(xsq, ysq, zs[-len(xsq):], dV2, -dV1, flat, length=0.08, normalize = True, pivot = 'middle', color = 'k', arrow_length_ratio = 0)
        ax.view_init(elev=10, azim=angle*joltAngle)
        plt.legend(loc=6)
        plt.title("3-D Landscape Plot", fontfamily="times new roman", fontsize=30, pad=-30)
        plt.pause(pause)       
    return



## Updated: Summer 2019
def phiphiphiPlot(bStuff, phiNums = (0,1,2), colorVel = False):
    phiStuff = bStuff[0]
    
    colors = None
    x, y, z = phiNums
    
    V, indices = extractV(bStuff, returnIndices = True)
    
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')

    colors = V
    if colorVel == True:
        colors = np.sqrt((phiStuff[indices,1,:]**2).sum(axis=1))
    
    sc = ax.scatter(phiStuff[indices,0,x], phiStuff[indices,0,y], phiStuff[indices,0,z], c = colors, cmap = plt.cm.get_cmap('coolwarm'))
    
    ax.set_xlabel("phi" + str(x))
    ax.set_ylabel("phi" + str(y))
    ax.set_zlabel("phi" + str(z))
    cbar = plt.colorbar(sc)
    cbar.set_label("Potential V [arbitrary]")
    plt.title("Three-dimensional phi Plot")
    return



## Updated: Summer 2019
def pseudoAnimatePlot(bStuff, animateStep = 1, vColor = False):
    phiStuff, vStuff, infoVec = bStuff[0], bStuff[1], bStuff[2]
    
    l = len(phiStuff)
    blockLen = len(phiStuff[0])+1
    for i in range(l):
        if (i+1)%animateStep==0: topoPlot((phiStuff[0:i+1], vStuff[0:blockLen*(i+1)], infoVec[:,0:blockLen*(i+1)]), topo = False, vColor = vColor)
    return



## Updated: 5-18-20
def manyPlot(bStuff):
    phiStuff, tValues = bStuff[0], bStuff[5]
    plt.figure(figsize = (10,10))
    for i in range(len(phiStuff[0,0])):
        plt.plot(tValues, phiStuff[:,0,i], label=rf"$\phi_{i}$")
        plt.legend()
    plt.xlabel(r"Time Since Inflation Begins [$t_{Pl}$]")
    plt.ylabel("Inflaton Field Values")
    plt.title("Inflation Fields (Component-Wise) VS. Time")
    return



## Updated: 5-18-20
def VPlot(bStuff):
    standardPlot(bStuff[5], extractV(bStuff), r"Time Since Inflation Begins [$t_{Pl}$]", "Potential (V) [Planck units]", "Inflaton Potential VS. Time")
    return



## Updated: Summer 2019
def findLandmarkChunks(bStuff, steps):
    phiStuff = bStuff[0]
    
    blockLen = len(phiStuff[0])+1
    V = extractV(bStuff)
    vStuff, indices = extractV(bStuff, forceBlock = True)
    
    alertUp = math.ceil(V[0]/steps)*steps
    alertDown = alertUp - steps
    landmarks = np.zeros((len(vStuff)))
    
    for i in range(len(V)):
        iVStuff = i*blockLen
        
        if V[i]>alertUp or V[i]<alertDown:
            
            landmarks[iVStuff:iVStuff+blockLen] = vStuff[iVStuff:iVStuff+blockLen]
            alertUp = math.ceil(V[i]/steps)*steps
            alertDown = alertUp - steps
    
    return landmarks, indices
    


## Updated: Summer 2019
def traceRatioPlot(s, phiPoints, infoVec, eigProblem, OO, decrease = False, log = False, annotate = False):
    ## Only functions inside compressVCOV: plots the cumulative trace in given the current OO, XTX situation.
    
    run = []
    l = len(eigProblem[0])
    
    capp = 1
    
    while capp<l:
        try:
            run = np.append(run, traceRatio(s, phiPoints, infoVec, eigProblem[1][-capp:], OO))
            capp = capp + 1
        except:
            capp = l
    
    if decrease and log:
        plt.semilogy(1-run)
        plt.ylabel("Portion of Trace (Info) Disregarded")
    elif decrease:
        plt.plot(1-run)
        plt.ylabel("Portion of Trace (Info) Disregarded")
    elif log:
        plt.semilogy(run)
        plt.ylabel("Portion of Trace (Info) Kept")
    else:
        plt.plot(run)
        plt.ylabel("Portion of Trace (Info) Kept")
    plt.xlabel("Eigenvalues Included")
    return run



## Updated: Fall 2019
def gridMap(s, corner1, corner2, horPoints, verPoints, vcov = [], phiStuff = [], vStuff = [], infoVec = [], waterfall = True, annotate = True, hardAnnotate = False):
    x1, y1 = corner1[0], corner1[1]
    x2, y2 = corner2[0], corner2[1]
    diffx, diffy = np.abs(x1-x2), np.abs(y1-y2)

    xPoints = np.arange(horPoints)*diffx/horPoints + x1
    yPoints = np.arange(verPoints)*diffy/verPoints + y1
    
    t = 0
    for i in range(len(xPoints)):
        for j in range(len(yPoints)):
            
            if annotate: print("Calculating point", xPoints[i], yPoints[j], "in grid...")
            if len(phiStuff)==0:
                phiStuff = [[xPoints[i], yPoints[j]], [0,0]]
            if len(vcov)==0:
                infoVec = [np.arange(-1,2),np.zeros((3))]
                vcov = VCOV(s, [phiStuff[0]], infoVec)
            if len(vStuff)==0:
                vStuff = getVSkew(vcov)
            if i!=0 or j!=0:
                phiStuff = np.concatenate(( phiStuff, [[[xPoints[i],yPoints[j]],[0,0]]] ))
                vcov = tetrisVCOV(s, phiStuff[:,0], vcov, infoVec)
            
            phiStuff, vStuff, infoVec, vcov = bushwhackOld(phiStuff, vStuff = vStuff, infoVec = infoVec, s = s, vcov = vcov, i = int(max(infoVec[1]))+1, t = t, tValues = np.arange(max(infoVec[1])+2), returnVCOV = True, annotate = hardAnnotate, compress = False)
            t += 1
    
    if waterfall:
        waterfallPlot((phiStuff, vStuff, infoVec), vSteps = -1)
    
    return phiStuff, vStuff, infoVec



## Scatterplots complex numbers in an argand diagram.
## Created: 5-12-20
## Updated: 5-12-20
def scatterComplex(zList, title = None, xlog=False, ylog=False):
    fig, ax = plt.subplots()
    ax.scatter(zList.real, zList.imag)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    if title is not None: plt.title(title)
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    return


## Created: 5-18-20
## Updated: 5-18-20
def standardPlot(x, y, xlabel, ylabel, title, figsize=10, titlesize=30):
    plt.figure(figsize=(figsize,figsize))
    plt.plot(x,y)
    plt.xlabel(xlabel, fontfamily='times new roman', fontsize = int(0.5*titlesize))
    plt.ylabel(ylabel, fontfamily='times new roman', fontsize = int(0.5*titlesize))
    plt.title(title, fontfamily='times new roman', fontsize = titlesize)
    return
    



###############################################################################
## ------------------------------ASSIMILATION------------------------------- ##
###############################################################################
    


## Updated: Spring 2019
def getVSkew(vcov, plot = False):

    L = sp.linalg.cholesky(vcov + noise*np.identity(len(vcov)), lower=True)
    
    y = np.random.normal(0, 1, len(vcov))
    mush = L @ y
    
    if plot: plt.plot(mush)
    
    return mush



## Updated: Summer 2019
def forceMin(s, dimPhi, ppd = 4, varScale = 0.1, topo = False, waterfall = False, phiphiphi = False, annotate = False):
    ## ppd: Points per dimension.
    
    phiStuff = np.zeros((1,2,dimPhi))
    infoVec = [np.arange(-1,dimPhi),np.zeros((dimPhi+1))]
    vStuff = np.zeros((dimPhi+1))
    
    if annotate: print("Randomly generating minimum at phi-origin...")
    
    for i in range(dimPhi*ppd):

        positive = False
        tracker = 0
        while not positive:    
            tracker += 1
            randomVec = np.random.normal(0, s*varScale, dimPhi)
            phiPoint = [randomVec, np.zeros((dimPhi))]
            tempPhiStuff = np.concatenate((phiStuff, [phiPoint]))
            tempInfoVec = np.concatenate((infoVec, [[-1],[i+1]]), axis = 1)
            tempVStuff = vStuff
            
            vcov = VCOV(s, tempPhiStuff[:,0], tempInfoVec)
            quartered = quarter(vcov, -1)
            OO, ON, NN = quartered[0], quartered[1], quartered[3]
            LOO = sp.linalg.cholesky(OO + noise*np.identity(len(OO)), lower = True)
            
            halfSandwich = sp.linalg.solve_triangular(LOO, ON, lower = True)
            gammaC = NN - (np.transpose(halfSandwich) @ halfSandwich) ## Cleverly, this is NO @ OOI @ ON
            
            mu = np.transpose(halfSandwich) @ sp.linalg.solve_triangular(LOO, tempVStuff, lower = True)
            skew = getVSkew(gammaC)
            
            check = mu + skew
            positive = check[0]>0
            if positive:
                phiStuff, vStuff, infoVec = tempPhiStuff, np.append(tempVStuff, check), tempInfoVec
                if annotate: print("Point", i, "found at", phiStuff[-1,0], "in", tracker, "attempts.")

    if topo: topoPlot((phiStuff, vStuff, infoVec), topo = False, vColor = True)
    if waterfall: waterfallPlot((phiStuff, vStuff, infoVec))
    return phiStuff, vStuff, infoVec


## Randomly generates a potential minimum at the origin via dimPhi-dimensional shells of radii multiples of r.
## E.g.: (2, 1, 2, shells = 1, insist = 0.25); E.g.: (5, 3, shells = 2, d2V = 0.005)
## Created: Summer 2019
## Updated: 5-13-20
def forceShellMin(s, V0, dimPhi, ppd = 20, r = None, shells = 2, insist = 0, d2V = 0, fMax = 20):
    
    phiStuff = np.zeros((1,2,dimPhi))
    infoVec = [np.arange(-1,dimPhi),np.zeros((dimPhi+1))]
    vStuff = np.zeros((dimPhi+1))
    
    if r==None: r = s/2
    for shellNum in range(shells):
        
        i = 0
        f = 0 ## Failures
        
        while i < (dimPhi*ppd):
            good = False
            randomVec = np.random.normal(0, r, dimPhi)
            randomVec = (randomVec/np.linalg.norm(randomVec))*r*(shellNum+1)
            phiPoint = [randomVec, np.zeros((dimPhi))]
            tempPhiStuff = np.concatenate((phiStuff, [phiPoint]))
            tempInfoVec = np.concatenate((infoVec, [[-1],[i+1]]), axis = 1)
            tempVStuff = vStuff
                
            vcov = VCOV(s, tempPhiStuff[:,0], tempInfoVec)
            quartered = quarter(vcov, -1)
            OO, ON, NN = quartered[0], quartered[1], quartered[3]
            LOO = sp.linalg.cholesky(OO + noise*np.identity(len(OO)), lower = True)
                
            halfSandwich = sp.linalg.solve_triangular(LOO, ON, lower = True)
            gammaC = NN - (np.transpose(halfSandwich) @ halfSandwich) ## Cleverly, this is NO @ OOI @ ON
            
            mu = np.transpose(halfSandwich) @ sp.linalg.solve_triangular(LOO, tempVStuff, lower = True)
            
            if (mu[0] + 3*np.sqrt(gammaC[0,0]) < d2V*(r**2)/2) or (mu[0] + 3*np.sqrt(gammaC[0,0]) < insist): 
                f += 1
                print(f"FAILURE #{f}: restarting after {i}/{dimPhi*ppd} points generated...")
                i = 0
                
                if f > fMax:
                    print("The method forceShellMin could not randomly generate a potential shell with the given constraints.")
                    return None
                phiStuff = np.zeros((1,2,dimPhi))
                infoVec = [np.arange(-1,dimPhi),np.zeros((dimPhi+1))]
                vStuff = np.zeros((dimPhi+1))
            else:
                while not good:
                
                    skew = getVSkew(gammaC)
                    check = mu + skew
                    good = (check[0]>d2V*(r**2)/2) and (check[0]>insist)
                    if good:
                        phiStuff, vStuff, infoVec = tempPhiStuff, np.append(tempVStuff, check), tempInfoVec
        
                i += 1
    
    bStuff = phiStuff, vStuff*V0, infoVec, s, V0, np.zeros(len(phiStuff))-1, 0, dimPhi, vcov, False
    return bStuff
            


## Created: 5-25-20
## Updated: 5-25-20
def unitize(bStuff, direction = 'unit'):
    phiStuff, vStuff, infoVec, s, V0, dimPhi, vcov = bStuff[0], bStuff[1], bStuff[2].astype(int), bStuff[3], bStuff[4], bStuff[7], bStuff[8]
    if direction.lower() == 'unit':
        phiStuff = phiStuff/s
        vStuff = vStuff/V0
        for i in range(dimPhi):
            dV, arrayLocs = extractV((None, vStuff, infoVec), index=i, returnArrayLocs=True)
            vStuff[arrayLocs] = vStuff[arrayLocs]*s
        if s != 1: vcov = smartVCOV(1, phiStuff[:,0], infoVec)
    
    elif direction.lower() == 'planck':
        phiStuff = phiStuff*s
        vStuff = vStuff*V0
        for i in range(dimPhi):
            dV, arrayLocs = extractV((None, vStuff, infoVec), index=i, returnArrayLocs=True)
            vStuff[arrayLocs] = vStuff[arrayLocs]/s
        if s != 1: vcov = smartVCOV(s, phiStuff[:,0], infoVec)
    
    bStuff = phiStuff, vStuff, infoVec, s, V0, bStuff[5], bStuff[6], dimPhi, vcov, bStuff[9]
    return bStuff



## Created: Spring 2019
## Renamed: bushwhack --> bushwhackOld
## Updated: 5-18-20
def bushwhackOld(phiStuff, s = 0.2, V0 = 1, how = 'til', tValues = np.arange(500)*0.01, mean = 3, vcov = [], vStuff = [], infoVec = [], newStart = [],
                 i = 1, t = 1, tStep = None, adapt = False, tPrime = 0.01, vMin = 0.01, aMin = 0.0001, tMin = 0, tMax = 150,
                 A = [], fullvcov = [], fullxtx = [], vStuffc = [], compress = True, maxMatSize = 3000, cap = -1, percent = 100-1e-10, printTraceRatio = False, plotTraceRatio = False, 
                 ppd = 4, varScale = 0.1,
                 topo = False, waterfall = False, phiphiphi = False, many = False, pseudoAnimate = False, scalePlot = False, VPlot = False, topoVelocity = False, d2 = False, phiphiphiNums = (0,1,2), animateStep = 5, vColor = True,
                 returnVCOV = False, annotate = True, hardAnnotate = False):
    ## h^(1/4)*E_planck is "energy scale of inflation"
    
    ## Setting up phiStuff, vStuff, vcov
    if annotate: print("Formatting and preparing to bushwhack...")
    
    phiStuff = np.array(phiStuff)
    try: phiStuff[0,0,0]
    except: phiStuff = np.array([phiStuff])
    how = how.lower() 
    
    dimPhi = len(phiStuff[0,0])
    blockLen = dimPhi + 1
    whack = True
    
    if tStep==None: tStep = s/(20*V0) ## I'm guessing at this relation til it works.
    
    if len(infoVec)!=0:
        whack = len(vStuff)!=len(infoVec[0]) ## Changed == to != to make findConverger work.
        t = np.max(infoVec[1]) ## Otherwise t stays one.
        infoVec = np.concatenate((infoVec, [np.arange(-1,dimPhi),np.zeros((blockLen))+t]), axis=1) ## Commenting this out made gridMap function properly.
    else:
        infoVec = [np.arange(-1,dimPhi),np.zeros((blockLen))]
        if len(vStuff)!=0:
            return "Program terminated because vStuff was given without corresponding infoVec."
    
    if len(newStart)!=0:
        phiStuff = np.concatenate((phiStuff, [newStart]))
        
    if len(vcov)==0:       
        vcov = VCOV(s, phiStuff[:,0], infoVec)
    
    if len(vStuff)==0: 
        vStuff = getVSkew(vcov)
        
    ## Preparing for the loop
    decomposed = False
    compressed = False
    
    condition = True if how=='til' else (i<len(tValues) and i<tMax)
    if how=='for' and annotate: print("Bushwhacking for", len(tValues), "iterations...")
    elif annotate: print("Bushwhacking until velocity <", vMin, "and acceleration <", aMin, "...")
    
    ## Looping
    while condition:
        
        if compress and len(vcov)>maxMatSize:
            if annotate: print("V covariance matrix exceeded maximum size of", maxMatSize, "by", maxMatSize, ".", '\n', "Compressing...")
            if not compressed:
                fullvcov = vcov
                fullxtx = XTX(s, phiStuff[:,0,:], infoVec)
            A, vcov = compressVCOV(s, phiStuff[:,0,:], infoVec, cap = cap, percent = percent, OO = fullvcov, xtx = fullxtx, returnCompressed = True, printTraceRatio = printTraceRatio, plotTraceRatio = plotTraceRatio, annotate = annotate)
            if annotate: print("V covariance matrix compressed to", len(vcov), "by", len(vcov), ".")
            
            vStuffc = A @ vStuff
            compressed = True
            decomposed = False
        
        if annotate and i%10==0 and not hardAnnotate: print("Beginning iteration", i, "at position", phiStuff[-1,0,:], "and potential", vStuff[-blockLen], "...")
        if hardAnnotate: print("Beginning iteration", i, "at position", phiStuff[-1,0,:], "and potential", vStuff[-blockLen], "...")
        
        if whack:
            if how=='for': tStep = tValues[i]-tValues[i-1]
            if not adapt: phiStuff = np.concatenate( (phiStuff, [eulerCosmo(phiStuff[-1], vStuff[-blockLen:], tStep, V0, mean)]) )
            if adapt: phiStuff = np.concatenate( (phiStuff, [eulerAdapt(phiStuff[-1], vStuff[-blockLen:], tPrime, V0, mean)]) )
            infoVec = np.concatenate((infoVec, [np.arange(-1,dimPhi),np.zeros((blockLen))+t]), axis = 1)
        
            if not compressed:
                vcov = tetrisVCOV(s, phiStuff[:,0], vcov, infoVec)
                
            else:
                vcov, fullvcov = tetrisCompressedVCOV(s, phiStuff[:,0], vcov, infoVec, A, alsoNormal = True, OO = fullvcov)
                fullxtx = tetrisXTX(s, phiStuff[:,0], fullxtx, infoVec)
        
        quartered = quarter(vcov, -1*blockLen)
        OO = quartered[0]
        ON = quartered[1]
        NN = quartered[3]
        
        if not decomposed:
            LOO = sp.linalg.cholesky(OO + noise*np.identity(len(OO)), lower = True)
            decomposed = True
        else:
            LOO = choleskyUpdate(LOO, s, phiStuff[:,0], OO)
        
        halfSandwich = sp.linalg.solve_triangular(LOO, ON, lower = True)
        gammaC = NN - (np.transpose(halfSandwich) @ halfSandwich) ## Cleverly, this is NO @ OOI @ ON
        
        if compressed:
            mu = np.transpose(halfSandwich) @ sp.linalg.solve_triangular(LOO, vStuffc, lower = True) ## Cleverly again, this is NO @ OOI @ xO
        else:
            mu = np.transpose(halfSandwich) @ sp.linalg.solve_triangular(LOO, vStuff, lower = True) ## Cleverly again, this is NOc @ OOIc @ xOc
        
        try: skew = getVSkew(gammaC)
        except: 
            print(sp.linalg.eigh(gammaC))
            return "Conditional covariance matrix 'gammaC' was not positive-definite."
        
        vStuff = np.append(vStuff, mu + skew)
        vStuffc = np.append(vStuffc, mu + skew)

        i = i+1
        t = t+1
        whack = True
        
        if how=='for' or i>2:
            condition = (( np.sqrt(sum(phiStuff[-1,1,:]**2))>vMin or np.sqrt(sum((phiStuff[-1,1,:]-phiStuff[-2,1,:])**2))>aMin or i<tMin ) and i<tMax) if how=='til' else (i<len(tValues) and i<tMax)
    
    infoVec = infoVec.astype(int)
    vStuff = V0*vStuff
    bStuff = phiStuff, vStuff, infoVec, s, V0, tStep
    
    if annotate or hardAnnotate: print("Iteration completed after", i, "points.")
    
    ## Plotting options
    if annotate or hardAnnotate: print("Generating requested visualizations...")
    
    if topo and blockLen==3: topoPlot(bStuff, velocity = topoVelocity, vColor = vColor)
    elif topo: print("Cannot execute a topoPlot in phi-space of dimension", blockLen-1, ".")
    if waterfall and blockLen==3: waterfallPlot(bStuff)
    elif waterfall: print("Cannot execute a waterfallPlot in phi-space of dimension", blockLen-1, ".")
    if pseudoAnimate and blockLen==3: pseudoAnimatePlot(bStuff, animateStep = animateStep)
    elif pseudoAnimate: print("Cannot execute a pseudoAnimatePLot in phi-space of dimension", blockLen-1, ".")
    if phiphiphi and blockLen==4: phiphiphiPlot(bStuff, phiNums = phiphiphiNums)
    elif phiphiphi: print("Cannot execute a phiphiphiPlot in phi-space of dimension", blockLen-1, ".")
    if many: manyPlot(bStuff)       
    if scalePlot: 
        if how=='til': tValues = []
        findScaleFactor(bStuff, tValues = tValues, d2 = d2, plot = True)
    if VPlot: VPlot(bStuff)
    
    if returnVCOV: ## Currently used for gridMap.
        return phiStuff, vStuff, infoVec, vcov
    
    return phiStuff, vStuff, infoVec, s, V0, tStep, mean, dimPhi, vcov, adapt



## Cleaner version of the bushwhack function, currently with no data compression. t: data point # overall, i: this loop iteration #.
## Created: 5-18-20
## Updated: 5-18-20
def bushwhack(bStuff = None, phiStart = None, s = 1, V0 = 1, dimPhi = 2, mean = None,
              tStep = None, coarseness = 1, adapt = False, tPrime = None, vMin = None, aMin = None, tMin = 0, tMax = 500, gofor = None,
              A = [], fullvcov = [], fullxtx = [], vStuffc = [], compress = False, maxMatSize = 3000, cap = -1, percent = 100-1e-10, printTraceRatio = False, plotTraceRatio = False, 
              ppd = 4, varScale = 0.1, visualize = False,
              returnVCOV = False, annotate = True):
    ## h^(1/4)*E_planck is "energy scale of inflation"
    
    ## Setting up phiStuff, vStuff, infoVec, vcov, tValues
    if annotate: print("Formatting and preparing to bushwhack...")
    
    phiStuff, vStuff, infoVec, vcov, tValues, vStuffFiltered, accel, whack, t, i = [], [], [], [], [], [], [], True, 0, 0
    
    if bStuff is None: 
        if phiStart==None: phiStart = np.zeros((2,dimPhi))
        phiStuff = np.array([phiStart])
        infoVec, tValues, t = np.array([np.arange(-1,dimPhi), np.zeros((dimPhi+1))+t]), [0], 1
        vcov = VCOV(1, phiStuff[:,0], infoVec)
        vStuff = getVSkew(vcov)
        if gofor is not None: tMin, tMax = gofor, gofor
    
    else:
        phiStuff, vStuff, infoVec, s, V0, tValues, tempmean, dimPhi, vcov = unitize(bStuff, direction='unit')[:9]
        if mean is None: mean = tempmean
        t = max(infoVec[1])+1
        if gofor is not None: tMin, tMax = t + gofor, t + gofor
        if phiStart is not None: 
            whack = False
            phiStuff = np.concatenate((phiStuff, [phiStart]/s))
            infoVec = np.concatenate((infoVec, [np.arange(-1,dimPhi),np.zeros((dimPhi+1))+t]), axis=1)
            tValues = np.append(tValues, [0])
            vcov = tetrisVCOV(1, phiStuff[:,0], vcov, infoVec)
    
    blockLen, vStuffFiltered = dimPhi + 1, vStuff
    if tStep is None and not adapt:
        tStep = coarseness*s/(100*np.sqrt(V0)) if s<1 else coarseness/(100*np.sqrt(V0))
    if adapt: 
        if tPrime is None: tPrime = s/(100*np.sqrt(V0)) if s<1 else 1/(100*np.sqrt(V0))
    if vMin is None: vMin = 0.008*s
    if aMin is None: aMin = (0.0001/s)*s
    if mean is None: mean = 3*V0
    
    ## Preparing for the loop
    decomposed, condition = False, True

    if annotate: print(f"Bushwhacking until velocity < {vMin} and acceleration < {aMin} OR t leaves ({tMin}, {tMax})...")
    
    ## Looping
    while condition:
        
        if annotate and t%10==0: print(f"Generating point {t} from position {phiStuff[-1,0,:]*s} and potential {vStuff[-blockLen]*V0}...")
        
        if whack:
            if not adapt: 
                phiNew = eulerCosmo(phiStuff[-1], vStuffFiltered[-blockLen:], tStep, s, V0, mean)
            else: 
                phiNew, tStep, accelNow = eulerAdapt(phiStuff[-1], vStuffFiltered[-blockLen:], tPrime, s, V0, mean, i, accel)
                tPrime = tStep
                accel = [accelNow] if i==0 else np.concatenate((accel, [accelNow]))
            
            phiStuff = np.concatenate((phiStuff, [phiNew]))
            infoVec = np.concatenate((infoVec, [np.arange(-1,dimPhi),np.zeros((blockLen))+t]), axis = 1)
            tValues = np.append(tValues, [tValues[-1] + tStep])
            vcov = tetrisVCOV(1, phiStuff[:,0], vcov, infoVec)
        
        quartered = quarter(vcov, -1*blockLen)
        OO, ON, NN = quartered[0], quartered[1], quartered[3]
        
        if not decomposed:
            LOO = sp.linalg.cholesky(OO + noise*np.identity(len(OO)), lower = True)
            Lvcov = sp.linalg.cholesky(vcov + noise*np.identity(len(vcov)), lower = True)
            decomposed = True
        else:
            LOO = choleskyUpdate(LOO, None, None, OO)
            Lvcov = choleskyUpdate(LOO, None, None, vcov)
        
        halfSandwich = sp.linalg.solve_triangular(LOO, ON, lower = True)
        gammaC = NN - (np.transpose(halfSandwich) @ halfSandwich) ## Cleverly, this is NO @ OOI @ ON
        
        mu = np.transpose(halfSandwich) @ sp.linalg.solve_triangular(LOO, vStuff, lower = True) ## Cleverly again, this is NO @ OOI @ xO
        skew = getVSkew(gammaC)
        
        vStuff = np.append(vStuff, mu + skew)
        vStuffFiltered = wienerFilter(vcov, Lvcov, vStuff)
        
        vCondition, aCondition = np.linalg.norm(phiStuff[-1,1,:])>vMin, np.linalg.norm(phiStuff[-1,1,:]-phiStuff[-2,1,:])>aMin
        #if annotate and not vCondition and t%1==0: print(Fore.GREEN + f"Minimum velocity condition currently met (t = {t}: {np.linalg.norm(phiStuff[-1,1,:])}).", Style.RESET_ALL)
        #if annotate and not aCondition and t%1==0: print(Fore.BLUE + f"Minimum acceleration condition currently met (t = {t}: {np.linalg.norm(phiStuff[-1,1,:]-phiStuff[-2,1,:])}).", Style.RESET_ALL)
        
        t, i, whack = t+1, i+1, True
        if t>2: condition = ( vCondition or aCondition or t<tMin ) and t<tMax
    
    ## Loop finished.
    if annotate: print("Iteration completed after", t, "data points generated.")
    bRaw = phiStuff, vStuff, infoVec.astype(int), s, V0, tValues, mean, dimPhi, vcov, adapt
    bStuff = phiStuff, vStuffFiltered, infoVec.astype(int), s, V0, tValues, mean, dimPhi, vcov, adapt
    bRaw = unitize(bRaw, direction='planck')
    bStuff = unitize(bStuff, direction='planck')
    
    if visualize: autoVisualize(bStuff)
    
    return bStuff

## bStuff[0]: phiStuff - All the information about inflaton field "phi" along path.
## bStuff[1]: vStuff - All the numerical values of the potential "V" along path.
## bStuff[2]: infoVec - Tells what each number in vStuff represents, which derivative at which time step.
## bStuff[3]: s - The covariance length scale of the simulated potential.
## bStuff[4]: V0 - The standard deviation (height) of simulated potential.
## bStuff[5]: tValues - The set of time values corresponding to each data point.
## bStuff[6]: mean - An offset which allows us to continue solving for equations of motion even once V<0.
## bStuff[7]: dimPhi - The number of inflaton fields solved in the path.
## bStuff[8]: vcov - The covariance matrix of constraints on the potential
## bStuff[9]: adapt - Whether or not adaptive timesteps were used.



def bushwhackSmart(bStuff = None, phiStart = None, s = 1, V0 = 1, dimPhi = 2, mean = None,
                   tStep = None, coarseness = 1, adapt = False, detect = True, tPrime = None, vMin = None, aMin = None, tMin = 0, tMax = 500, gofor = None,
                   A = [], fullvcov = [], fullxtx = [], vStuffc = [], compress = False, maxMatSize = 3000, cap = -1, percent = 100-1e-10, printTraceRatio = False, plotTraceRatio = False, 
                   ppd = 4, varScale = 0.1, visualize = False,
                   returnVCOV = False, annotate = True):
    ## h^(1/4)*E_planck is "energy scale of inflation"... HOW DID I KNOW THIS
    
    ## Setting up various important variables.
    if annotate: print("Formatting and preparing to bushwhack...")
    
    phiStuff, vStuff, infoVec, vcov, tValues, vStuffFiltered, accel, whack, t, i = [], [], [], [], [], [], [], True, 0, 0
    
    if bStuff is None: 
        if phiStart==None: phiStart = np.zeros((2,dimPhi))
        phiStuff = np.array([phiStart])
        infoVec, tValues, t = np.array([np.arange(-1,dimPhi), np.zeros((dimPhi+1))+t]).astype(int), [0], 1
        vcov = smartVCOV(1, phiStuff[:,0], infoVec)
        vStuff = getVSkew(vcov)
        if gofor is not None: tMin, tMax = gofor, gofor
    
    else:
        phiStuff, vStuff, infoVec, s, V0, tValues, tempmean, dimPhi, vcov = unitize(bStuff, direction='unit')[:9]
        if mean is None: mean = tempmean
        t = max(infoVec[1])+1
        if gofor is not None: tMin, tMax = t + gofor, t + gofor
        if phiStart is not None: 
            whack = False
            phiStuff = np.concatenate((phiStuff, np.array([phiStart])/s))
            infoVec = np.concatenate((infoVec, [np.arange(-1,dimPhi),np.zeros((dimPhi+1))+t]), axis=1)
            tValues = np.append(tValues, [0])
            accel = [np.zeros(dimPhi)]
            vcov = tetrisVCOV(1, phiStuff[:,0], vcov, infoVec)
            
    blockLen, vStuffFiltered = dimPhi + 1, vStuff
    if tStep is None and not adapt:
        tStep = coarseness*s/(100*np.sqrt(V0)) if s<1 else coarseness/(100*np.sqrt(V0))
    if adapt: 
        if tPrime is None: tPrime = s/(100*np.sqrt(V0)) if s<1 else 1/(100*np.sqrt(V0))
    if vMin is None: vMin = 0.008*s
    if aMin is None: aMin = (0.0001/s)*s
    if mean is None: mean = 3*V0
    
    ## Preparing for the loop
    decomposed, condition = False, True

    if annotate: print(f"Bushwhacking until velocity < {vMin} and acceleration < {aMin} OR t leaves ({tMin}, {tMax})...")
    
    ## Looping
    while condition:

        if annotate and t%10==0: print(f"Generating point {t} from position {phiStuff[-1,0,:]*s} and potential {vStuff[-blockLen]*V0}...")
        
        if whack:
            if not adapt and not detect: 
                phiNew = eulerCosmo(phiStuff[-1], vStuffFiltered[-blockLen:], tStep, s, V0, mean)
            elif adapt: 
                phiNew, tStep, accelNow = eulerAdapt(phiStuff[-1], vStuffFiltered[-blockLen:], tPrime, s, V0, mean, i, accel)
                tPrime = tStep
                accel = [accelNow] if len(accel)==0 else np.concatenate((accel, [accelNow]))
            elif detect:
                phiStuff, tStep = eulerDetect(phiStuff, vStuff, tStep, s, V0, mean, tValues)
            if not detect: phiStuff = np.concatenate((phiStuff, [phiNew]))
            infoVec = np.concatenate((infoVec, np.array([np.arange(-1,dimPhi),np.zeros((blockLen))+t]).astype(int)), axis = 1)
            tValues = np.append(tValues, [tValues[-1] + tStep])
            vcov = tetrisSmartVCOV(1, phiStuff[:,0], infoVec, vcov)

        quartered = quarter(vcov, -1*blockLen)
        OO, ON, NN = quartered[0], quartered[1], quartered[3]
        
        if not decomposed:
            LOO = sp.linalg.cholesky(OO + noise*np.identity(len(OO)), lower = True)
            Lvcov = sp.linalg.cholesky(vcov + noise*np.identity(len(vcov)), lower = True)
            decomposed = True
        else:
            LOO = choleskyUpdate(LOO, None, None, OO)
            Lvcov = choleskyUpdate(LOO, None, None, vcov)

        halfSandwich = sp.linalg.solve_triangular(LOO, ON, lower = True)
        gammaC = NN - (np.transpose(halfSandwich) @ halfSandwich) ## Cleverly, this is NO @ OOI @ ON
        
        mu = np.transpose(halfSandwich) @ sp.linalg.solve_triangular(LOO, vStuff, lower = True) ## Cleverly again, this is NO @ OOI @ xO
        skew = getVSkew(gammaC)
        
        vStuff = np.append(vStuff, mu + skew)
        vStuffFiltered = wienerFilter(vcov, Lvcov, vStuff)
        
        vCondition, aCondition = np.linalg.norm(phiStuff[-1,1,:])>vMin, np.linalg.norm(phiStuff[-1,1,:]-phiStuff[-2,1,:])>aMin
        
        t, i, whack = t+1, i+1, True
        if t>2: condition = ( vCondition or aCondition or t<tMin ) and t<tMax

    ## Loop finished.
    if annotate: print("Iteration completed after", t, "data points generated.")
    bRaw = phiStuff, vStuff, infoVec.astype(int), s, V0, tValues, mean, dimPhi, vcov, adapt
    bStuff = phiStuff, vStuffFiltered, infoVec.astype(int), s, V0, tValues, mean, dimPhi, vcov, adapt
    bRaw = unitize(bRaw, direction='planck')
    bStuff = unitize(bStuff, direction='planck')
    
    if visualize: autoVisualize(bStuff)    
    return bStuff



###############################################################################
## ----------------------------FUNCTION TESTERS----------------------------- ##
###############################################################################


## A simple, reliable method for generating paths converging to a zero-potential minimum.
## E.g.: PROVIDE
## Created: April 2020
## Updated: 5-14-20
def findConverger(s, V0, phiStart, testargs = [5,5,5], gofor = None, adapt = False):
    
    ## Set up a minimum at the origin.
    dimPhi = len(phiStart[0])
    minInfo = forceShellMin(s, V0, dimPhi, r = s/2, shells = 1, insist = 0.25)
    print("Minimum successfully generated. Sending to bushwhack...")
    
    ## Send minimum information to bushwhack function with new starting position.
    b = bushwhackSmart(bStuff = minInfo, phiStart = phiStart, gofor = gofor, adapt = adapt)
    
    minPoints = 20*dimPhi+2
    phiStuff = b[0][minPoints:]
    vStuff = b[1][(minPoints+2*dimPhi):]
    infoVec = np.array( [b[2][0][(minPoints+2*dimPhi):],  b[2][1][(minPoints+2*dimPhi):] - minPoints] )
    tValues = b[5][minPoints:]
    
    bStuff = phiStuff, vStuff, infoVec, b[3], b[4], tValues, b[6], b[7], b[8], b[9]
    convergenceTest = convergingToOrigin(bStuff, testargs[0], testargs[1], testargs[2])
    if np.all(convergenceTest): 
        print(Fore.GREEN + "Convergent path likely.", Style.RESET_ALL)
    else:
        print(Fore.RED + "Divergent path possible.", Style.RESET_ALL)
    
    return bStuff


## A method for testing number of e-folds of inflation as a function of s, V0, and starting positions.
## E.g.: testEFolds([1,2], [1], [0.25, 0.5, 1])... startRatios are how far out to drop phi-ball as fraction of "s".
## Created: 4-19-20
## Updated: 5-14-20 
def testEFolds(sValues, V0Values, startRatios, dimPhi = 2, tStepPrecision = 20):
    storage = []
    i = 0
    for s in sValues:
        for V0 in V0Values:
            for startRatio in startRatios:
            
                start = np.append([startRatio*s], np.zeros(dimPhi-1))
                start = np.append([start], [np.zeros(dimPhi)], axis=0)
                print(f"Gathering and analysing data with s = {s} and starting position {start} (path {i+1}).")
            
                b = findConverger(s, V0, start)
                scale = findScaleFactor(b, plot=False)
                eFolds = np.log(scale[-1]) - np.log(scale[0])
                together = np.array([s, start, b, scale, eFolds])
            
                if i==0: storage = [together]
                else: storage = np.concatenate((storage, [together]))
                i += 1
    
    return storage
            


## Created: 5-20-20
## Updated: 5-20-20 GENERALIZE SOON
def simulateRegularPaths(sValues, trials):
    storage = []
    i = 0
    for s in sValues:
        for trial in range(trials):
            b = bushwhack(s = s, tMax = 750)
            storage = [b] if i==0 else np.concatenate((storage, [b]))
            i += 1
    return storage

"""
def gaussian0(xData, sig, h):
    return (h/np.sqrt(2*np.pi*sig**2))*np.exp(-(xData**2)/(2*sig**2))
""" 
"""
#popt, pcov = curve_fit(gaussian0, np.linspace(1.6,24.,20)/8.0, tsa[4,1:])
#plt.plot(np.linspace(1.6,24.,20)/8.0, gaussian0(np.linspace(1.6,24.,20)/8.0, popt[0], popt[1]))

# TED: Do whatever you have to do to make phi list and V list

## We have all information stored in 'ts8', particularly 'ts8[1][0]'
## philist = ts8[1][0][0][0:161,0,:]
## vlist = vlist[0:161], originally 'extractV(ts8[1][0][1], ts8[1][0][2])'
"""
"""
def chiSquared(h):
    ## h: the coefficients of quadratic expansion - can be specified to fit N-D paraboloid-like shape 
    
    n = np.sqrt(len(h)).astype(int)
    hmatrix = np.reshape(h,(n,n))
    temp = 0
    for a in range(len(philist)):
        temp += (vlist[a]-0.5*(philist[a] @ hmatrix @ philist[a]))**2
    return temp

    #htest = minimize(chiSquared, np.zeros(64))
    #hanswer = np.reshape(htest['x'], (8,8))
    #happlied = np.diag(0.5*(philist @ hanswer @ philist.T))
"""



"""
start = time.time()
smartVCOV(1, b[0][:,0], b[2])
end = time.time()
print(end-start)

start2 = time.time()
VCOV(1, b[0][:,0], b[2])
end2 = time.time()
print(end2 - start2)
"""

    





###############################################################################
## -------------------------------TO DO LIST-------------------------------- ##
###############################################################################
    
# 5. Make an animation of topoPlot (follow pseudoAnimatePlot)
# 10. Create a master plot showcasing many visualizations for any path.
# 14. Create reasonable, sophisticated methods for identifying end of path... 
#     perhaps soon after aH begins to decrease?
# 15. Spline interpolation between points as output for some of our data?
# 16. Big undertaking: use canned differential equation solver for random model.

###############################################################################
## -----------------------------DONE THIS WEEK------------------------------ ##
###############################################################################

