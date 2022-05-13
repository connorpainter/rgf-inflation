import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate
#from sklearn.preprocessing import normalize
#from numba import jit
import math

mpl.rcParams['text.usetex'] = False
s, V0 = 7, 1e-9
defaultMMCVals = 10**np.array([-11.4331129107601,-11.0209589994406,-10.8103553863187,-10.6473001974953,-10.5097113433471,
                  -10.3877731294661,-10.2754640883194,-10.1677245681175,-10.0575191091359,-9.90201016940867])

## Test of differential equation solver on a quadratic potential function in phi.
def V(cVals, phi): return 1/2*sum(cVals*phi**2)
def dV(cVals, phi): return cVals*phi
def pdV(cVals, phi, i): return cVals[i]*phi[i]
def pdpdV(cVals, phi, i, j): return 0 if i!=j else cVals[i] 



def eulerNDQV(dimPhi, tStep = None, cVals = None, coarseness = 1, iMax = 1e3, alpha = 1/20, factorSV0 = True):
    if cVals is None: cVals = np.random.normal(loc = 1, scale = 0.3, size = dimPhi)
    if factorSV0: cVals *= V0/s**2
    phiStart = np.random.normal(size = dimPhi)
    phiStart = s*phiStart/np.linalg.norm(phiStart)
    phiStuff = np.zeros((1,2,dimPhi))
    phiStuff[0,0] = phiStart
    
    if tStep is None: tStep = coarseness*s/(100*np.sqrt(V0)) if s<1 else coarseness/(100*np.sqrt(V0))
    
    i, t = 1, np.array([0]) 
    while i < iMax:
        if i%500==0: print(f"At iteration {i}, we are at phi = {phiStuff[-1,0]}.")
        
        phi = phiStuff[-1,0]
        dphi = phiStuff[-1,1]
        ddphi = -np.sqrt(24*np.pi*(1/2*np.dot(dphi,dphi) + V(cVals, phi)))*dphi - dV(cVals, phi)
        
        phiNew = phiStuff[-1] + np.array((dphi, ddphi))*tStep
        phiStuff = np.concatenate((phiStuff, [phiNew]))
        
        if i>5 and np.linalg.norm(phiStuff[-1,1]*tStep) < alpha:
                tStep *= 1.01
                print(f"Time step increased to {tStep} at i = {i}.")
        if i>30 and np.linalg.norm(phiStuff[-30,0]) < alpha*s: i = iMax
        i += 1
        t = np.append(t, [t[-1] + tStep])
        
    return phiStuff, t, cVals, dimPhi



def getV(ndqv, plot=True):
    cVals = ndqv[2]
    phi = ndqv[0][:,0]
    dimPhi = len(ndqv[0][0,0])
    
    VList = 1/2*np.sum(cVals*phi**2, axis=1)
    if plot:
        plt.plot(ndqv[1], VList)
        plt.xlabel(r"Time [$t_{Pl}$]")
        plt.ylabel("Inflaton Potential")
    return VList
def getdV(ndqv, plot=True): 
    
    dVList = np.zeros((len(ndqv[1]), ndqv[3]))
    for i in range(len(ndqv[1])):
        dVList[i] = dV(ndqv[2], ndqv[0][i,0])
    
    if plot: plt.plot(ndqv[1], dVList) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel("Components of Gradient of Potential")
    
    return dVList
def getpdV(ndqv, i, plot=True): 
    
    pdVList = getdV(ndqv, plot=False)[:,i]
    if plot: plt.plot(ndqv[1], pdVList) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel(f"Partial Derivative of Potential (Dimension {i})")
    
    return getdV(ndqv, plot=False)[:,i]
def getpdpdV(ndqv, i, j, plot=True): 
    
    pdpdVList = np.zeros(len(ndqv[1]))
    for t in range(len(ndqv[1])):
        pdpdVList[t] = pdpdV(ndqv[2], ndqv[0][t,0], i, j)
        
    if plot: plt.plot(ndqv[1], pdpdVList) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel(f"Mixed Partial Derivatives of Potential (Dimensions {i}, {j})")
    
    return pdpdVList



def getPhi(ndqv, i=0, plot=True):
    dimPhi = ndqv[3]
    if plot:
        plt.plot(ndqv[1], ndqv[0][:,i])
        plt.title(f"{dimPhi}-Dimensional Quadratic Inflaton")
        plt.xlabel(r"Time [$t_{Pl}$]")
        if i==0: plt.ylabel("Inflaton Field Components")
        if i==1: plt.ylabel("Inflaton Field Component Velocities")
    return ndqv[0][:,i]



def getH(ndqv, units='default', plot=True): 
    
    dphi, V, H = getPhi(ndqv, i=1, plot=False), getV(ndqv, plot=False), []
    if units.lower()=='default': H = np.sqrt(8*np.pi/3*(1/2*np.linalg.norm(dphi, axis=1)**2 + V))
    elif units.lower()=='mmc': H = np.sqrt(1/3*(1/2*np.linalg.norm(dphi, axis=1)**2 + V))
    else: 
        print("Invalid units, idiot!")
        return None
    if plot:
        if len(ndqv)==4: plt.plot(ndqv[1], H, 'r--')
        elif len(ndqv)==3: plt.plot(ndqv[0].t, H, 'r--')
        plt.xlabel(r"Time [$t_{Pl}$]")
        plt.ylabel("Hubble Parameter")
        plt.grid()
    return H



def findScaleFactor(ndqv, units='default', atf = 1.9236e-32, nReheat = 1, plot = True): 
    H = getH(ndqv, units = units, plot=False)
    tSteps = np.roll(ndqv[1], -1) - ndqv[1]
    ln_a = np.cumsum(H[:-1]*tSteps[:-1])
    
    atf = atf*nReheat
    ln_a = (ln_a - ln_a[-1]) + np.log(atf)
    
    if plot: plt.plot(ndqv[1][:-1], ln_a) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel("Scale Factor") and plt.grid()
    
    return np.exp(ln_a)



def getEpsilon(ndqv, units='mmc', plot=False): 
    
    dphi, H = getPhi(ndqv, i=1, plot=False)[:-1], getH(ndqv, units=units, plot=False)[:-1]
    epsilon = 1/(2*H**2)*np.linalg.norm(dphi, axis=1)**2
    if plot: plt.semilogy(ndqv[1][:-1], epsilon) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel(r"$\epsilon$") and plt.grid()
    
    return epsilon



def couplingTensor(ndqv, units = 'mmc'): 
    dimPhi = ndqv[3]
    coupler = np.zeros((dimPhi, dimPhi, len(ndqv[0])-1))
    H = getH(ndqv, units = units, plot = False)[:-1]
    a = findScaleFactor(ndqv, units = units, plot = False)
    dphi = getPhi(ndqv, i=1, plot=False)[:-1]
    epsilon = getEpsilon(ndqv, units=units)
    
    for i in range(dimPhi):
        for j in range(dimPhi):
            term1 = getpdpdV(ndqv, i, j, plot=False)[:-1]/H**2
            term2 = a/H**2*dphi[:,i]*getpdV(ndqv, j, plot=False)[:-1]
            term3 = a/H**2*dphi[:,j]*getpdV(ndqv, i, plot=False)[:-1]
            term4 = a**2*(3 - epsilon)*dphi[:,i]*dphi[:,j]
            
            coupler[i,j] = term1 + term2 + term3 + term4
            
    return coupler



def scalarPower(ndqv, points = 100, kVals = [], units = 'mmc', C = (0,1)): 
    
    ## Find psi and its derivative.
    dimPhi = ndqv[3]
    times = ndqv[1][:-1]
    print("Finding and interpolating a(t), H(t), epsilon(t), dphi(t).")
    a = sp.interpolate.interp1d(times, findScaleFactor(ndqv, units=units, plot=False))
    H = sp.interpolate.InterpolatedUnivariateSpline(times, getH(ndqv, units=units, plot=False)[:-1])
    dH = H.derivative()
    epsilon = sp.interpolate.interp1d(times, getEpsilon(ndqv, units=units))
    dphi = getPhi(ndqv, i=1, plot=False)[:-1]
    dphiFuncs = np.empty((dimPhi), dtype=object)
    for i in range(dimPhi): dphiFuncs[i] = sp.interpolate.interp1d(times, dphi[:,i])
    dphi = dphiFuncs
    
    print("Computing coupling tensor.")
    coupler = couplingTensor(ndqv, units=units)
    
    print("Interpolating coupling tensor.")
    newCoupler = np.empty((dimPhi, dimPhi), dtype=object)
    for i in range(dimPhi):
        for j in range(dimPhi):
            func = sp.interpolate.interp1d(times, coupler[i,j])
            newCoupler[i,j] = func        
    coupler = newCoupler
    
    ## Evaluates the coupling tensor at a particular value in time (as it is a matrix of interpolating functions).
    def callCoupler(t):
        thisCoupler = np.zeros((dimPhi,dimPhi))
        for i in range(dimPhi):
            for j in range(dimPhi):
                thisCoupler[i,j] = coupler[i,j](t)
        return thisCoupler
    
    ## Differential equation used in solve_ivp.
    @jit(nopython=True)
    def f(t, psiFlat):
        psiStuff = np.reshape(psiFlat, (2, dimPhi, dimPhi))
        psi, dpsi = psiStuff[0], psiStuff[1]
        
        term1 = dH(t)/H(t)*dpsi
        term2 = H(t)*(epsilon(t) - 1)*dpsi
        term3 = H(t)**2*(2 - epsilon(t) - k**2/(a(t)**2*H(t)**2))*psi
        term4 = -1*H(t)**2*(callCoupler(t) @ psi)
        ddpsi = term1 + term2 + term3 + term4
        return np.append(dpsi.flatten(), ddpsi.flatten())
    
    ## Given a wavenumber (k), solve differential equation (3.14) in MultiModeCode paper.
    ## Outputs N^2 psi functions of time, N^2 "field-space" functions of time (3.19), and the "adiabatic curvature" function of time.
    def scalarPowerPoint(k):
        print(f"Solving differential equation from {times[0]} to {times[-1]}.")
        psiStart = np.identity(dimPhi)*1/np.sqrt(2*k)*(C[0] + C[1])
        dpsiStart = np.identity(dimPhi)*1j/(a(0)**2*H(0))*(C[0] - C[1])*np.sqrt(k/2)
        psiStuff = sp.integrate.solve_ivp(f, (times[0], times[-1]), np.append(psiStart.flatten(), dpsiStart.flatten()) ) 
        
        ## Use psi to compute various power spectra.
        print("Computing power spectra.")
        psi = np.reshape(psiStuff.y, (2, dimPhi, dimPhi, len(psiStuff.t)))[0]
        aDiscrete = a(psiStuff.t)
        fieldSpace = np.empty((dimPhi, dimPhi, len(psiStuff.t)), dtype=complex)
        fieldSpaceFuncs = np.empty((dimPhi,dimPhi), dtype=object)
        for i in range(len(psiStuff.t)):
            fieldSpace[:,:,i] = k**3/(2*np.pi**2*aDiscrete[i]**2)*psi[:,:,i] @ np.matrix(psi[:,:,i], dtype=complex).H
        for i in range(dimPhi):
            for j in range(dimPhi):
                fieldSpaceFuncs[i,j] = sp.interpolate.interp1d(psiStuff.t, fieldSpace[i,j,:])
    
        dphiAdjusted = np.empty((len(psiStuff.t), dimPhi))
        for i in range(dimPhi): dphiAdjusted[:,i] = dphi[i](psiStuff.t)
        omega = normalize(dphiAdjusted, axis=1)
        epsilonDiscrete = epsilon(psiStuff.t)
        adiabaticCurv = np.empty((len(psiStuff.t)), dtype=complex)
        for i in range(len(psiStuff.t)):
            adiabaticCurv[i] = 1/(2*epsilonDiscrete[i])*omega[i] @ fieldSpace[:,:,i] @ np.transpose(omega[i]) if epsilonDiscrete[i]!=0 else np.nan
        adiabaticCurv = sp.interpolate.interp1d(psiStuff.t, adiabaticCurv)
        
        return psiStuff, fieldSpaceFuncs, adiabaticCurv
    
    if len(kVals)==0:
        evenTimes = np.linspace(times[0], times[-1], points)
        kVals = a(evenTimes)*H(evenTimes)/1e20
    else: points = len(kVals)
    sols, fieldSpaces, ACs =  np.empty(points, dtype=object), np.empty((points, dimPhi, dimPhi), dtype=object), np.empty(points, dtype=object)
    for i in range(len(kVals)):
        k = kVals[i]
        print(f"Computing all the data for k = {k}.")
        sol, fieldSpace, AC = scalarPowerPoint(k)
        sols[i], fieldSpaces[i,:,:], ACs[i] = sol, fieldSpace, AC
        
    return kVals, sols, fieldSpaces, ACs
    
    
    
def plotAC(spStuff, t = None):
    kVals, sols, fieldSpaces, ACs, intervals = spStuff
    ACdiscrete = np.empty((len(kVals)),dtype=complex)
    if t is None: t = math.floor(max(intervals[:,1]))
    for i in range(len(kVals)):
        ACdiscrete[i] = ACs[i](t)
    plt.loglog(kVals, ACdiscrete, 'r+', label=rf'$t = {t} t_P$') and plt.xlabel("k") and plt.ylabel("Power") and plt.title("Adiabatic Curvature Power Spectrum")
    plt.legend()
    return ACdiscrete
        


def plotACEvolution(spStuff, kIndices, points = 100):
    kVals, sols, fieldSpaces, ACs, intervals = spStuff
    plt.figure(figsize=(8,8))
    for i in kIndices:
        sampleTimes = np.linspace(intervals[i,0], intervals[i,1], points)
        samplePowers = ACs[i](sampleTimes)
        plt.semilogy(sampleTimes, samplePowers, label = f"k = {kVals[i]}")
    plt.xlabel("Time") and plt.ylabel("Power") and plt.legend()
    return


###############################################################################
#################  "CANNED" DIFFERENTIAL EQUATION SOLVER  #####################
###############################################################################

 
    
def expertNDQV(dimPhi, units = 'default', tEnd = None, cVals = None, phiStart = None, startAtRest = True, startSlowRolling = False, 
               method = 'Radau', max_step = None, factorSV0 = True):
    
    if cVals is None: cVals = np.random.normal(loc=1, scale=0.3, size=dimPhi)
    if factorSV0: cVals *= V0/s**2
    
    if phiStart is None: 
        phiStart = np.random.normal(size = dimPhi)
        phiStart = s*phiStart/np.linalg.norm(phiStart)
    if len(phiStart)==dimPhi:
        dphiStart = []
        if startSlowRolling: 
            VStart, dVStart = V(cVals, phiStart), dV(cVals, phiStart)
            if units=='default': dphiStart = np.sqrt(-VStart + np.sqrt(VStart**2 + np.linalg.norm(dVStart)**2/(12*np.pi)))*(-1*normalize([dVStart]))
            if units=='mmc': dphiStart = np.sqrt(-VStart + np.sqrt(VStart**2 + np.linalg.norm(dVStart)**2/(3/2)))*(-1*normalize([dVStart]))
        elif startAtRest: 
            dphiStart = np.zeros(dimPhi)
        phiStart = np.append(phiStart, dphiStart)
    
    if tEnd is None: tEnd = dimPhi**(1/4)*100*s/np.sqrt(V0) ## Total guess
    if max_step is None: max_step = tEnd/100
    
    def f(t, phiStuff):
        phiStuff = phiStuff.reshape((2,dimPhi))
        phi, dphi = phiStuff[0], phiStuff[1]
        ddphi = []
        if units=='default': ddphi = -np.sqrt(24*np.pi*(1/2*np.dot(dphi,dphi) + V(cVals, phi)))*dphi - dV(cVals, phi)
        elif units=='mmc': ddphi = -np.sqrt(3*(1/2*np.dot(dphi,dphi) + V(cVals, phi)))*dphi - dV(cVals, phi)
        return np.append(dphi, ddphi).flatten()
    
    phiStuff = sp.integrate.solve_ivp(f, [0, tEnd], phiStart, method = method, max_step = max_step)
    
    return phiStuff, cVals, dimPhi


def getTimes(ndqv, plot=False, log=False): 
    
    t = ndqv[0].t
    if plot: 
        if not log: plt.plot(t)
        else: plt.semilogy(t)
        plt.xlabel("Evaluation Number") and plt.ylabel(r"Time [$t_{Pl}$]")
        
    return t
def getV2(ndqv, plot=False): 
    
    cVals, dimPhi = ndqv[1], ndqv[2]
    phi = np.transpose(ndqv[0].y[:dimPhi])
    V = 1/2*np.sum(cVals*phi**2, axis=1)
    if plot: plt.plot(ndqv[0].t, V) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel("Inflaton Potential")
    
    return V
def getdV2(ndqv, plot=False): 
    
    phiStuff, cVals, dimPhi, t = ndqv[0], ndqv[1], ndqv[2], getTimes(ndqv)
    dVList = np.zeros((len(t), dimPhi))
    for i in range(len(t)):
        dVList[i] = dV(cVals, phiStuff.y[:dimPhi,i])
    
    if plot: plt.plot(t, dVList) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel("Components of Gradient of Potential")
    
    return dVList
def getpdV2(ndqv, i, plot=False): 
    
    pdVList = getdV2(ndqv)[:,i]
    if plot: plt.plot(getTimes(ndqv), pdVList) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel(f"Partial Derivative of Potential (Dimension {i})")
    
    return pdVList
def getpdpdV2(ndqv, i, j, plot=False): 
    
    dimPhi, tVals = ndqv[2], getTimes(ndqv)
    pdpdVList = np.zeros(len(tVals))
    for t in range(len(tVals)):
        pdpdVList[t] = pdpdV(ndqv[1], ndqv[0].y[:dimPhi,0], i, j)
    
    if plot: plt.plot(tVals, pdpdVList) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel(f"Mixed Partial Derivatives of Potential (Dimensions {i}, {j})")
    
    return pdpdVList
    


def getPhi2(ndqv, i=0, plot=False):
    dimPhi = ndqv[2]
    phi = ndqv[0].y[0:dimPhi] if i==0 else ndqv[0].y[dimPhi:]
    if plot: 
        [plt.plot(ndqv[0].t, phi[dim]) for dim in range(dimPhi)]
        plt.xlabel(r"Time [$t_{Pl}$]")
        plt.title("My Code Output")
        if i==0: plt.ylabel("Inflaton Field Components")
        if i==1: plt.ylabel("Inflaton Field Component Velocities")
    return np.transpose(phi)
    


def getH2(ndqv, units="default", plot=False):
    
    t, dphi, V, H = getTimes(ndqv), getPhi2(ndqv, i=1, plot=False), getV2(ndqv, plot=False), []
    if units.lower()=='default': H = np.sqrt(8*np.pi/3*(1/2*np.linalg.norm(dphi, axis=1)**2 + V))
    elif units.lower()=='mmc': H = np.sqrt(1/3*(1/2*np.linalg.norm(dphi, axis=1)**2 + V))
    if plot: plt.plot(t, H) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel("Hubble Parameter") and plt.grid()
    
    return H



def findScaleFactor2(ndqv, units='default', atf = 1.9236e-32, nReheat = 1, log = False, plot = False): 
    
    t, H = getTimes(ndqv), getH2(ndqv, units=units)
    tSteps = np.roll(t, -1) - t
    ln_a = np.cumsum(H[:-1]*tSteps[:-1])
    
    atf = atf*nReheat
    ln_a = (ln_a - ln_a[-1]) + np.log(atf)
    
    if plot: plt.plot(t[:-1], ln_a) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel("Ln Scale Factor") and plt.grid()
    
    if log: return ln_a
    return np.exp(ln_a)



def NE(ndqv, units='default', plot = False):
    
    ln_a = findScaleFactor2(ndqv, units=units, log=True)
    nE = ln_a - ln_a[0]
    if plot: plt.plot(getTimes(ndqv)[:-1], nE) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel("Number of e-Folds") and plt.grid()
    
    return nE


def aH2(ndqv, units='default', plot=False):
    H, a = getH2(ndqv, units=units)[:-1], findScaleFactor2(ndqv, units=units)
    aH = a*H
    if plot: plt.semilogy(getTimes(ndqv)[:-1], aH) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel("aH")
    return aH



def getEpsilon2(ndqv, units='mmc', plot=False):
    
    t, dphi, H = getTimes(ndqv), getPhi2(ndqv, i=1)[:-1], getH2(ndqv, units=units)[:-1]
    epsilon = 1/(2*H**2)*np.linalg.norm(dphi, axis=1)**2
    if plot: plt.semilogy(t[:-1], epsilon) and plt.xlabel(r"Time [$t_{Pl}$]") and plt.ylabel(r"$\epsilon$") and plt.grid()
    
    return epsilon



def couplingTensor2(ndqv, units ='mmc'): 
    
    phiStuff, cVals, dimPhi = ndqv
    t, H, a, dphi, epsilon = getTimes(ndqv), getH2(ndqv, units=units)[:-1], findScaleFactor2(ndqv, units=units), getPhi2(ndqv, i=1)[:-1], getEpsilon2(ndqv, units=units)
    coupler = np.zeros((dimPhi, dimPhi, len(t)-1))
    
    for i in range(dimPhi):
        for j in range(dimPhi):
            term1 = getpdpdV2(ndqv, i, j)[:-1]/H**2
            term2 = a/H**2*dphi[:,i]*getpdV2(ndqv, j)[:-1]
            term3 = a/H**2*dphi[:,j]*getpdV2(ndqv, i)[:-1]
            term4 = a**2*(3 - epsilon)*dphi[:,i]*dphi[:,j]
            
            coupler[i,j] = term1 + term2 + term3 + term4
            
    return coupler



def scalarPower2(ndqv, points = 100, kVals = [], units = 'mmc', C = (0,1), eFoldsB4Crossing = 3):
    tVals, dimPhi = getTimes(ndqv)[:-1], ndqv[2]
    a = sp.interpolate.interp1d(tVals, findScaleFactor2(ndqv, units=units))
    H = sp.interpolate.InterpolatedUnivariateSpline(tVals, getH2(ndqv, units=units)[:-1])
    dH = H.derivative()
    epsilon = sp.interpolate.interp1d(tVals, getEpsilon2(ndqv, units=units))
    dphi = getPhi2(ndqv, i=1, plot=False)[:-1]
    dphiFuncs = np.empty((dimPhi), dtype=object)
    for i in range(dimPhi): dphiFuncs[i] = sp.interpolate.interp1d(tVals, dphi[:,i])
    dphi = dphiFuncs
    coupler = couplingTensor2(ndqv, units=units)
    
    newCoupler = np.empty((dimPhi, dimPhi), dtype=object)
    for i in range(dimPhi):
        for j in range(dimPhi):
            func = sp.interpolate.interp1d(tVals, coupler[i,j])
            newCoupler[i,j] = func        
    coupler = newCoupler

    if len(kVals)==0:
        i, tMin, tMax = 0, 0, tVals[np.argmax(a(tVals)*H(tVals))]
        while np.log(a(tVals[i])/a(tVals[0])) < eFoldsB4Crossing: i += 1
        tMin = tVals[i]
        evenTimes = np.linspace(tMin, tMax, points)
        kVals = a(evenTimes)*H(evenTimes) ## What are the correct range of kVals??    
    else: points = len(kVals)
    
    
    
    ## Evaluates the coupling tensor at a particular value in time (as it is a matrix of interpolating functions).
    def callCoupler(t):
        thisCoupler = np.zeros((dimPhi,dimPhi))
        for i in range(dimPhi):
            for j in range(dimPhi):
                thisCoupler[i,j] = coupler[i,j](t)
        return thisCoupler
    
    
    
    ## Differential equation used in solve_ivp.
    def f(t, psiFlat):
        # print(f"Eval: t = {t}")
        psiStuff = np.reshape(psiFlat, (2, dimPhi, dimPhi))
        psi, dpsi = psiStuff[0], psiStuff[1]
        
        term1 = dH(t)/H(t)*dpsi
        term2 = H(t)*(epsilon(t) - 1)*dpsi
        term3 = H(t)**2*(2 - epsilon(t) - k**2/(a(t)**2*H(t)**2))*psi
        term4 = -1*H(t)**2*(callCoupler(t) @ psi)
        ddpsi = term1 + term2 + term3 + term4
        return np.append(dpsi.flatten(), ddpsi.flatten())
    
    
    
    ## Given a wavenumber (k), solve differential equation (3.14) in MultiModeCode paper.
    ## Outputs N^2 psi functions of time, N^2 "field-space" functions of time (3.19), and the "adiabatic curvature" function of time.
    def scalarPowerPoint(k, crossingTime):
        print("Setting up time interval for solution to differential equation.")
        tStart, tEnd = crossingTime, tMax
        testStartTimes = np.linspace(tVals[0], crossingTime, 100)
        testScaleVals = a(testStartTimes)
        tStart = testStartTimes[np.argmin(np.abs((np.log(testScaleVals/a(crossingTime)) + eFoldsB4Crossing)))]
               
        print(f"Solving differential equation from {tStart} to {tEnd}.")
        psiStart = np.identity(dimPhi)*1/np.sqrt(2*k)*(C[0] + C[1])
        dpsiStart = np.identity(dimPhi)*1j/(a(0)**2*H(0))*(C[0] - C[1])*np.sqrt(k/2)
        psiStuff = sp.integrate.solve_ivp(f, (tStart, tEnd), np.append(psiStart.flatten(), dpsiStart.flatten()) ) 
        
        ## Use psi to compute various power spectra.
        print("Computing power spectra.")
        psi = np.reshape(psiStuff.y, (2, dimPhi, dimPhi, len(psiStuff.t)))[0]
        aDiscrete = a(psiStuff.t)
        fieldSpace = np.empty((dimPhi, dimPhi, len(psiStuff.t)), dtype=complex)
        fieldSpaceFuncs = np.empty((dimPhi,dimPhi), dtype=object)
        for i in range(len(psiStuff.t)):
            fieldSpace[:,:,i] = k**3/(2*np.pi**2*aDiscrete[i]**2)*psi[:,:,i] @ np.matrix(psi[:,:,i], dtype=complex).H
        for i in range(dimPhi):
            for j in range(dimPhi):
                fieldSpaceFuncs[i,j] = sp.interpolate.interp1d(psiStuff.t, fieldSpace[i,j,:])
    
        dphiAdjusted = np.empty((len(psiStuff.t), dimPhi))
        for i in range(dimPhi): dphiAdjusted[:,i] = dphi[i](psiStuff.t)
        omega = normalize(dphiAdjusted, axis=1)
        epsilonDiscrete = epsilon(psiStuff.t)
        adiabaticCurv = np.empty((len(psiStuff.t)), dtype=complex)
        for i in range(len(psiStuff.t)):
            adiabaticCurv[i] = 1/(2*epsilonDiscrete[i])*omega[i] @ fieldSpace[:,:,i] @ np.transpose(omega[i]) if epsilonDiscrete[i]!=0 else np.nan
        adiabaticCurv = sp.interpolate.interp1d(psiStuff.t, adiabaticCurv)
        
        return psiStuff, fieldSpaceFuncs, adiabaticCurv, np.array([tStart, tEnd])
    
    
    
    sols, fieldSpaces, ACs, intervals =  np.empty(points, dtype=object), np.empty((points, dimPhi, dimPhi), dtype=object), np.empty(points, dtype=object), np.empty((points, 2))
    for i in range(len(kVals)):
        k, crossingTime = kVals[i], evenTimes[i]
        print(f"Computing all the data for k = {k} (wavenumber {i+1}/{len(kVals)}).")
        sol, fieldSpace, AC, interval = scalarPowerPoint(k, crossingTime)
        sols[i], fieldSpaces[i,:,:], ACs[i], intervals[i] = sol, fieldSpace, AC, interval
    
    return kVals, sols, fieldSpaces, ACs, intervals



def plotPhiByNE(ndqv, i=0, units='mmc'):
    
    dimPhi = ndqv[2]
    ln_a = findScaleFactor2(ndqv, units=units, log=True)
    ln_a -= ln_a[0]
    
    H = getH2(ndqv, units=units)[:-1]
    phi = getPhi2(ndqv, i=i)[:-1]
    newphi = np.zeros(np.shape(phi))
    if i==1:
        for t in range(len(phi)): 
            newphi[t] = phi[t]/H[t]
        phi = newphi
    
    [plt.plot(ln_a, phi[:,dim]) for dim in range(dimPhi)]
    plt.xlabel("Number of e-Folds") and plt.title("My Code Output")
    if i==0: plt.ylabel("Inflaton Field Components")
    if i==1: plt.ylabel(r"Inflaton Field Component Velocities ($\frac{d\phi}{dN_e}$)")
    
    return



def normalize(a, axis=1):
    b = []
    if len(a)>0:
        b = np.array( [a[0]/np.linalg.norm(a[0])] )
        b = np.array([a[i]/np.linalg.norm(a[i]) for i in np.arange(1,len(a))])
    return b

###############################################################################
########################  TEST MULTIMODECODE OUTPUT  ##########################
###############################################################################


def plotPhi(df, dimPhi, i=0, NETime = True):
    
    for dim in np.arange(1,dimPhi+1):
        stringNum = f"{dim}"
        while len(stringNum) < 4:
            stringNum = "0" + stringNum
        if i==0: plt.plot(df['N'], df[f'phi{stringNum}']) and plt.xlabel("Number of e-Folds") and plt.ylabel("Inflaton Components")
        if i==1: 
            if NETime: plt.plot(df['N'], df[f'dphi{stringNum}']) and plt.xlabel("Number of e-Folds") and plt.ylabel(r"Inflaton Component Velocities $d\phi/dN_e$")
            else: plt.plot(df['N'], df[f'dphi{stringNum}']*df['H']) and plt.xlabel("Number of e-Folds") and plt.ylabel(r"Inflaton Component Velocities $d\phi/dt$")
        plt.title("MultiModeCode Output")
    return


###############################################################################
##########################  FIGURES FOR THE PAPER  ############################
###############################################################################

## ndqvs: Input two ndqvs (suggested: one with dimPhi=2, one with dimPhi=10) as tuple
## tSettles: Input times for each ndqv at which to zoom in and see the oscillation
def plotForPaper(ndqvs, fontsize=15):
    
    fig, ax = plt.subplots(nrows=len(ndqvs), ncols=4, figsize=(30,15))
    
    for i in range(len(ndqvs)):
        ndqv = ndqvs[i]
        t = ndqv[0].t
        N = ndqv[2]
        phi = getPhi2(ndqv, i=0)
        dphi = getPhi2(ndqv, i=1)
        nE = NE(ndqv)
        V = getV2(ndqv)
        
        [ax[i,0].plot(t, phi[:,j]) for j in range(N)]
        ax[i,0].set_xlabel(r"$t\ [t_{Pl}]$", fontsize=fontsize)
        ax[i,0].set_ylabel(r"$\phi^{(\alpha)}\ [m_{Pl}]$", fontsize=fontsize)
        
        [ax[i,1].plot(t, dphi[:,j]) for j in range(N)]
        ax[i,1].set_xlabel(r"$t\ [t_{Pl}]$", fontsize=fontsize)
        ax[i,1].set_ylabel(r"$d{\phi}^{(\alpha)}/dt\ [m_{Pl}/t_{Pl}]$", fontsize=fontsize)
        
        [ax[i,2].plot(nE, phi[:-1,j]) for j in range(N)]
        ax[i,2].set_xlabel(r"$N_e$", fontsize=fontsize)
        ax[i,2].set_ylabel(r"$\phi^{(\alpha)}\ [m_{Pl}]$", fontsize=fontsize)
        
        ax[i,3].plot(t[:-1], nE, c='r')
        ax[i,3].set_xlabel(r"$t\ [t_{Pl}]$", fontsize=fontsize)
        ax[i,3].set_ylabel(r"$N_e$", fontsize=fontsize)
        
        [ax[i,j].grid() for j in range(4)]
        
    return



def topoPlotForPaper(ndqv):
    
    fig, ax = plt.subplots()
    phi = getPhi2(ndqv, i=0)
    V = getV2(ndqv)
    ax.scatter(phi[:,0], phi[:,0], marker='+', c=V)
    ax.set_xlabel(r"$\phi^{(0)}$") and ax.set_ylabel(r"$\phi^{(1)}$")
    ax.grid()
    
    return



## The 'H' in the differential equation needs to be changed to mmc units as well.





