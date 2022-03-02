## This is the document that will contain the methods suitable for execution in Quark.
## The purpose of this document is to hone in on the relevant tools and methods,
## omit the unnecessary content, and satisfy that craving for elegance.

###############################################################################
## ------------------------------IMPORTATIONS------------------------------- ##
###############################################################################


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import pandas as pd
import time
import itertools
import os
import shutil
import scipy.linalg
import scipy.signal
import scipy.interpolate
import scipy.integrate
import scipy.stats
import concurrent.futures
from scipy.interpolate import CubicSpline as CS
from datetime import date



###############################################################################
## ----------------------------GLOBAL CONSTANTS----------------------------- ##
###############################################################################



G = 6.674e-11
hbar = 1.055e-34
c = 2.998e8
useMmcUnits = True
plc = 1/3 if useMmcUnits else 8*np.pi/3
plancklabel = r"$(8 \pi G)^{-1} = 1$" if useMmcUnits else r"$G = 1$"
nEBack4Pivot = 55
kPivot = 5.25e-60 ## = 0.002 Mpc^-1
kStar = 1.3125e-58 ## 0.05 Mpc^-1
noise = 1e-12
maxCovLen = 1800
timelabel = r"Time [$t_{Pl}$]"



###############################################################################
## ------------------------COVARIANCE MATRIX AND XTX------------------------ ##
###############################################################################



def initiateVCOV(s, phiPoints, infoVec):
    
    phiPoints = np.transpose(phiPoints)
    l = len(infoVec[0])
    vcov = np.zeros((l,l))
    infoVec = infoVec.astype(int)
    
    vIndices, sIndices = np.where(infoVec[1]==-1)[0], np.where(infoVec[2]!=-1)[0]
    gIndices = np.intersect1d(np.where(infoVec[1]!=-1)[0], np.where(infoVec[2]==-1)[0])
    
    if len(vIndices) != 0:
        
        ## VV
        vLocations = np.array(list(itertools.combinations_with_replacement(vIndices, 2)))
        diffVecs = phiPoints[infoVec[0, vLocations[:,0]]] - phiPoints[infoVec[0, vLocations[:,1]]]
        
        vcov[vLocations[:,0], vLocations[:,1]] = np.exp(-1*np.linalg.norm(diffVecs, axis=1)**2/(2*s**2))
        vcov[vLocations[:,1], vLocations[:,0]] = vcov[vLocations[:,0], vLocations[:,1]]
    
    if len(gIndices) != 0:
        
        ## VG and GV
        gLocations = np.array(list(itertools.product(vIndices, gIndices)))
        gvLocations = gLocations[gLocations[:,0] > gLocations[:,1]]
        vgLocations = gLocations[gLocations[:,0] < gLocations[:,1]]
        
        gvt1s, gvt2s = infoVec[0, gvLocations[:,0]], infoVec[0, gvLocations[:,1]]
        gvDiffVecs = phiPoints[gvt1s] - phiPoints[gvt2s]
        gvBase = np.exp(-1*np.linalg.norm(gvDiffVecs, axis=1)**2/(2*s**2))
        
        vgt1s, vgt2s = infoVec[0, vgLocations[:,0]], infoVec[0, vgLocations[:,1]]
        vgDiffVecs = phiPoints[infoVec[0, vgLocations[:,0]]] - phiPoints[infoVec[0, vgLocations[:,1]]]
        vgBase = np.exp(-1*np.linalg.norm(vgDiffVecs, axis=1)**2/(2*s**2))
        
        alphas, betas = infoVec[1, gvLocations[:,1]], infoVec[1, vgLocations[:,1]]
        
        vcov[gvLocations[:,0], gvLocations[:,1]] = 1/(s**2)*(phiPoints[gvt1s, alphas] - phiPoints[gvt2s, alphas])*gvBase ## order could be flipped
        vcov[vgLocations[:,0], vgLocations[:,1]] = 1/(s**2)*(phiPoints[vgt1s, betas] - phiPoints[vgt2s, betas])*vgBase
        vcov[gLocations[:,1], gLocations[:,0]] = vcov[gLocations[:,0], gLocations[:,1]]
        
        ## GG (same and different)
        ggLocations = np.array(list(itertools.combinations_with_replacement(gIndices, 2)))
        ggSameLocations = ggLocations[infoVec[1, ggLocations[:,0]] == infoVec[1, ggLocations[:,1]]]
        ggDiffLocations = ggLocations[infoVec[1, ggLocations[:,0]] != infoVec[1, ggLocations[:,1]]]
        
        ggSamet1s, ggSamet2s = infoVec[0, ggSameLocations[:,0]], infoVec[0, ggSameLocations[:,1]]
        ggSameDiffVecs = phiPoints[ggSamet1s] - phiPoints[ggSamet2s]
        ggSameBase = np.exp(-1*np.linalg.norm(ggSameDiffVecs, axis=1)**2/(2*s**2))
        
        ggDifft1s, ggDifft2s = infoVec[0, ggDiffLocations[:,0]], infoVec[0, ggDiffLocations[:,1]]
        ggDiffDiffVecs = phiPoints[ggDifft1s] - phiPoints[ggDifft2s]
        ggDiffBase = np.exp(-1*np.linalg.norm(ggDiffDiffVecs, axis=1)**2/(2*s**2))
        
        alphas = infoVec[1, ggSameLocations[:,0]]
        betas, gammas = infoVec[1, ggDiffLocations[:,0]], infoVec[1, ggDiffLocations[:,1]]
        
        ggSameMultiplier = -1/(s**4)*(phiPoints[ggSamet1s, alphas] - phiPoints[ggSamet2s, alphas])**2 + 1/(s**2)
        ggDiffMultiplier = -1/(s**4)*(phiPoints[ggDifft1s, betas] - phiPoints[ggDifft2s, betas])*(phiPoints[ggDifft1s, gammas] - phiPoints[ggDifft2s, gammas])
        vcov[ggSameLocations[:,0], ggSameLocations[:,1]] = ggSameMultiplier*ggSameBase
        vcov[ggDiffLocations[:,0], ggDiffLocations[:,1]] = ggDiffMultiplier*ggDiffBase
        vcov[ggLocations[:,1], ggLocations[:,0]] = vcov[ggLocations[:,0], ggLocations[:,1]]
    
    if len(sIndices) != 0:
        
        ## VS (= SV) (same and different)
        vsLocations = np.array(list(itertools.product(vIndices, sIndices)))
        vsSameLocations = vsLocations[infoVec[1, vsLocations[:,1]] == infoVec[2, vsLocations[:,1]]]
        vsDiffLocations = vsLocations[infoVec[1, vsLocations[:,1]] != infoVec[2, vsLocations[:,1]]]
        
        vsSamet1s, vsSamet2s = infoVec[0, vsSameLocations[:,0]], infoVec[0, vsSameLocations[:,1]]
        vsSameDiffVecs = phiPoints[vsSamet1s] - phiPoints[vsSamet2s]
        vsSameBase = np.exp(-1*np.linalg.norm(vsSameDiffVecs, axis=1)**2/(2*s**2))
        
        vsDifft1s, vsDifft2s = infoVec[0, vsDiffLocations[:,0]], infoVec[0, vsDiffLocations[:,1]]
        vsDiffDiffVecs = phiPoints[vsDifft1s] - phiPoints[vsDifft2s]
        vsDiffBase = np.exp(-1*np.linalg.norm(vsDiffDiffVecs, axis=1)**2/(2*s**2))
        
        alphas = infoVec[1, vsSameLocations[:,1]]
        vsSameMultiplier = -1/s**2 + (phiPoints[vsSamet2s, alphas] - phiPoints[vsSamet1s, alphas])**2/s**4
        
        alphas, betas = infoVec[1, vsDiffLocations[:,1]], infoVec[2, vsDiffLocations[:,1]]
        vsDiffMultiplier = (phiPoints[vsDifft2s, alphas] - phiPoints[vsDifft1s, alphas])*(phiPoints[vsDifft2s, betas] - phiPoints[vsDifft1s, betas])/s**4
        
        vcov[vsSameLocations[:,0], vsSameLocations[:,1]] = vsSameMultiplier*vsSameBase
        vcov[vsDiffLocations[:,0], vsDiffLocations[:,1]] = vsDiffMultiplier*vsDiffBase
        vcov[vsLocations[:,1], vsLocations[:,0]] = vcov[vsLocations[:,0], vsLocations[:,1]]
        
        ## GS and SG (different, one pair, all same)
        gsLocations = np.array(list(itertools.product(gIndices, sIndices)))
        alphas, betas, gammas, deltas = infoVec[1, gsLocations[:,0]], infoVec[2, gsLocations[:,0]], infoVec[1, gsLocations[:,1]], infoVec[2, gsLocations[:,1]]
        gsSameLocations, gs2OAKLocations, gsDiffLocations = np.array([[]]), np.array([[]]), np.array([[]])
        for i in range(len(gsLocations)):
            sames = 4 - len(set([alphas[i], betas[i], gammas[i], deltas[i]]))
            loc = np.array([gsLocations[i]])
            if sames == 2: gsSameLocations = loc if len(gsSameLocations[0])==0 else np.concatenate((gsSameLocations, loc))
            elif sames == 1: gs2OAKLocations = loc if len(gs2OAKLocations[0])==0 else np.concatenate((gs2OAKLocations, loc))
            elif sames == 0: gsDiffLocations = loc if len(gsDiffLocations[0])==0 else np.concatenate((gsDiffLocations, loc))
        
        if len(gsSameLocations[0]) != 0:
            gsSamet1s, gsSamet2s = infoVec[0, gsSameLocations[:,0]], infoVec[0, gsSameLocations[:,1]]
            gsSameDiffVecs = phiPoints[gsSamet1s] - phiPoints[gsSamet2s]
            gsSameBase = np.exp(-1*np.linalg.norm(gsSameDiffVecs, axis=1)**2/(2*s**2))
            
            alphas, betas, gammas, deltas = infoVec[1, gsSameLocations[:,0]], infoVec[2, gsSameLocations[:,0]], infoVec[1, gsSameLocations[:,1]], infoVec[2, gsSameLocations[:,1]]
            gsSameMultiplier = (phiPoints[gsSamet2s, alphas] - phiPoints[gsSamet1s, alphas])/s**2*(3/s**2 - (phiPoints[gsSamet2s, alphas] - phiPoints[gsSamet1s, alphas])**2/s**4)
            gsSameMultiplier[deltas == alphas] = -1*gsSameMultiplier[deltas == alphas]
            vcov[gsSameLocations[:,0], gsSameLocations[:,1]] = gsSameBase*gsSameMultiplier
        
        if len(gs2OAKLocations[0]) != 0:
            gs2OAKt1s, gs2OAKt2s = infoVec[0, gs2OAKLocations[:,0]], infoVec[0, gs2OAKLocations[:,1]]
            gs2OAKDiffVecs = phiPoints[gs2OAKt1s] - phiPoints[gs2OAKt2s]
            gs2OAKBase = np.exp(-1*np.linalg.norm(gs2OAKDiffVecs, axis=1)**2/(2*s**2))
            
            alphas, betas, gammas, deltas = infoVec[1, gs2OAKLocations[:,0]], infoVec[2, gs2OAKLocations[:,0]], infoVec[1, gs2OAKLocations[:,1]], infoVec[2, gs2OAKLocations[:,1]]
            gs2OAKMultiplier = []
            for i in range(len(alphas)):
                indices = [alphas[i], betas[i], gammas[i], deltas[i]]
                indices = [j for j in indices if j!=-1]
                same = sp.stats.mode(indices)[0][0]
                diff = [j for j in indices if j!=same]
                mult = (phiPoints[gs2OAKt2s[i], diff] - phiPoints[gs2OAKt1s[i], diff])/s**2*(1/s**2 - (phiPoints[gs2OAKt2s[i], same] - phiPoints[gs2OAKt1s[i], same])**2/s**4)
                if betas[i]==-1: mult *= -1
                gs2OAKMultiplier = np.append(gs2OAKMultiplier, mult)
            vcov[gs2OAKLocations[:,0], gs2OAKLocations[:,1]] = gs2OAKBase*gs2OAKMultiplier
            
        if len(gsDiffLocations[0]) != 0:
            gsDifft1s, gsDifft2s = infoVec[0, gsDiffLocations[:,0]], infoVec[0, gsDiffLocations[:,1]]
            gsDiffDiffVecs = phiPoints[gsDifft1s] - phiPoints[gsDifft2s]
            gsDiffBase = np.exp(-1*np.linalg.norm(gsDiffDiffVecs, axis=1)**2/(2*s**2))
                
            alphas, betas, gammas, deltas = infoVec[1, gsDiffLocations[:,0]], infoVec[2, gsDiffLocations[:,0]], infoVec[1, gsDiffLocations[:,1]], infoVec[2, gsDiffLocations[:,1]]
            betas_deltas = list(np.transpose([betas, deltas]).flatten())
            betas_deltas = np.array([bd for bd in betas_deltas if bd != -1])
            gsDiffMultiplier = (phiPoints[gsDifft2s, alphas] - phiPoints[gsDifft1s, alphas])*(phiPoints[gsDifft2s, gammas] - phiPoints[gsDifft1s, gammas])*(phiPoints[gsDifft2s, betas_deltas] - phiPoints[gsDifft1s, betas_deltas])/s**6
            vcov[gsDiffLocations[:,0], gsDiffLocations[:,1]] = gsDiffBase*gsDiffMultiplier    
        
        vcov[gsLocations[:,1], gsLocations[:,0]] = vcov[gsLocations[:,0], gsLocations[:,1]]
        
        ## SS (different, one pair, two pair, three of a kind, all same)
        ssLocations = np.array(list(itertools.combinations_with_replacement(sIndices, 2)))
        alphas, betas, gammas, deltas = infoVec[1, ssLocations[:,0]], infoVec[2, ssLocations[:,0]], infoVec[1, ssLocations[:,1]], infoVec[2, ssLocations[:,1]]
        allindices = np.array([alphas, betas, gammas, deltas])
        sames = np.array([4 - len(set(allindices[:,i])) for i in range(len(alphas))])
        
        ssSameLocations = ssLocations[sames == 3]
        ss3OAKLocations, ss2ParLocations = [], []
        for i in np.where(sames == 2)[0]:
            if np.mean(allindices[:,i]) != (max(allindices[:,i]) + min(allindices[:,i]))/2: 
                ss3OAKLocations = np.array([ssLocations[i]]) if len(ss3OAKLocations)==0 else np.concatenate((ss3OAKLocations, [ssLocations[i]]))
            else: 
                ss2ParLocations = np.array([ssLocations[i]]) if len(ss2ParLocations)==0 else np.concatenate((ss2ParLocations, [ssLocations[i]]))
        ss2OAKLocations = ssLocations[sames == 1]
        ssDiffLocations = ssLocations[sames == 0]
        
        if len(ssSameLocations) != 0: 
            ssSamet1s, ssSamet2s = infoVec[0, ssSameLocations[:,0]], infoVec[0, ssSameLocations[:,1]]
            ssSameDiffVecs = phiPoints[ssSamet1s] - phiPoints[ssSamet2s]
            ssSameBase = np.exp(-1*np.linalg.norm(ssSameDiffVecs, axis=1)**2/(2*s**2))
            
            alphas, betas, gammas, deltas = infoVec[1, ssSameLocations[:,0]], infoVec[2, ssSameLocations[:,0]], infoVec[1, ssSameLocations[:,1]], infoVec[2, ssSameLocations[:,1]]
            ssSameMultiplier = (phiPoints[ssSamet2s, alphas] - phiPoints[ssSamet1s, alphas])**4/s**8 - 6/s**2*(phiPoints[ssSamet2s, alphas] - phiPoints[ssSamet1s, alphas])**2/s**4 + 3/s**4
            vcov[ssSameLocations[:,0], ssSameLocations[:,1]] = ssSameBase*ssSameMultiplier
            
        if len(ss3OAKLocations) != 0: 
            ss3OAKt1s, ss3OAKt2s = infoVec[0, ss3OAKLocations[:,0]], infoVec[0, ss3OAKLocations[:,1]]
            ss3OAKDiffVecs = phiPoints[ss3OAKt1s] - phiPoints[ss3OAKt2s]
            ss3OAKBase = np.exp(-1*np.linalg.norm(ss3OAKDiffVecs, axis=1)**2/(2*s**2))
            
            alphas, betas, gammas, deltas = infoVec[1, ss3OAKLocations[:,0]], infoVec[2, ss3OAKLocations[:,0]], infoVec[1, ss3OAKLocations[:,1]], infoVec[2, ss3OAKLocations[:,1]]
            ss3OAKMultiplier = []
            for i in range(len(alphas)):
                indices = [alphas[i], betas[i], gammas[i], deltas[i]]
                same = sp.stats.mode(indices)[0][0]
                diff = np.setdiff1d(indices, [same], assume_unique=True)[0]
                mult = (phiPoints[ss3OAKt2s[i], same] - phiPoints[ss3OAKt1s[i], same])/s**2*(phiPoints[ss3OAKt1s[i], diff] - phiPoints[ss3OAKt2s[i], diff])/s**2*(3/s**2 - (phiPoints[ss3OAKt2s[i], same] - phiPoints[ss3OAKt1s[i], same])**2/s**4)
                ss3OAKMultiplier = np.append(ss3OAKMultiplier, mult)
            vcov[ss3OAKLocations[:,0], ss3OAKLocations[:,1]] = ss3OAKBase*ss3OAKMultiplier
            
        if len(ss2ParLocations) != 0: 
            ss2Part1s, ss2Part2s = infoVec[0, ss2ParLocations[:,0]], infoVec[0, ss2ParLocations[:,1]]
            ss2ParDiffVecs = phiPoints[ss2Part1s] - phiPoints[ss2Part2s]
            ss2ParBase = np.exp(-1*np.linalg.norm(ss2ParDiffVecs, axis=1)**2/(2*s**2))
            
            alphas, betas, gammas, deltas = infoVec[1, ss2ParLocations[:,0]], infoVec[2, ss2ParLocations[:,0]], infoVec[1, ss2ParLocations[:,1]], infoVec[2, ss2ParLocations[:,1]]
            ss2ParMultiplier = []
            for i in range(len(alphas)):
                indices = [alphas[i], betas[i], gammas[i], deltas[i]]
                first, second = list(set(indices))[0], list(set(indices))[1]
                
                mult = (1/s**2 - (phiPoints[ss2Part2s[i], first] - phiPoints[ss2Part1s[i], first])**2/s**4)*(1/s**2 - (phiPoints[ss2Part2s[i], second] - phiPoints[ss2Part1s[i], second])**2/s**4)
                ss2ParMultiplier = np.append(ss2ParMultiplier, mult)
            vcov[ss2ParLocations[:,0], ss2ParLocations[:,1]] = ss2ParBase*ss2ParMultiplier
            
        if len(ss2OAKLocations) != 0:
            ss2OAKt1s, ss2OAKt2s = infoVec[0, ss2OAKLocations[:,0]], infoVec[0, ss2OAKLocations[:,1]]
            ss2OAKDiffVecs = phiPoints[ss2OAKt1s] - phiPoints[ss2OAKt2s]
            ss2OAKBase = np.exp(-1*np.linalg.norm(ss2OAKDiffVecs, axis=1)**2/(2*s**2))
            
            alphas, betas, gammas, deltas = infoVec[1, ss2OAKLocations[:,0]], infoVec[2, ss2OAKLocations[:,0]], infoVec[1, ss2OAKLocations[:,1]], infoVec[2, ss2OAKLocations[:,1]]
            ss2OAKMultiplier = []
            for i in range(len(alphas)):
                indices = [alphas[i], betas[i], gammas[i], deltas[i]]
                same = sp.stats.mode(indices)[0][0]
                diff1, diff2 = list(np.setdiff1d(indices, [same], assume_unique=True))
                mult = ss2OAKDiffVecs[i, diff1]/s**2*ss2OAKDiffVecs[i, diff2]/s**2*(-1/s**2 + ss2OAKDiffVecs[i, same]**2/s**4)
                ss2OAKMultiplier = np.append(ss2OAKMultiplier, mult)
            vcov[ss2OAKLocations[:,0], ss2OAKLocations[:,1]] = ss2OAKBase*ss2OAKMultiplier
            
        if len(ssDiffLocations) != 0: 
            ssDifft1s, ssDifft2s = infoVec[0, ssDiffLocations[:,0]], infoVec[0, ssDiffLocations[:,1]]
            ssDiffDiffVecs = phiPoints[ssDifft1s] - phiPoints[ssDifft2s]
            ssDiffBase = np.exp(-1*np.linalg.norm(ssDiffDiffVecs, axis=1)**2/(2*s**2))
            
            alphas, betas, gammas, deltas = infoVec[1, ssDiffLocations[:,0]], infoVec[2, ssDiffLocations[:,0]], infoVec[1, ssDiffLocations[:,1]], infoVec[2, ssDiffLocations[:,1]]
            ssDiffMultiplier = (phiPoints[ssDifft2s, alphas] - phiPoints[ssDifft1s, alphas])*(phiPoints[ssDifft2s, betas] - phiPoints[ssDifft1s, betas])*(phiPoints[ssDifft2s, gammas] - phiPoints[ssDifft1s, gammas])*(phiPoints[ssDifft2s, deltas] - phiPoints[ssDifft1s, deltas])/s**8
            vcov[ssDiffLocations[:,0], ssDiffLocations[:,1]] = ssDiffBase*ssDiffMultiplier
        
        vcov[ssLocations[:,1], ssLocations[:,0]] = vcov[ssLocations[:,0], ssLocations[:,1]]
        
    return vcov



def updateBeefCOV(s, phiPoints, infoVec, beefCov, bar):
    
    phiPoints = np.transpose(phiPoints)
    infoVec = infoVec.astype(int)
    
    vIndices, sIndices = np.where(infoVec[1]==-1)[0], np.where(infoVec[2]!=-1)[0]
    gIndices = np.intersect1d(np.where(infoVec[1]!=-1)[0], np.where(infoVec[2]==-1)[0])
    
    ovIndices, ogIndices, osIndices = vIndices[vIndices < bar], gIndices[gIndices < bar], sIndices[sIndices < bar]
    nvIndices, ngIndices, nsIndices = vIndices[vIndices >= bar], gIndices[gIndices >= bar], sIndices[sIndices >= bar]
    
    if len(vIndices) > 0:
        
        ## VV
        vvLocations = np.array(list(itertools.product(vIndices, nvIndices)))
        
        if len(vvLocations) > 0:
            diffVecs = phiPoints[infoVec[0, vvLocations[:,0]]] - phiPoints[infoVec[0, vvLocations[:,1]]]    
            beefCov[vvLocations[:,0], vvLocations[:,1]] = np.exp(-1*np.linalg.norm(diffVecs, axis=1)**2/(2*s**2))
            beefCov[vvLocations[:,1], vvLocations[:,0]] = beefCov[vvLocations[:,0], vvLocations[:,1]]
    
    if len(gIndices) > 0:
        
        ## VG and GV
        gvLocations = np.array(list(itertools.product(nvIndices, ogIndices)))
        vgLocations = np.array(list(itertools.product(ovIndices, ngIndices)))
        
        if len(gvLocations) > 0:
            gvt1s, gvt2s = infoVec[0, gvLocations[:,0]], infoVec[0, gvLocations[:,1]]
            gvDiffVecs = phiPoints[gvt1s] - phiPoints[gvt2s]
            gvBase = np.exp(-1*np.linalg.norm(gvDiffVecs, axis=1)**2/(2*s**2))
            alphas = infoVec[1, gvLocations[:,1]]
            beefCov[gvLocations[:,0], gvLocations[:,1]] = 1/(s**2)*(phiPoints[gvt1s, alphas] - phiPoints[gvt2s, alphas])*gvBase
            beefCov[gvLocations[:,1], gvLocations[:,0]] = beefCov[gvLocations[:,0], gvLocations[:,1]]
        
        if len(vgLocations) > 0:
            vgt1s, vgt2s = infoVec[0, vgLocations[:,0]], infoVec[0, vgLocations[:,1]]
            vgDiffVecs = phiPoints[infoVec[0, vgLocations[:,0]]] - phiPoints[infoVec[0, vgLocations[:,1]]]
            vgBase = np.exp(-1*np.linalg.norm(vgDiffVecs, axis=1)**2/(2*s**2))
            betas = infoVec[1, vgLocations[:,1]]
            beefCov[vgLocations[:,0], vgLocations[:,1]] = 1/(s**2)*(phiPoints[vgt1s, betas] - phiPoints[vgt2s, betas])*vgBase
            beefCov[vgLocations[:,1], vgLocations[:,0]] = beefCov[vgLocations[:,0], vgLocations[:,1]]
    
        ## GG (same and different)
        ggLocations = np.array(list(itertools.product(gIndices, ngIndices)))
        
        if len(ggLocations) > 0:
            ggSameLocations = ggLocations[infoVec[1, ggLocations[:,0]] == infoVec[1, ggLocations[:,1]]]
            ggDiffLocations = ggLocations[infoVec[1, ggLocations[:,0]] != infoVec[1, ggLocations[:,1]]]
        
            if len(ggSameLocations) > 0:
                ggSamet1s, ggSamet2s = infoVec[0, ggSameLocations[:,0]], infoVec[0, ggSameLocations[:,1]]
                ggSameDiffVecs = phiPoints[ggSamet1s] - phiPoints[ggSamet2s]
                ggSameBase = np.exp(-1*np.linalg.norm(ggSameDiffVecs, axis=1)**2/(2*s**2))
                alphas = infoVec[1, ggSameLocations[:,0]]
                ggSameMultiplier = -1/(s**4)*(phiPoints[ggSamet1s, alphas] - phiPoints[ggSamet2s, alphas])**2 + 1/(s**2)
                beefCov[ggSameLocations[:,0], ggSameLocations[:,1]] = ggSameMultiplier*ggSameBase
        
            if len(ggDiffLocations) > 0:
                ggDifft1s, ggDifft2s = infoVec[0, ggDiffLocations[:,0]], infoVec[0, ggDiffLocations[:,1]]
                ggDiffDiffVecs = phiPoints[ggDifft1s] - phiPoints[ggDifft2s]
                ggDiffBase = np.exp(-1*np.linalg.norm(ggDiffDiffVecs, axis=1)**2/(2*s**2))
                betas, gammas = infoVec[1, ggDiffLocations[:,0]], infoVec[1, ggDiffLocations[:,1]]
                ggDiffMultiplier = -1/(s**4)*(phiPoints[ggDifft1s, betas] - phiPoints[ggDifft2s, betas])*(phiPoints[ggDifft1s, gammas] - phiPoints[ggDifft2s, gammas])
                beefCov[ggDiffLocations[:,0], ggDiffLocations[:,1]] = ggDiffMultiplier*ggDiffBase
        
            beefCov[ggLocations[:,1], ggLocations[:,0]] = beefCov[ggLocations[:,0], ggLocations[:,1]]
    
    if len(sIndices) > 0:
        
        ## VS (= SV) (same and different)
        vsLocations = np.array(list(itertools.product(nvIndices, osIndices)))
        svLocations = np.array(list(itertools.product(ovIndices, nsIndices)))
        
        if len(vsLocations)==0: vsLocations = svLocations
        elif len(svLocations)==0: pass
        else: vsLocations = np.concatenate((vsLocations, svLocations), axis=0)
        
        if len(vsLocations) > 0:
            vsSameLocations = vsLocations[infoVec[1, vsLocations[:,1]] == infoVec[2, vsLocations[:,1]]]
            vsDiffLocations = vsLocations[infoVec[1, vsLocations[:,1]] != infoVec[2, vsLocations[:,1]]]
            
            if len(vsSameLocations) > 0:
                vsSamet1s, vsSamet2s = infoVec[0, vsSameLocations[:,0]], infoVec[0, vsSameLocations[:,1]]
                vsSameDiffVecs = phiPoints[vsSamet1s] - phiPoints[vsSamet2s]
                vsSameBase = np.exp(-1*np.linalg.norm(vsSameDiffVecs, axis=1)**2/(2*s**2))
                alphas = infoVec[1, vsSameLocations[:,1]]
                vsSameMultiplier = -1/s**2 + (phiPoints[vsSamet2s, alphas] - phiPoints[vsSamet1s, alphas])**2/s**4
                beefCov[vsSameLocations[:,0], vsSameLocations[:,1]] = vsSameMultiplier*vsSameBase
            
            if len(vsDiffLocations) > 0:
                vsDifft1s, vsDifft2s = infoVec[0, vsDiffLocations[:,0]], infoVec[0, vsDiffLocations[:,1]]
                vsDiffDiffVecs = phiPoints[vsDifft1s] - phiPoints[vsDifft2s]
                vsDiffBase = np.exp(-1*np.linalg.norm(vsDiffDiffVecs, axis=1)**2/(2*s**2))
                alphas, betas = infoVec[1, vsDiffLocations[:,1]], infoVec[2, vsDiffLocations[:,1]]
                vsDiffMultiplier = (phiPoints[vsDifft2s, alphas] - phiPoints[vsDifft1s, alphas])*(phiPoints[vsDifft2s, betas] - phiPoints[vsDifft1s, betas])/s**4            
                beefCov[vsDiffLocations[:,0], vsDiffLocations[:,1]] = vsDiffMultiplier*vsDiffBase
            
            beefCov[vsLocations[:,1], vsLocations[:,0]] = beefCov[vsLocations[:,0], vsLocations[:,1]]
        
        ## GS and SG (different, one pair, all same)
        gsLocations = np.array(list(itertools.product(ngIndices, osIndices)))
        sgLocations = np.array(list(itertools.product(ogIndices, nsIndices)))
        
        if len(gsLocations)==0: gsLocations = sgLocations
        elif len(sgLocations)==0: pass
        else: gsLocations = np.concatenate((gsLocations, sgLocations), axis=0)
        
        
        if len(gsLocations) > 0:
            alphas, betas, gammas, deltas = infoVec[1, gsLocations[:,0]], infoVec[2, gsLocations[:,0]], infoVec[1, gsLocations[:,1]], infoVec[2, gsLocations[:,1]]
            gsSameLocations, gs2OAKLocations, gsDiffLocations = np.array([[]]), np.array([[]]), np.array([[]])
            for i in range(len(gsLocations)):
                sames = 4 - len(set([alphas[i], betas[i], gammas[i], deltas[i]]))
                loc = np.array([gsLocations[i]])
                if sames == 2: gsSameLocations = loc if len(gsSameLocations[0])==0 else np.concatenate((gsSameLocations, loc))
                elif sames == 1: gs2OAKLocations = loc if len(gs2OAKLocations[0])==0 else np.concatenate((gs2OAKLocations, loc))
                elif sames == 0: gsDiffLocations = loc if len(gsDiffLocations[0])==0 else np.concatenate((gsDiffLocations, loc))
                
            if len(gsSameLocations[0]) != 0:
                gsSamet1s, gsSamet2s = infoVec[0, gsSameLocations[:,0]], infoVec[0, gsSameLocations[:,1]]
                gsSameDiffVecs = phiPoints[gsSamet1s] - phiPoints[gsSamet2s]
                gsSameBase = np.exp(-1*np.linalg.norm(gsSameDiffVecs, axis=1)**2/(2*s**2))
            
                alphas, betas, gammas, deltas = infoVec[1, gsSameLocations[:,0]], infoVec[2, gsSameLocations[:,0]], infoVec[1, gsSameLocations[:,1]], infoVec[2, gsSameLocations[:,1]]
                gsSameMultiplier = (phiPoints[gsSamet2s, alphas] - phiPoints[gsSamet1s, alphas])/s**2*(3/s**2 - (phiPoints[gsSamet2s, alphas] - phiPoints[gsSamet1s, alphas])**2/s**4)
                gsSameMultiplier[deltas == alphas] = -1*gsSameMultiplier[deltas == alphas]
                beefCov[gsSameLocations[:,0], gsSameLocations[:,1]] = gsSameBase*gsSameMultiplier
        
            if len(gs2OAKLocations[0]) != 0:
                gs2OAKt1s, gs2OAKt2s = infoVec[0, gs2OAKLocations[:,0]], infoVec[0, gs2OAKLocations[:,1]]
                gs2OAKDiffVecs = phiPoints[gs2OAKt1s] - phiPoints[gs2OAKt2s]
                gs2OAKBase = np.exp(-1*np.linalg.norm(gs2OAKDiffVecs, axis=1)**2/(2*s**2))
                
                alphas, betas, gammas, deltas = infoVec[1, gs2OAKLocations[:,0]], infoVec[2, gs2OAKLocations[:,0]], infoVec[1, gs2OAKLocations[:,1]], infoVec[2, gs2OAKLocations[:,1]]
                gs2OAKMultiplier = []
                for i in range(len(alphas)):
                    indices = [alphas[i], betas[i], gammas[i], deltas[i]]
                    indices = [j for j in indices if j!=-1]
                    same = sp.stats.mode(indices)[0][0]
                    diff = [j for j in indices if j!=same]
                    mult = (phiPoints[gs2OAKt2s[i], diff] - phiPoints[gs2OAKt1s[i], diff])/s**2*(1/s**2 - (phiPoints[gs2OAKt2s[i], same] - phiPoints[gs2OAKt1s[i], same])**2/s**4)
                    if betas[i]==-1: mult *= -1
                    gs2OAKMultiplier = np.append(gs2OAKMultiplier, mult)
                
                beefCov[gs2OAKLocations[:,0], gs2OAKLocations[:,1]] = gs2OAKBase*gs2OAKMultiplier
            
            if len(gsDiffLocations[0]) != 0:
                gsDifft1s, gsDifft2s = infoVec[0, gsDiffLocations[:,0]], infoVec[0, gsDiffLocations[:,1]]
                gsDiffDiffVecs = phiPoints[gsDifft1s] - phiPoints[gsDifft2s]
                gsDiffBase = np.exp(-1*np.linalg.norm(gsDiffDiffVecs, axis=1)**2/(2*s**2))
                    
                alphas, betas, gammas, deltas = infoVec[1, gsDiffLocations[:,0]], infoVec[2, gsDiffLocations[:,0]], infoVec[1, gsDiffLocations[:,1]], infoVec[2, gsDiffLocations[:,1]]
                betas_deltas = list(np.transpose([betas, deltas]).flatten())
                betas_deltas = np.array([bd for bd in betas_deltas if bd != -1])
                gsDiffMultiplier = (phiPoints[gsDifft2s, alphas] - phiPoints[gsDifft1s, alphas])*(phiPoints[gsDifft2s, gammas] - phiPoints[gsDifft1s, gammas])*(phiPoints[gsDifft2s, betas_deltas] - phiPoints[gsDifft1s, betas_deltas])/s**6
                beefCov[gsDiffLocations[:,0], gsDiffLocations[:,1]] = gsDiffBase*gsDiffMultiplier    
        
            beefCov[gsLocations[:,1], gsLocations[:,0]] = beefCov[gsLocations[:,0], gsLocations[:,1]]
        
        ## SS (different, one pair, two pair, three of a kind, all same)
        ssLocations = np.array(list(itertools.product(sIndices, nsIndices)))
        
        if len(ssLocations) > 0:
            alphas, betas, gammas, deltas = infoVec[1, ssLocations[:,0]], infoVec[2, ssLocations[:,0]], infoVec[1, ssLocations[:,1]], infoVec[2, ssLocations[:,1]]
            allindices = np.array([alphas, betas, gammas, deltas])
            sames = np.array([4 - len(set(allindices[:,i])) for i in range(len(alphas))])
            
            ssSameLocations = ssLocations[sames == 3]
            ss3OAKLocations, ss2ParLocations = [], []
            for i in np.where(sames == 2)[0]:
                if np.mean(allindices[:,i]) != (max(allindices[:,i]) + min(allindices[:,i]))/2: 
                    ss3OAKLocations = np.array([ssLocations[i]]) if len(ss3OAKLocations)==0 else np.concatenate((ss3OAKLocations, [ssLocations[i]]))
                else: 
                    ss2ParLocations = np.array([ssLocations[i]]) if len(ss2ParLocations)==0 else np.concatenate((ss2ParLocations, [ssLocations[i]]))
            ss2OAKLocations = ssLocations[sames == 1]
            ssDiffLocations = ssLocations[sames == 0]
            #print(ssSameLocations); print(ss3OAKLocations); print(ss2ParLocations); print(ss2OAKLocations); print(ssDiffLocations)
            if len(ssSameLocations) != 0: 
                ssSamet1s, ssSamet2s = infoVec[0, ssSameLocations[:,0]], infoVec[0, ssSameLocations[:,1]]
                ssSameDiffVecs = phiPoints[ssSamet1s] - phiPoints[ssSamet2s]
                ssSameBase = np.exp(-1*np.linalg.norm(ssSameDiffVecs, axis=1)**2/(2*s**2))
                
                alphas, betas, gammas, deltas = infoVec[1, ssSameLocations[:,0]], infoVec[2, ssSameLocations[:,0]], infoVec[1, ssSameLocations[:,1]], infoVec[2, ssSameLocations[:,1]]
                ssSameMultiplier = (phiPoints[ssSamet2s, alphas] - phiPoints[ssSamet1s, alphas])**4/s**8 - 6/s**2*(phiPoints[ssSamet2s, alphas] - phiPoints[ssSamet1s, alphas])**2/s**4 + 3/s**4
                beefCov[ssSameLocations[:,0], ssSameLocations[:,1]] = ssSameBase*ssSameMultiplier
            
            if len(ss3OAKLocations) != 0: 
                ss3OAKt1s, ss3OAKt2s = infoVec[0, ss3OAKLocations[:,0]], infoVec[0, ss3OAKLocations[:,1]]
                ss3OAKDiffVecs = phiPoints[ss3OAKt1s] - phiPoints[ss3OAKt2s]
                ss3OAKBase = np.exp(-1*np.linalg.norm(ss3OAKDiffVecs, axis=1)**2/(2*s**2))
                
                alphas, betas, gammas, deltas = infoVec[1, ss3OAKLocations[:,0]], infoVec[2, ss3OAKLocations[:,0]], infoVec[1, ss3OAKLocations[:,1]], infoVec[2, ss3OAKLocations[:,1]]
                ss3OAKMultiplier = []
                for i in range(len(alphas)):
                    indices = [alphas[i], betas[i], gammas[i], deltas[i]]
                    same = sp.stats.mode(indices)[0][0]
                    diff = np.setdiff1d(indices, [same], assume_unique=True)[0]
                    mult = (phiPoints[ss3OAKt2s[i], same] - phiPoints[ss3OAKt1s[i], same])/s**2*(phiPoints[ss3OAKt1s[i], diff] - phiPoints[ss3OAKt2s[i], diff])/s**2*(3/s**2 - (phiPoints[ss3OAKt2s[i], same] - phiPoints[ss3OAKt1s[i], same])**2/s**4)
                    ss3OAKMultiplier = np.append(ss3OAKMultiplier, mult)
                
                beefCov[ss3OAKLocations[:,0], ss3OAKLocations[:,1]] = ss3OAKBase*ss3OAKMultiplier
            
            if len(ss2ParLocations) != 0: 
                ss2Part1s, ss2Part2s = infoVec[0, ss2ParLocations[:,0]], infoVec[0, ss2ParLocations[:,1]]
                ss2ParDiffVecs = phiPoints[ss2Part1s] - phiPoints[ss2Part2s]
                ss2ParBase = np.exp(-1*np.linalg.norm(ss2ParDiffVecs, axis=1)**2/(2*s**2))
                
                alphas, betas, gammas, deltas = infoVec[1, ss2ParLocations[:,0]], infoVec[2, ss2ParLocations[:,0]], infoVec[1, ss2ParLocations[:,1]], infoVec[2, ss2ParLocations[:,1]]
                ss2ParMultiplier = []
                for i in range(len(alphas)):
                    indices = [alphas[i], betas[i], gammas[i], deltas[i]]
                    first, second = list(set(indices))[0], list(set(indices))[1]
                    
                    mult = (1/s**2 - (phiPoints[ss2Part2s[i], first] - phiPoints[ss2Part1s[i], first])**2/s**4)*(1/s**2 - (phiPoints[ss2Part2s[i], second] - phiPoints[ss2Part1s[i], second])**2/s**4)
                    ss2ParMultiplier = np.append(ss2ParMultiplier, mult)
                
                beefCov[ss2ParLocations[:,0], ss2ParLocations[:,1]] = ss2ParBase*ss2ParMultiplier
            
            if len(ss2OAKLocations) != 0:
                ss2OAKt1s, ss2OAKt2s = infoVec[0, ss2OAKLocations[:,0]], infoVec[0, ss2OAKLocations[:,1]]
                ss2OAKDiffVecs = phiPoints[ss2OAKt1s] - phiPoints[ss2OAKt2s]
                ss2OAKBase = np.exp(-1*np.linalg.norm(ss2OAKDiffVecs, axis=1)**2/(2*s**2))
                
                alphas, betas, gammas, deltas = infoVec[1, ss2OAKLocations[:,0]], infoVec[2, ss2OAKLocations[:,0]], infoVec[1, ss2OAKLocations[:,1]], infoVec[2, ss2OAKLocations[:,1]]
                ss2OAKMultiplier = []
                for i in range(len(alphas)):
                    indices = [alphas[i], betas[i], gammas[i], deltas[i]]
                    same = sp.stats.mode(indices)[0][0]
                    diff1, diff2 = list(np.setdiff1d(indices, [same], assume_unique=True))
                    mult = ss2OAKDiffVecs[i, diff1]/s**2*ss2OAKDiffVecs[i, diff2]/s**2*(-1/s**2 + ss2OAKDiffVecs[i, same]**2/s**4)
                    ss2OAKMultiplier = np.append(ss2OAKMultiplier, mult)
                
                beefCov[ss2OAKLocations[:,0], ss2OAKLocations[:,1]] = ss2OAKBase*ss2OAKMultiplier
            
            if len(ssDiffLocations) != 0: 
                ssDifft1s, ssDifft2s = infoVec[0, ssDiffLocations[:,0]], infoVec[0, ssDiffLocations[:,1]]
                ssDiffDiffVecs = phiPoints[ssDifft1s] - phiPoints[ssDifft2s]
                ssDiffBase = np.exp(-1*np.linalg.norm(ssDiffDiffVecs, axis=1)**2/(2*s**2))
                
                alphas, betas, gammas, deltas = infoVec[1, ssDiffLocations[:,0]], infoVec[2, ssDiffLocations[:,0]], infoVec[1, ssDiffLocations[:,1]], infoVec[2, ssDiffLocations[:,1]]
                ssDiffMultiplier = (phiPoints[ssDifft2s, alphas] - phiPoints[ssDifft1s, alphas])*(phiPoints[ssDifft2s, betas] - phiPoints[ssDifft1s, betas])*(phiPoints[ssDifft2s, gammas] - phiPoints[ssDifft1s, gammas])*(phiPoints[ssDifft2s, deltas] - phiPoints[ssDifft1s, deltas])/s**8
                beefCov[ssDiffLocations[:,0], ssDiffLocations[:,1]] = ssDiffBase*ssDiffMultiplier
            
            beefCov[ssLocations[:,1], ssLocations[:,0]] = beefCov[ssLocations[:,0], ssLocations[:,1]]
            
    return beefCov



def updateCholesky(beefCov_L, vcov, bar):
    
    lv = len(vcov)
    
    quartered = quarter(vcov, bar)
    ON, NN = quartered[1], quartered[3]
    
    n = sp.linalg.solve_triangular(beefCov_L[:bar,:bar], ON, lower = True, check_finite = False)
    toChol = NN - np.transpose(n) @ n
    l = sp.linalg.cholesky(toChol + noise*np.identity(len(toChol)), lower = True)    
    
    beefCov_L[bar:lv, :bar] = np.transpose(n)
    beefCov_L[bar:lv, bar:lv] = l
    
    return beefCov_L



###############################################################################
## -----------------------------PATH GENERATION----------------------------- ##
###############################################################################



class BStuff():
    
    def __init__(self, **kwargs):
        
        self.phi = kwargs.get('phi')
        self.dphi = kwargs.get('dphi')
        self.V = kwargs.get('V') 
        self.dV = kwargs.get('dV')
        self.ddV = kwargs.get('ddV') 
        self.t = kwargs.get('t')
        self.nE = kwargs.get('nE')
        self.params = kwargs.get('params')
        self.H = kwargs.get('H') 
        self.a = kwargs.get('a') 
        self.epsilon = kwargs.get('epsilon') 
        self.k = kwargs.get('k') 
        self.inw = kwargs.get('inw')
        self.vcov = kwargs.get('vcov') 
        self.vcov_L = kwargs.get('vcov_L')
        self.beefCov = np.zeros((maxCovLen,maxCovLen)) if kwargs.get('beefCov') is None else kwargs.get('beefCov')
        self.beefCov_L = np.zeros((maxCovLen,maxCovLen)) if kwargs.get('beefCov_L') is None else kwargs.get('beefCov_L')
        self.phiStuff = kwargs.get('phiStuff')
        self.vStuff = kwargs.get('vStuff') 
        self.vFull = kwargs.get('vFull')
        self.infoVec = kwargs.get('infoVec')
        self.infoFull = kwargs.get('infoFull')
        self.fulltimes = [] if kwargs.get('fulltimes') is None else kwargs.get('fulltimes')
        self.bowl = kwargs.get('bowl') 
        self.N = None if kwargs.get('N') is None else int(kwargs.get('N'))
        self.state = kwargs.get('state')
        self.nfev = 0 if kwargs.get('nfev') is None else kwargs.get('nfev')
        self.phiSol = kwargs.get('phiSol') 
        self.planck = plancklabel if kwargs.get('planck') is None else kwargs.get('planck') 
        self.upscale = kwargs.get('upscale') 
        self.simtime = kwargs.get('simtime') 
        self.fails = {"Inwardness": 0, "e-Folds": 0, "Total": 0, "Exception": False, "Overall": False} if kwargs.get('fails') is None else kwargs.get('fails')
        self.success = {"Inwardness": False, "Convergence": False, "e-Folds": False} if kwargs.get('success') is None else kwargs.get('success')
        self.forgets = 0 if kwargs.get('forgets') is None else kwargs.get('forgets')
        self.name = kwargs.get('name') 
        
        self.coupler = kwargs.get('coupler') 
        self.tensor_spec = kwargs.get('tensor_spec') 
        self.A_t = kwargs.get('A_t') 
        self.n_t = kwargs.get('n_t')
        self.A_s = kwargs.get('A_s') 
        self.n_s = kwargs.get('n_s')
        self.A_iso = kwargs.get('A_iso')
        self.n_iso = kwargs.get('n_iso')
        self.r = kwargs.get('r')
        
        return
        
    def __str__(self, shapes=False): 
        print(f"Parameters:\ns (Coherence scale) = {self.params[0]}\nV_0 (Inflationary Energy Scale) = {self.params[1]}\nN (# of inflaton fields) = {self.params[2]}\n")
        print(f"Planck units: {self.planck}")
        print(f"Simulation time: {self.simtime} seconds")
        print(f"Name: {self.name}\n")
        print(f"Quantities:\nA_t = {self.A_t}, n_t = {self.n_t}\nA_s = {self.A_s}, n_s = {self.n_s}\nr = {self.r}")
        if shapes:
            print(f"\nShape of phi: {np.shape(self.phi)}")
            print(f"Shape of d(phi)/dN_e: {np.shape(self.dphi)}")
            print(f"Shape of V: {np.shape(self.V)}")
            print(f"Shape of grad(V): {np.shape(self.dV)}")
            print(f"Shape of laplacian(V): {np.shape(self.ddV)}")
            print(f"Shape of t: {np.shape(self.t)}")
            print(f"Shape of nE: {np.shape(self.nE)}")
            print(f"Shape of H: {np.shape(self.H)}")
            print(f"Shape of a: {np.shape(self.a)}")
            print(f"Shape of epsilon: {np.shape(self.epsilon)}")
            print(f"Shape of k: {np.shape(self.k)}")
            print(f"Shape of V covariance matrix: {np.shape(self.vcov)}")
            print(f"Shape of coupling tensor: {np.shape(self.coupler)}")
            print(f"Shape of tensor spectrum: {np.shape(self.tensor_spec)}")
        return f"Random path {self.name}."
    
    def s(self): return self.params[0]
    def V0(self): return self.params[1]
    def strongMin(self): return self.V0()/self.s()**2
    def minStrength(self): return self.bowl/self.strongMin()*100
    def nEBounds(self): I = self.V[0]/np.linalg.norm(self.dV[:,0])*np.linalg.norm(self.phi[:,0]); return (I/4, 3*I/4) 
    
    def phiFunc(self, nEVals, strictlyInc=False, d=0):
        
        funclist = np.empty(self.N, dtype=object)
        for dim in range(self.N):
            if strictlyInc: funclist[dim] = CS(self.nE, self.phi[dim]).derivative(d)
            else: funclist[dim] = sp.interpolate.interp1d(self.nE, self.phi[dim])
        phiVals = np.zeros((self.N, len(nEVals)))
        for i in range(len(nEVals)):
            for j in range(len(funclist)):
                phiVals[j,i] = funclist[j](nEVals[i])
        
        return phiVals
    
    def dphiFunc(self, nEVals, strictlyInc=False, d=0):
        
        funclist = np.empty(self.N, dtype=object)
        for dim in range(self.N):
            if strictlyInc: funclist[dim] = CS(self.nE, self.dphi[dim]).derivative(d)
            else: funclist[dim] = sp.interpolate.interp1d(self.nE, self.dphi[dim])
        dphiVals = np.zeros((self.N, len(nEVals)))
        for i in range(len(nEVals)):
            for j in range(len(funclist)):
                dphiVals[j,i] = funclist[j](nEVals[i])
        
        return dphiVals
    
    def VFunc(self, nEVals, strictlyInc=False, d=0):
        
        func = []
        if strictlyInc: func = CS(self.nE, self.V).derivative(d)
        else: 
            func = sp.interpolate.interp1d(self.nE, self.V)
            if d>0: print("Time values must be strictly increasing to get derivatives.")
        VVals = np.zeros(len(nEVals))
        for i in range(len(nEVals)):
            VVals[i] = func(nEVals[i])
        
        return VVals
    
    def dVFunc(self, nEVals, strictlyInc=False, d=0):
        
        funclist = np.empty(self.N, dtype=object)
        for dim in range(self.N):
            if strictlyInc: funclist[dim] = CS(self.nE, self.dV[dim]).derivative(d)
            else: funclist[dim] = sp.interpolate.interp1d(self.nE, self.dV[dim])
        dVVals = np.zeros((self.N, len(nEVals)))
        for i in range(len(nEVals)):
            for j in range(len(funclist)):
                dVVals[j,i] = funclist[j](nEVals[i])
        
        return dVVals
    
    def traj_df(self): 
        
        cols = ["N_e", "t", "H", "a", "epsilon", "V", "k", "inwardness"]
        cols = np.append(cols, [f"dV{i}" for i in range(self.N)])
        cols = np.append(cols, [f"phi{i}" for i in range(self.N)])
        cols = np.append(cols, [f"dphi{i}" for i in range(self.N)])
        df = pd.DataFrame(columns = cols)
        
        df['N_e'] = self.nE
        df['t'] = self.t
        df['H'] = self.H
        df['a'] = self.a
        df['epsilon'] = self.epsilon
        df['V'] = self.V
        df['k'] = self.k
        df['inwardness'] = self.inw
        for i in range(self.N):
            if self.phi is not None: df[f'phi{i}'] = self.phi[i]
            if self.dphi is not None: df[f'dphi{i}'] = self.dphi[i]
            if self.dV is not None: df[f'dV{i}'] = self.dV[i]
        
        return df
    
    def stats_df(self):
         
        cols = ['id', 's', 'V0', 'N', 'from0', 'N_e', 'A_t', 'n_t', 'A_s', 'n_s', 'A_iso', 'n_iso', 'r', 'sim_time', 'nfev', 'forgets', 'upscale', 'fails:inwardness', 'fails:e-folds', 'fails:total', 'fails:exception', 'fails:overall', 'success:convergence', 'success:e-folds']
        cols = np.append(cols, [f'bowl{i}' for i in range(self.N)])
        df = pd.DataFrame(columns = cols)
        
        df['id'] = [hex(id(self))] if self.name is None else [self.name]
        df['s'] = [self.s()]
        df['V0'] = [self.V0()]
        df['N'] = [self.N]
        if self.phi is not None: df['from0'] = np.linalg.norm(self.phi[:,0])/self.s()
        if self.nE is not None: df['N_e'] = self.nE[-1]
        df['A_t'] = [self.A_t]
        df['n_t'] = [self.n_t]
        df['A_s'] = [self.A_s]
        df['n_s'] = [self.n_s]
        df['A_iso'] = [self.A_iso]
        df['n_iso'] = [self.n_iso]
        df['r'] = [self.r]
        df['sim_time'] = self.simtime
        df['nfev'] = self.nfev
        df['forgets'] = self.forgets
        df['upscale'] = self.upscale
        df['fails:inwardness'] = self.fails['Inwardness']
        df['fails:e-folds'] = self.fails['e-Folds']
        df['fails:total'] = self.fails['Total']
        df['fails:exception'] = self.fails['Exception']
        df['fails:overall'] = self.fails['Overall']
        df['success:convergence'] = self.success['Convergence']
        df['success:e-folds'] = self.success['e-Folds']
        for i in range(self.N):
            df[f'bowl{i}'] = self.bowl[i]
        
        return df



def getVSkew(cov):

    L = sp.linalg.cholesky(cov + noise*np.identity(len(cov)), lower=True)
    y = np.random.normal(0, 1, len(cov))
    mush = L @ y
    
    return mush



def quarter(m, definedCut):
    
    return m[:definedCut, :definedCut], m[:definedCut, definedCut:], m[definedCut:, :definedCut], m[definedCut:, definedCut:]



def beefItUp(m, beef):
    
    m = np.concatenate((m, np.zeros((beef, len(m)))), axis=0)
    m = np.concatenate((m, np.zeros((len(m), beef))), axis=1)
    
    return m



def sampleVData(b, phiPoint, d1=(-1,), d2=(-1,), include=True):
    
    d1, d2 = np.array(d1), np.array(d2)
    if len(d1) != len(d2): d2 = np.append( d2, [-1]*(len(d1)-len(d2)) ) if len(d1) > len(d2) else print("Supply first derivatives, too!")
    if max(np.append(d1, d2) > b.N-1): print("There aren't that many dimensions!")
    if (len(b.vcov) + len(d1) >= maxCovLen) and include: forgetEarlies(b, cut=1/2)
    elif len(b.vcov) + len(d1) >= maxCovLen: print("No more space - set include True to forget"); return
    
    phiList, vStuff, infoVec, s, V0, N = b.phiStuff[0], b.vStuff, b.infoVec, b.s(), b.V0(), b.N
    vcov, beefCov, vcov_L, beefCov_L = b.vcov, b.beefCov, b.vcov_L, b.beefCov_L
    
    phiList = np.concatenate((phiList, np.transpose([phiPoint])), axis=1)
    infoPart = np.array([[infoVec[0,-1]+1]*len(d1), d1, d2])
    infoVec = np.concatenate((infoVec, infoPart), axis=1)
    
    bar = len(vcov)
    beefCov = updateBeefCOV(s, phiList, infoVec, beefCov, bar)
    vcov = beefCov[:len(infoVec[0]), :len(infoVec[0])]
    OO, ON, NO, NN = quarter(vcov, -len(d1))
    
    halfSandwich = sp.linalg.solve_triangular(vcov_L, ON, lower = True)
    gammaC = NN - (np.transpose(halfSandwich) @ halfSandwich) ## Cleverly, this is NO @ OOI @ ON
    skew = getVSkew(gammaC)*V0
    
    mu = np.transpose(halfSandwich) @ sp.linalg.solve_triangular(vcov_L, vStuff, lower = True) ## Cleverly again, this is NO @ OOI @ xO
    vNew = mu + skew
    
    if include:
        b.phiStuff = np.concatenate((b.phiStuff, np.array([phiPoint, np.zeros(N)]).reshape(2,N,1)), axis=2)
        b.vStuff = np.append(b.vStuff, mu + skew)
        b.vFull = np.append(b.vFull, mu + skew) if b.vFull is not None else b.vStuff
        b.infoVec = infoVec
        b.infoFull = np.concatenate((b.infoFull, infoPart), axis=1) if b.infoFull is not None else b.infoVec
        b.beefCov = beefCov
        b.vcov = vcov
        b.beefCov_L = updateCholesky(beefCov_L, vcov, bar)
        b.vcov_L = b.beefCov_L[:len(infoVec[0]), :len(infoVec[0])]
        b.nfev = b.nfev + 1
        
    return vNew



def sampleVConcavity(b, points=10, include=True, annotate=True):
    
    nE, N = b.nE, b.N
    sparsenE = np.linspace(nE[0], nE[-1], points)
    sparsephi = b.phiFunc(sparsenE)
    
    ddV = np.zeros((N,N,points))
    offdiags = np.array(list(itertools.combinations(np.arange(N), r=2)))
    
    for p in range(points):
        if annotate: print(f"Simulating d2V/d(phi)2 at N_e = {np.round(sparsenE[p],3)}...")
        if N>1: 
            sampleVData(b, sparsephi[:,p], d1=offdiags[:,0], d2=offdiags[:,1])
            ddV[offdiags[:,0], offdiags[:,1], p] = b.vStuff[-len(offdiags):]
            ddV[offdiags[:,1], offdiags[:,0], p] = ddV[offdiags[:,0], offdiags[:,1], p]
        for i in range(N): 
            sampleVData(b, sparsephi[:,p], d1=(i,), d2=(i,))
            ddV[i,i,p] = b.vStuff[-1]
    
    ddVfunc = np.empty((N,N), dtype=object)
    ddVfull = np.zeros((N,N,len(nE)))
    for i in range(N):
        for j in range(N):
            ddVfunc[i,j] = CS(sparsenE, ddV[i,j])
            ddVfull[i,j] = ddVfunc[i,j](nE)
    
    if include: b.ddV = ddVfull
    
    return ddVfull



def unitize(b, redoVCOV=False):
    
    s, V0 = b.params[:2]
    
    if isinstance(b.phi, np.ndarray): b.phi = b.phi*s
    if isinstance(b.dphi, np.ndarray): b.dphi = b.dphi*s
    if isinstance(b.V, np.ndarray): b.V = b.V*V0
    if isinstance(b.dV, np.ndarray): b.dV = b.dV*V0/s
    if isinstance(b.ddV, np.ndarray): b.ddV = b.ddV*V0/s**2
    if isinstance(b.phiStuff, np.ndarray): b.phiStuff = b.phiStuff*s
    if isinstance(b.vStuff, np.ndarray): 
        b.vStuff = b.vStuff*V0
        b.vStuff[b.infoVec[1] != -1] = b.vStuff[b.infoVec[1] != -1]/s
        b.vStuff[b.infoVec[2] != -1] = b.vStuff[b.infoVec[2] != -1]/s
    if isinstance(b.vFull, np.ndarray):
        b.vFull = b.vFull*V0
        b.vFull[b.infoFull[1] != -1] = b.vFull[b.infoFull[1] != -1]/s
        b.vFull[b.infoFull[2] != -1] = b.vFull[b.infoFull[2] != -1]/s
    if redoVCOV and isinstance(b.vcov, np.ndarray): b.vcov = initiateVCOV(s, b.phiStuff[0], b.infoVec) 
    
    return b
    


def forceMin(s, V0, N): 
    
    b = BStuff(params = (s,V0,N), N=N)
    
    ## Zeroeth and first derivatives forced to zero.
    b.phiStuff = np.zeros((2,N,1))
    b.infoVec = np.array([np.zeros((N+1)), np.arange(-1,N), -1*np.ones(N+1)], dtype=int)
    b.vStuff = np.zeros((N+1))
    b.vcov = initiateVCOV(s, b.phiStuff[0], b.infoVec)
    b.vcov_L = sp.linalg.cholesky(b.vcov + noise*np.identity(N+1), lower = True)
    b.beefCov[:N+1, :N+1] = b.vcov
    b.beefCov_L[:N+1, :N+1] = b.vcov_L
    
    ## Randomly simulate diagonal of second derivative matrix, then make them positive.
    sampleVData(b, np.zeros(N), d1=np.arange(N), d2=np.arange(N))
    b.bowl = np.abs(b.vStuff[-N:])
    b.vStuff[-N:] = b.bowl        
    
    ## Set off-diagonals of second derivative matrix to zero.
    if N > 1:
        mixed = np.transpose(np.array(list(itertools.permutations(np.arange(N), 2))))
        infoPart = np.concatenate((np.array([np.ones(len(mixed[0]))]), mixed))
        b.infoVec = np.concatenate((b.infoVec, infoPart), axis=1)
        
        bar = len(b.vcov)
        b.beefCov = updateBeefCOV(s, b.phiStuff[0], b.infoVec, b.beefCov, bar)
        b.vcov = b.beefCov[:len(b.infoVec[0]), :len(b.infoVec[0])]
        b.beefCov_L = updateCholesky(b.beefCov_L, b.vcov, bar)
        b.vcov_L = b.beefCov_L[:len(b.infoVec[0]), :len(b.infoVec[0])]
        
        b.vStuff = np.append(b.vStuff, np.zeros(len(mixed[0])))
    
    b.nfev = 1
    
    return b



def testStart(b, phi0, s, nE_bounds):
    
    vNew = sampleVData(b, phi0, d1=np.arange(-1,b.N), include=True)
    V, dV = vNew[0], vNew[1:]
    phihat, dVhat = phi0/np.linalg.norm(phi0), dV/np.linalg.norm(dV)
    nEEst = 1/2*np.linalg.norm(phi0)*V/np.linalg.norm(dV)*s**2
    inwardness = np.dot(phihat, dVhat)
    enough_nE = (nEEst > nE_bounds[0] or nE_bounds[0] is None) and (nEEst < nE_bounds[1] or nE_bounds[1] is None)
    inward = inwardness > 0
    
    return enough_nE, inward, nEEst, inwardness



def formatTraj(b):
   
    b.nE, newnE = b.fulltimes, []
    resorter = np.argsort(b.fulltimes)
    fullV = b.vFull[np.where(b.infoFull[1]==-1)[0]]
    fulldV = np.reshape(b.vFull[np.intersect1d(np.where(b.infoFull[1]!=-1)[0], np.where(b.infoFull[2]==-1)[0])], (b.N,len(fullV)), order='F')
    if b.phiSol is None:
        fullphi, fulldphi = b.phiStuff[0,:,2:len(b.fulltimes)+2], b.phiStuff[1,:,2:len(b.fulltimes)+2]
        fullV, fulldV = fullV[1:len(b.nE)+1], fulldV[:,1:len(b.nE)+1]
        newnE = np.linspace(min(b.fulltimes), max(b.fulltimes), 200)
        b.phi, b.dphi = fullphi[:,resorter], fulldphi[:,resorter]
        b.phi, b.dphi = b.phiFunc(newnE), b.dphiFunc(newnE)
    else:
        b.phi, b.dphi = b.phiSol.y[:b.N], b.phiSol.y[b.N:]
        fullV, fulldV = fullV[1:], fulldV[:,1:]
        newnE = b.phiSol.t
    b.V, b.dV = fullV[resorter], fulldV[:,resorter] ## dummy step to use d/VFunc and sample at b.nE
    b.V, b.dV = b.VFunc(newnE), b.dVFunc(newnE)
    
    b.nE = newnE
    
    nans = np.isnan(b.nE) | np.isnan(b.phi[0]) | np.isnan(b.dphi[0]) | np.isnan(b.V) | np.isnan(b.dV[0])
    b.nE, b.phi, b.dphi, b.V, b.dV = b.nE[~nans], b.phi[:,~nans], b.dphi[:,~nans], b.V[~nans], b.dV[:,~nans]
    
    return b



def determineSuccessfulness(b, nEBack4Psi, buffer):
    
    conv = np.linalg.norm(b.phi[:,-1])<1/10
    nE = b.nE[-1] > nEBack4Pivot + nEBack4Psi + buffer
    b.success = {"Convergence": conv, "e-Folds": nE}
    
    return b



def calculateQuantities(b, which, **kwargs):
    
    if which=='basic':
        nEPivot = b.nE[-1] - nEBack4Pivot
        b.epsilon = np.linalg.norm(b.dphi, axis=0)**2/2
        b.H = np.sqrt(b.V/(3 - b.epsilon))
        b.a = np.exp(b.nE - nEPivot)
        b.t = np.append([0], sp.integrate.cumtrapz(1/b.H, x = b.nE))
        b.inw = [np.dot(b.phi[:,i]/np.linalg.norm(b.phi[:,i]), b.dV[:,i]/np.linalg.norm(b.dV[:,i])) for i in range(len(b.dV[0]))]
        if nEPivot>0:
            Hfunc = CS(b.nE, b.H)
            b.upscale = Hfunc(nEPivot)/kStar
            b.k = b.a*b.H/b.upscale   
    if which=='adv':
        psi_integrate = kwargs.get('psi_integrate')
        b.A_t, b.n_t, b.tensor_spec = getTensorSpectrum(b)
        b.coupler = getCouplingTensor(b)
        b.A_s, b.n_s, b.A_iso, b.n_iso = getMatterSpectrum(b, nE_integrate=psi_integrate)
        b.r = getSpectralRatio(b)
    
    return b



def forgetEarlies(b, cut=1/2):
    
    ## Update vcov, vcov_L, beefCov, beefCov_L, phiStuff, vStuff, infoVec, NOT nfev
    vcov = b.vcov
    seedNum = b.N**2 + b.N + 1
    cut = int((len(vcov)-seedNum)*cut)
    newvcov = np.zeros((len(vcov)-cut, len(vcov)-cut))
    newvcov[:seedNum, :seedNum] = vcov[:seedNum, :seedNum]
    newvcov[seedNum:, seedNum:] = vcov[(seedNum+cut):, (seedNum+cut):]
    newvcov[:seedNum, seedNum:] = vcov[:seedNum, (seedNum+cut):]
    newvcov[seedNum:, :seedNum] = np.transpose(newvcov[:seedNum, seedNum:])
    
    b.vcov = newvcov
    b.vcov_L = sp.linalg.cholesky(b.vcov + noise*np.identity(len(b.vcov)), lower=True)
    b.beefCov, b.beefCov_L = np.zeros((2,maxCovLen,maxCovLen))
    b.beefCov[:len(b.vcov),:len(b.vcov)] = b.vcov
    b.beefCov_L[:len(b.vcov),:len(b.vcov)] = b.vcov_L
    b.vStuff = np.append(b.vStuff[:seedNum], b.vStuff[(seedNum+cut):])
    b.infoVec = np.concatenate((b.infoVec[:,:seedNum], b.infoVec[:,(seedNum+cut):]), axis=1)
    ## phiStuff shouldn't have to be changed since it is referenced properly by infoVec
    
    b.forgets += 1
    
    return b



def annotater(section, **kwargs):
    
    if section=='start':
        print("Finding minimum...\n")
    if section=='min found':
        from0 = kwargs.get('from0')
        good = kwargs.get('good')
        fails = kwargs.get('fails')
        nEEst = kwargs.get('nEEst')
        inwardness = kwargs.get('inwardness')
        print(f"Minimum found: {good}")
        print(f"Fails: Inwardness - {fails['Inwardness']}, e-Folds - {fails['e-Folds']}, Total - {fails['Total']}")
        if good:
            print(f"|phi_0| = {from0}")
            print(f"Est. # e-folds: {nEEst}")
            print(f"Inwardness: {inwardness}\n")
    if section=='during':
        nE = kwargs.get('nE')
        phi = kwargs.get('phi')
        print(f"Integrating eqs. of motion at N_e = {np.round(nE,3)}, |phi| = {np.round(np.linalg.norm(phi),3)}...")
    if section=='exception':
        e = kwargs.get('e')
        print(f"\nIntegration failure, exception raised:\n{repr(e)}\n")
    if section=='solved':
        success = kwargs.get('success')
        calculable = kwargs.get('calculable')
        print("Integration finished.\n")
        print(f"Ended near origin: {success['Convergence']}")
        print(f"Enough inflation: {success['e-Folds']}")
        print(f"Mode equations solvable: {calculable}")
        if calculable: print("\nComputing d2V/d(phi)2 and reformatting...\n")
    if section=='modes':
        print("\nSolving mode equations...\n")
    if section=='failure':
        print("Maximum allowable failures reached with given parameters.")

    return



def bushwhack(s = 30, V0 = 5e-9, N = 2, from0_bounds = (0.2,0.8), nE_bounds = (45,500), psi_integrate = (-3,None), 
              fails_coef = 100, maxJumpFrac = 1/20, nE_buffer = 2, state = None, method = 'RK45', name = None, annotate = True): 
    
    start = time.time()
    if annotate: annotater('start')
    np.random.seed()
    if state is not None: np.random.set_state(state)
    state = np.random.get_state()
    
    phi0, from0, enough_nE, inward, fails_nE, fails_inward, fails, nEEst, inwardness = None, None, False, False, -1, -1, -1, 0, 0
    while not (enough_nE and inward) and fails < fails_coef*2**(N-1):
        
        if not enough_nE: fails_nE += 1
        if not inward: fails_inward += 1
        if not (enough_nE and inward): fails += 1
        b = forceMin(1, 1, N)
        b.fails['e-Folds'], b.fails['Inwardness'], b.fails['Total'] = fails_nE, fails_inward, fails
        b.bowl *= V0/s**2
        
        phi0, from0 = np.random.uniform(-1,1,N), np.random.uniform(from0_bounds[0], from0_bounds[1])
        phi0 = from0*phi0/np.linalg.norm(phi0)
        enough_nE, inward, nEEst, inwardness = testStart(b, phi0, s, nE_bounds)
    
    if annotate: annotater('min found', from0=from0, good=(enough_nE and inward), fails=b.fails, nEEst=nEEst, inwardness=inwardness)
    dphi0 = -b.vStuff[-N:]/(b.vStuff[-N-1]*s**2)
    if (enough_nE and inward):
        
        def expertCosmo(nE, phiFlat): 
            
            phi, dphi = np.reshape(phiFlat, (2,N))
            b.phiStuff[1,:,-1] = dphi
            b.fulltimes = np.append(b.fulltimes, [nE])
            
            vNew = sampleVData(b, phi, d1=np.arange(-1,N))
            V, dV = vNew[0], vNew[1:]
            phi, dphi, V, dV = phi*s, dphi*s, V*V0, dV*V0/s
            
            epsilon = np.dot(dphi,dphi)/2
            H = np.sqrt(V/(3 - epsilon))
            ddphi = (epsilon - 3)*dphi - 1/H**2*dV
            
            if b.nfev%20==0 and annotate: annotater('during', nE=nE, phi=phi)
            
            return np.append(dphi/s, ddphi/s).flatten()
        
        def stopCriterion(nE, phiFlat): 
            
            phi, dphi = np.reshape(phiFlat, (2,N))
            dphi = dphi*s
            epsilon = np.dot(dphi,dphi)/2
            
            return epsilon - 1
        
        def failCriterion(nE, phiFlat): 
            
            phi, dphi = np.reshape(phiFlat, (2,N))
            phi, dV = phi*s, b.vStuff[-N:]
            phihat, dVhat = phi/np.linalg.norm(phi), dV/np.linalg.norm(dV)
            
            return np.dot(phihat, dVhat)
        
        def submerging(nE, phiFlat):
            
            return b.vStuff[-N-1]
        
        stopCriterion.terminal = True
        failCriterion.terminal = True
        submerging.terminal = True
        nEEnd = nEEst*10
        
        try: b.phiSol = sp.integrate.solve_ivp(expertCosmo, [0,nEEnd], np.append(phi0,dphi0), events = (stopCriterion, failCriterion, submerging), 
                                               method = method, max_step = nEEnd*maxJumpFrac, first_step = 0.8)
        except Exception as e: 
            if annotate: annotater('exception', e=e)
            b.fails['Exception'] = True
        
        formatTraj(b)
        determineSuccessfulness(b, -1*psi_integrate[0], 5)
        
        calculable = all(val for val in b.success.values())
        if annotate: annotater('solved', success=b.success, calculable=calculable)
        if calculable: sampleVConcavity(b, annotate=annotate)
            
        b.params = (s,V0,N)
        unitize(b, redoVCOV=False)
        calculateQuantities(b, which='basic')
        
        if calculable:
            if annotate: annotater('modes')
            calculateQuantities(b, which='adv', psi_integrate=psi_integrate)
        else:
            b.fails['Overall'] = True
            
    else:
        b.fails['Overall'] = True
        b.params = (s,V0,N)
        if annotate: annotater("failure")
    b.name = np.random.randint(1e12, dtype=np.int64)
    b.state = np.array(state, dtype=object)
    b.simtime = time.time() - start
    if annotate: print("Finished.\n"); print(b)
    
    return b



###############################################################################
## -------------------------COSMOLOGICAL QUANTITIES------------------------- ##
###############################################################################



def kPowerFit(kVals, quantities):
    
    p = np.polyfit(np.log(kVals), np.log(quantities), 1)
    A_q, n_q = np.exp(p[1] + p[0]*np.log(kStar)), p[0]
    
    return A_q, n_q



def getTensorSpectrum(b, points=5, plot=False):
    
    spectrum = (2/np.pi**2)*b.H**2
    specfunc = sp.interpolate.interp1d(b.k, spectrum)
    kVals = kPivot*np.logspace(-0.4*2, 0.4*2, points, base=np.e)
    A_t, n_t = kPowerFit(kVals, specfunc(kVals))
    
    if plot:
        plt.loglog(b.k, spectrum, c='b', label='Spectrum')
        plt.scatter(kVals, specfunc(kVals), marker='+', c='k', label='Sampled Points')
        plt.loglog(b.k, A_t*(b.k/kStar)**n_t, c='g', label="Power Law Fit")
        plt.legend() and plt.title("Tensor Perturbation Spectrum") and plt.xlabel(r"Wavenumber [$l_{Pl}^{-1}$]")
    
    return A_t, n_t, spectrum



def getCouplingTensor(b, cont=False, plot=False): 
    
    dV, ddV, dphi, nE, H, epsilon, N = b.dV, b.ddV, b.dphi, b.nE, b.H, b.epsilon, b.N
    C = np.empty((N,N,len(nE)))
    for i in range(N):
        for j in range(N):
            term1 = ddV[i,j]/H**2
            term2 = 1/H**2*dphi[i]*dV[j]
            term3 = 1/H**2*dphi[j]*dV[i]
            term4 = (3 - epsilon)*dphi[i]*dphi[j]
            C[i,j] = term1 + term2 + term3 + term4
    
    if cont:
        Cfuncs = np.empty((N,N), dtype=object)
        for i in range(N):
            for j in range(N):
                Cfuncs[i,j] = CS(nE, C[i,j])
        C = Cfuncs
        
    if plot:
        for i in range(N):
            for j in range(N):
                plt.semilogy(nE, np.abs(C[i,j]))
        plt.xlabel("Number of e-Folds") and plt.ylabel(r"Abs. Value $C[i,j]$")
    
    return C
    
    

def getMatterSpectrumPoint(b, k=None, nE_integrate=(-5,None), ICs=(0,1)):
    
    nE, a, H, epsilon, N = b.nE, b.a, b.H, b.epsilon, b.N
    aHfunc, epsfunc = CS(nE, a*H), CS(nE, epsilon)
    C = getCouplingTensor(b, cont=True)
    
    nE_kfunc, kScaled, kCorrect = sp.interpolate.interp1d(b.k, b.nE), None, None
    if k is None: kScaled, kCorrect = kStar*b.upscale, kStar
    else: kScaled, kCorrect = k*b.upscale, k
    nECrossing = nE_kfunc(kCorrect)
    
    def callCoupler(nE):
        CVals = np.empty((N,N))
        for i in range(N):
            for j in range(N):
                CVals[i,j] = C[i,j](nE)
        return CVals
    
    def f(nE, psiFlat):
        psiStuff = np.reshape(psiFlat, (2,N,N))
        psi, dpsi = psiStuff[0], psiStuff[1]
        
        term1 = (epsfunc(nE) - 1)*dpsi
        term2 = (2 - (kScaled/aHfunc(nE))**2 - epsfunc(nE))*psi
        term3 = -1*callCoupler(nE) @ psi
        ddpsi = term1 + term2 + term3
        return np.append(dpsi.flatten(), ddpsi.flatten())
    
    nEStart = nE[0] if nE_integrate[0] is None else nECrossing + nE_integrate[0]
    nEEnd = nE[-1] if nE_integrate[1] is None else nECrossing + nE_integrate[1]
    
    psi0 = np.identity(N)*(ICs[0] + ICs[1])/np.sqrt(2*kCorrect)
    dpsi0 = np.identity(N)*kScaled*1j/(aHfunc(nEStart)*(ICs[0] - ICs[1])*np.sqrt(2*kCorrect))
    
    psiStuff = sp.integrate.solve_ivp(f, (nEStart, nEEnd), np.append(psi0.flatten(), dpsi0.flatten()), max_step=(nEEnd-nEStart)/500)             
    psi = np.reshape(psiStuff.y[:N**2], (N,N,len(psiStuff.t)))
    
    fieldSpace = kCorrect/(2*np.pi**2)*(kScaled/a[-1])**2*(psi[:,:,-1] @ np.matrix(psi[:,:,-1]).H)
    omega = b.dphi[:,-1].flatten()
    omega = omega/np.linalg.norm(omega)
    sK = np.transpose(sp.linalg.null_space([omega]))
    adiabatic = np.real(np.sum(omega @ fieldSpace @ np.transpose(omega))/2)
    isocurvature = np.real(np.trace((sK @ fieldSpace @ np.transpose(sK))/2))
    #cross = np.real(np.sum((omega @ (fieldSpace + np.transpose(fieldSpace)) @ np.transpose(sK))/2))
    
    return adiabatic, isocurvature



def getMatterSpectrum(b, points=5, nE_integrate=(-3,None), ICs=(0,1), plot=False):
    
    adiabatics, isocurvatures = [], [] 
    kVals = kPivot*np.logspace(-0.4*2, 0.4*2, points, base=np.e)
    for k in kVals: 
        adiabatic, isocurvature = getMatterSpectrumPoint(b, k=k, nE_integrate=nE_integrate, ICs=ICs)
        adiabatics = np.append(adiabatics, adiabatic)
        isocurvatures = np.append(isocurvatures, isocurvature)
    
    A_s, n_s = kPowerFit(kVals, adiabatics)
    A_iso, n_iso = 0, 0
    if b.N>1: A_iso, n_iso = kPowerFit(kVals, isocurvatures)
    n_s, n_iso = n_s+1, n_iso+1
    
    if plot:
        plt.scatter(kVals, adiabatics, marker='+', c='k', label="Sampled Points")
        plt.loglog(b.k, A_s*(b.k/kStar)**(n_s - 1), label="Power Law Fit")
        plt.xlabel("Wavenumber") and plt.ylabel(r"$P_R(k)$") and plt.title("Adiabatic Curvature Power Spectrum")
        plt.legend()
    
    return A_s, n_s, A_iso, n_iso



def getSpectralRatio(b):
    
    return b.A_t/b.A_s*(kStar/kPivot)**(1+b.n_t-b.n_s)



###############################################################################
## ---------------------------------PLOTTERS-------------------------------- ##
###############################################################################



def masterPlot(b):
    
    fig, axs = plt.subplots(2,4,figsize=(20,10))
    [axs[0,0].plot(b.nE, b.phi[i], label=rf"$\phi_{i}$") for i in range(b.N)]
    axs[0,0].set_title("Inflaton Values") and axs[0,0].legend()
    [axs[1,0].plot(b.nE, b.dphi[i], label=rf"$d\phi_{i}/dN_e$") for i in range(b.N)]
    axs[1,0].set_title("Inflaton Component Velocities") and axs[1,0].legend()
    axs[0,1].plot(b.nE, b.V) and axs[0,1].set_title("Inflaton Potential")
    [axs[1,1].plot(b.nE, b.dV[i], label=rf"$\partial V/\partial\phi_{i}$") for i in range(b.N)]
    axs[1,1].set_title("Inflaton Potential Gradient") and axs[1,1].legend()
    axs[0,2].plot(b.nE, b.H) and axs[0,2].set_title("Hubble Parameter")
    axs[1,2].plot(b.t, b.nE) and axs[1,2].set_title("Inflation")
    axs[0,3].loglog(b.k, (2/np.pi**2)*b.H**2, label="Spectrum") 
    axs[0,3].loglog(b.k, b.A_t*(b.k/kStar)**b.n_t, label="Power Law Fit")
    axs[0,3].set_title("Tensor Perturbation Spectrum") and axs[0,3].legend()
    
    axs[1,3].loglog(b.k, b.A_s*(b.k/kStar)**(b.n_s - 1), label="Power Law Fit")
    axs[1,3].set_title("Matter Perturbation Spectrum")
    
    try: 
        if b.name is not None: fig.suptitle(f"Master Plot of b:{hex(id(b))} AKA $\it{b.name}$", fontsize=30)
    except: 
        fig.suptitle(f"Master Plot of {hex(id(b))}", fontsize=30)
    return



def manyPlot(b, d=0):
    
    if d==0: [plt.plot(b.nE, b.phi[i], label=rf"$\phi_{i}$") for i in range(b.N)]
    if d==1: [plt.plot(b.nE, b.dphi[i], label=rf"$d\phi_{i}/dN_e$") for i in range(b.N)]
    
    plt.xlabel("# of e-Folds") and plt.legend()
    plt.ylabel("Inflaton Component Velocities") if d==1 else plt.ylabel("Inflaton Components")
    
    return



def VPlot(b, d1=-1, d2=-1):
    
    if d1==-1: plt.plot(b.nE, b.V) and plt.ylabel("Inflaton Potential Value")
    if d1!=-1 and d2==-1: plt.plot(b.nE, b.dV[d1]) and plt.ylabel(f"Gradient of Potential: Axis {d1}")
    if d1!=-1 and d2!=-1: plt.plot(b.nE, b.ddV[d1,d2]) and plt.ylabel(f"Concavity of Potential: Axes {d1}, {d2}")
    
    plt.xlabel("# of e-Folds")
    
    return



def crossSectionPlot(b, i=0):
    
    if isinstance(i, int): plt.plot(b.phi[i], b.V, label = rf"$V(\phi_{i})$")
    else: [plt.plot(b.phi[dim], b.V, label = rf"$V(\phi_{dim})$") for dim in i]
    plt.xlabel("Inflaton Space") and plt.ylabel("Inflaton Potential Value") and plt.legend()
    
    return



def topoPlot(b, i=(0,1)):
    
    if len(i)!=2 or b.N<max(i)+1: return "Cannot execute topoPlot with given data."
    
    plt.scatter(b.phi[i[0]], b.phi[i[1]], c=b.V, marker='+') and plt.colorbar()
    plt.axis("scaled") and plt.xlabel(rf"$\phi_{i[0]}$") and plt.ylabel(rf"$\phi_{i[1]}$")
    plt.title("Topographic Plot")
    
    return



def earthwormPlot(b, i=(0,1,2)):
    
    if b.N<3 or len(i)!=3: return "Cannot execute earthwormPlot with given input."
    
    ax = plt.figure().add_subplot(111, projection='3d')
    sc = ax.scatter(b.phi[i[0]], b.phi[i[1]], b.phi[i[2]], c = b.V)
    ax.set_xlabel(rf"$\phi_{i[0]}$") and ax.set_ylabel(rf"$\phi_{i[1]}$") and ax.set_zlabel(rf"$\phi_{i[2]}$")
    cbar = plt.colorbar(sc)
    cbar.set_label("Inflaton Potential")
    plt.title("3-Dimensional Inflaton Plot")
    
    return



def bowlPlot2d(s, V0, ppr=10, domain=(-1,1), proj_d=(1,3)): 
    
    bowl = forceMin(s, V0, 2)
    vL, newphiList = len(bowl.vStuff), np.linspace(s*domain[0],s*domain[1],ppr)
    newphiList = np.transpose(list(itertools.product(newphiList, repeat=2)))
    
    for i in range(ppr**2): sampleVData(bowl, newphiList[:,i], include=True)
    
    if 1 in proj_d:
        fig1, ax1 = plt.subplots()
        [ax1.plot(newphiList[1, ppr*i:ppr*(i+1)], bowl.vStuff[vL + ppr*i:vL + ppr*(i+1)]) for i in range(ppr)]
        ax1.set_xlabel(r"$\phi_1$") and ax1.set_ylabel("V") and plt.show()
    
    if 2 in proj_d:
        fig2, ax2 = plt.subplots()
        sc = ax2.scatter(newphiList[0], newphiList[1], c=bowl.vStuff[vL:], cmap='inferno') 
        plt.colorbar(sc) and ax2.set_xlabel(r"$\phi_0$") and ax2.set_ylabel(r"$\phi_1$") and plt.show()
    
    if 3 in proj_d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(newphiList[0], newphiList[1], bowl.vStuff[vL:], c=bowl.vStuff[vL:], cmap='inferno')
        plt.colorbar(sc) and ax.set_xlabel("$\phi_0$") and ax.set_ylabel("$\phi_1$") and ax.set_zlabel("V") and plt.show()
    
    return bowl



def bowlPlot1d(s, V0, points=100, domain=(-1,1)):
    
    bowl = forceMin(s, V0, 1)
    vL, newphiList = len(bowl.vStuff), np.array([np.linspace(s*domain[0],s*domain[1],points)])
    
    for i in range(points): sampleVData(bowl, newphiList[:,i], include=True)
    
    plt.plot(newphiList[0], bowl.vStuff[vL:]) and plt.xlabel(r"$\phi_0$") and plt.ylabel("V")
    
    return bowl



def vcovPlot(b, xrange=(None,None), yrange=(None,None)):
    
    vcov, infoVec = b.vcov, b.infoVec
    vIndices = np.where(infoVec[1]==-1)[0]
    vvLocations = np.array(list(itertools.product(vIndices, repeat=2)))
    vcovV = vcov[vvLocations[:,0], vvLocations[:,1]].reshape(len(vIndices),len(vIndices))
    plt.matshow(vcovV[xrange[0]:xrange[1], yrange[0]:yrange[1]]) and plt.colorbar()
    
    return vcovV



def manySnakes(b, d=0, fps=20, name=None):
    
    fig, ax = plt.subplots()
    
    def animate(i):
        ax.clear()
        ax.set_xlabel('# of e-Folds')
        ylabel = 'Inflaton Components' if d==0 else 'Inflaton Component Velocities'
        ax.set_ylabel(ylabel)
        if d==0: [ax.plot(b.nE[:i], b.phi[dim,:i]) for dim in range(b.N)]
        if d==1: [ax.plot(b.nE[:i], b.dphi[dim,:i]) for dim in range(b.N)]
    
    anim = ani.FuncAnimation(fig, animate, frames=len(b.nE))
    writer = ani.PillowWriter(fps=fps)
    if name is None: name = hex(id(anim)) if b.name is None else b.name
    anim.save(f'manySnakes_{name}.gif', writer=writer)
    
    return



###############################################################################
## -----------------------------DATA COLLECTORS----------------------------- ##
###############################################################################



def explore3(testname='test', sVals=(30,), V0Vals=(5e-9,), NVals=(2,), from0_lbs=(0.5,), from0_ubs=(0.8,), sims=5):
    
    today = date.today().strftime("%m-%d-%y")
    dirname = f"{testname}_{today}"
    path = os.path.join(os.getcwd(), dirname)
    os.mkdir(path)
    
    combos = list(itertools.product(sVals, V0Vals, NVals))
    combos = tackBounds(combos, from0_lbs, from0_ubs)
    
    for combo in combos:
        
        nesteddirname = f"params_{combo[0]}_{combo[1]}_{int(combo[2])}_{combo[3]}_{combo[4]}"
        nestedpath = os.path.join(path, nesteddirname)
        os.mkdir(nestedpath)
        allstats = []
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(interior, combo) for sim in range(sims)]
            
            for f in concurrent.futures.as_completed(results):
                name, traj, stats, state = f.result()
                
                if traj is not None: traj.to_csv(os.path.join(nestedpath, f"sim{name}_trajectory.csv"), index=False)
                if stats is not None: allstats = stats if len(allstats)==0 else pd.concat((allstats, stats))
                if state is not None: np.save(os.path.join(nestedpath, f"sim{name}_state"), state)
                
        if len(allstats)>0: allstats.to_csv(os.path.join(nestedpath, "major_quantities.csv"), index=False)
    
    shutil.make_archive(dirname, 'zip', path)
    
    return



def interior(combo):
    
    name, stats, traj, state = None, None, None, None
    try:
        b = bushwhack(s = combo[0], V0 = combo[1], N = int(combo[2]), from0_bounds = (combo[3],combo[4]), fails_coef = 100, annotate=False)
        name = b.name
        try: stats = b.stats_df()
        except: pass
        try: traj = b.traj_df()
        except: pass
        try: state = b.state
        except: pass
    except Exception as e:
        print(f'Critical failure: {repr(e)}')
    
    return name, traj, stats, state



def tackBounds(combos, from0_lbs, from0_ubs):
    
    fullcombos = []
    for i in range(len(from0_lbs)): fullcombos = combos if i==0 else np.concatenate((fullcombos, combos))
    fullcombos = np.concatenate((fullcombos, np.zeros((len(fullcombos),2))), axis=1)
    
    for i in range(len(from0_lbs)):
        fullcombos[i*len(combos):(i+1)*len(combos),-2] = np.repeat(from0_lbs[i], len(combos))
        fullcombos[i*len(combos):(i+1)*len(combos),-1] = np.repeat(from0_ubs[i], len(combos))
        
    return fullcombos



## Small tests.
"""
if __name__=='__main__':
    
    explore3('minitest', from0_lbs=[0.3], from0_ubs=[0.4], sims=10)
"""

## Quark tests.
#"""
if __name__=='__main__':
    NVals = (1,2,3)
    from0_lbs = np.append(np.arange(3,13)/10, [2.0])
    from0_ubs = np.append(np.arange(4,14)/10, [2.1])
    explore3('megatest', NVals=NVals, from0_lbs=from0_lbs, from0_ubs=from0_ubs, sims=1000)
#"""







