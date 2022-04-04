# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 14:48:05 2021

@author: Alberto Castrignanò, Stefano Rainò
"""

import sys
import numpy
import scipy.linalg
import scipy.special
import scipy.optimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

numpy.seterr(divide='ignore', invalid='ignore')

def vrow(arr):
    return arr.reshape((1,arr.size))

def vcol(v):
    return v.reshape((v.size, 1))

''' used '''
def load(fname): 
    DList = []
    labelsList = []
    hLabels = {
        '0-class': 0,
        '1-class': 1,
        }
    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:11]
                attrs = vcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                if (name == '0'):
                    label = hLabels['0-class']
                    DList.append(attrs)
                    labelsList.append(label)
                elif (name == '1'):
                    label = hLabels['1-class']
                    DList.append(attrs)
                    labelsList.append(label)
            except:
                pass
    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def plot_hist (D, L):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    hMeasures = {
        0: 'Fixed acidity',
        1: 'Volatile acidity',
        2: 'Critic acid',
        3: 'Residual sugar',
        4: 'Chlorides',
        5: 'Free sulfur dioxide',
        6: 'Total sulfur dioxide',
        7: 'Density',
        8: 'pH',
        9: 'Sulphates',
        10: 'Alcohol'
        }
    for dIdx in range(11):
        plt.figure()
        plt.xlabel(hMeasures[dIdx])
        plt.hist(D0[dIdx, :], bins = 30, density = True, alpha = 0.4, label = '0-class')
        plt.hist(D1[dIdx, :], bins = 30, density = True, alpha = 0.4, label = '1-class')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./hist/hist_%d.png' % dIdx)
    plt.show()
 
''' used '''
def plot_scatter (D, L, version):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    hMeasures = {
        0: 'Fixed acidity',
        1: 'Volatile acidity',
        2: 'Critic acid',
        3: 'Residual sugar',
        4: 'Chlorides',
        5: 'Free sulfur dioxide',
        6: 'Total sulfur dioxide',
        7: 'Density',
        8: 'pH',
        9: 'Sulphates',
        10: 'Alcohol'
        }
    
    for dIdx1 in range(11):
        for dIdx2 in range(11):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hMeasures[dIdx1])
            plt.ylabel(hMeasures[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = '0-class')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = '1-class')
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('./scatter/scatter_'+str(version)+'_%d_%d_.png' % (dIdx1, dIdx2))
        plt.show()

''' used '''
def compute_PCA (D, L, m):
    #compute mean and the broadcasting center data
    mu = vcol(D.mean(1))
    DC = D - mu
    #PCA  analysis
    C = numpy.dot(DC, DC.T) / float(D.shape[1])
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    DP = numpy.dot(P.T, D)
    return DP, L

def compute_LDA (D, L, m):
    mu = vcol(D.mean(1))
    SB = 0
    for i in range(2):
        Di = D[:, L==i]
        mui = vcol(Di.mean(1))
        nci = Di.shape[1]
        SB += nci * (numpy.dot(mui - mu, (mui - mu).T))
    SB = SB / float(D.shape[1])
    SW = 0
    for i in range(2):
        Di = D[:, L==i]
        nci = vcol(Di.mean(1))
        mui = vcol(Di.mean(1))
        SWc = numpy.dot(Di - mui, (Di - mui).T) / nci
        SW += SWc * nci
    SW = SW / D.shape[1]
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:m]
    U, s, _ = numpy.linalg.svd(SW)
    P1 = numpy.dot(numpy.dot(U, numpy.diag(1.0/(s**0.5))), U.T)
    SBT = numpy.dot(numpy.dot(P1, SB), P1.T)
    s, U = numpy.linalg.eigh(SBT)
    P2 = U[:, ::-1][:, 0:m]
    DL = numpy.dot(numpy.dot(P2.T, P1), D)
    return DL, L

''' used '''
def compute_means (D, L):
    list0 = []
    list1 = []
    for i in range(L.shape[0]):
        if (L[i] == 0):
            list0.append(D[:, i])
        if (L[i] == 1):
            list1.append(D[:, i])   
    DT0 = numpy.array(list0);
    DT1 = numpy.array(list1);
    mu0 = DT0.mean(0)
    mu1 = DT1.mean(0)
    list = []
    for i in range(DT0.shape[1]):
        list.append(DT0[:, i])
    DT0 = numpy.array(list).reshape(DT0.shape[1], DT0.shape[0])
    list = []
    for i in range(DT1.shape[1]):
        list.append(DT1[:, i])
    DT1 = numpy.array(list).reshape(DT1.shape[1], DT1.shape[0])
    return vcol(mu0), vcol(mu1), DT0, DT1

''' used '''
def compute_covariance (DT, mu, version):
    DC = DT - mu
    C = numpy.dot(DC, DC.T) / float(DT.shape[1])
    if (version == 'nb'):
        return C*numpy.eye(C.shape[0])
    else: 
        return C

''' used '''
def GAU_ND_logpdf (XGAU, mu, C):
    first = (-mu.shape[0]/2)*numpy.log(2*numpy.pi)
    second = (-0.5)*numpy.linalg.slogdet(C)[1]
    third = (-0.5)*((XGAU-mu).T).dot(numpy.linalg.inv(C)).dot((XGAU-mu))
    return numpy.diag(first+second+third)

''' used '''
def compute_loglikelihoodsMatrix (DTR, LTR, DTE, LTE, version):
    mu0, mu1, DT0, DT1 = compute_means(DTR, LTR)
    C0 = compute_covariance(DT0, mu0, version)
    C1 = compute_covariance(DT1, mu1, version)
    if (version == 'tied'):
        Cstar = ((DT0.shape[1]*C0)+(DT1.shape[1]*C1)) / (DTR.shape[1])
        C0 = Cstar;
        C1 = Cstar;
    list = []
    list.append(GAU_ND_logpdf(DTE, mu0, C0)) #Cstar
    list.append(GAU_ND_logpdf(DTE, mu1, C1)) #Cstar
    S = numpy.array(list)
    return S

''' used '''
def compute_loglikelihoodsRatios (DTR, LTR, DTE, LTE, version):
    S = compute_loglikelihoodsMatrix(DTR, LTR, DTE, LTE, version)
    J = S + vcol(numpy.array([numpy.log(1/2), numpy.log(1/2)])) # Compute joint probability
    ll = scipy.special.logsumexp(J, axis = 0) # Compute marginal likelihood log f(x)
    PL = J - ll # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    P = numpy.exp(PL)
    llr = numpy.zeros(S.shape[1])
    for i in range (llr.size):
        llr[i] = numpy.log(P[1][i] / P[0][i])
    return llr

def compute_error (DTR, LTR, DTE, LTE, version):
    S = compute_loglikelihoodsMatrix(DTR, LTR, DTE, LTE, version)
    SJoint = S + vcol(numpy.array([numpy.log(1/2), numpy.log(1/2)]))
    ll = scipy.special.logsumexp(SJoint, axis=0)
    SPost = SJoint - ll
    PL = numpy.exp(SPost)
    PL = numpy.argmax(PL, 0)
    nCorrectPred = (PL == LTE).sum()
    #SJoint = S/2
    #SPost = SJoint / SJoint.sum(0)
    #PL = SPost.argmax(0)
    #nCorrectPred = (PL == LTE).sum()
    if (version == 'loo'):
        acc = nCorrectPred
    else:
        acc = nCorrectPred / LTE.shape[0] #if we want the version without leave one out
    err = 1 - acc
    return err;
     
def compute_confMatrix (DTR, LTR, DTE, LTE, version):
    S = compute_loglikelihoodsMatrix(DTR, LTR, DTE, LTE, version)
    SJoint = S + vcol(numpy.array([numpy.log(1/2), numpy.log(1/2)]))
    ll = scipy.special.logsumexp(SJoint, axis=0)
    SPost = SJoint - ll
    PL = numpy.exp(SPost)
    PL = numpy.argmax(PL, 0)
    #SJoint = S/2
    #SPost = SJoint / SJoint.sum(0)
    #PL = SPost.argmax(0)  
    confMatrix = numpy.zeros((2,2))
    for i in range(PL.size):
        confMatrix[PL[i]][LTE[i]] += 1;
    return confMatrix

''' used '''
def compute_confMatrix_bayes (llr, labels, pi, Cfn, Cfp):
    #C = numpy.array([0, Cfn, Cfp, 0]).reshape(2,2)
    PCP = numpy.array([1-pi, pi])
    Cstar = numpy.zeros(llr.size, dtype=numpy.int32)
    t = - numpy.log( (PCP[1]*Cfn) / ((1-PCP[1])*Cfp) )
    for i in range(Cstar.size):
        if (llr[i] > t):
            Cstar[i] = 1
        else:
            Cstar[i] = 0      
    confMatrix = numpy.zeros((2,2));
    for i in range(Cstar.size):
        confMatrix[Cstar[i]][labels[i]] += 1;
    return confMatrix

''' used '''
def compute_Bayes_risk (llr, labels, pi, Cfn, Cfp):
    M = compute_confMatrix_bayes(llr, labels, pi, Cfn, Cfp)
    FNR = M[0, 1] / M[:, 1].sum()
    FPR = M[1, 0] / M[:, 0].sum()
    DCFu = (pi*Cfn*FNR) + ((1-pi)*Cfp*FPR)
    return DCFu

''' used '''
def compute_Bayes_risk_normalized (llr, labels, pi, Cfn, Cfp):
    DCFu = compute_Bayes_risk(llr, labels, pi, Cfn, Cfp)
    Bdummy = min(pi*Cfn, (1-pi)*Cfp)
    return DCFu / Bdummy

''' used '''
def compute_Bayes_risk_minimal (llr, labels, pi, Cfn, Cfp, version, roc=0):
    thresholds = numpy.concatenate(([sys.float_info.min], numpy.sort(llr), [sys.float_info.max]))
    list = []
    x = []
    y = []
    for t in thresholds:
        Cstar = numpy.zeros(llr.size, dtype=numpy.int32)
        for i in range(Cstar.size):
            if (llr[i] > t):
                Cstar[i] = 1
            else:
                Cstar[i] = 0
        M = numpy.zeros((2,2));
        for i in range(Cstar.size):
            M[Cstar[i]][labels[i]] += 1;
        FNR = M[0, 1] / M[:, 1].sum()
        FPR = M[1, 0] / M[:, 0].sum()
        x.append(FPR)
        y.append(1-FNR)
        DCFu = (pi*Cfn*FNR) + ((1-pi)*Cfp*FPR)
        Bdummy = min(pi*Cfn, (1-pi)*Cfp)
        list.append(DCFu / Bdummy)
    return min(list)    

''' used '''
def compute_optimalDecisions (llr, labels, version):
    #Binary task: optimal decisions
    confMatrix_bayes = []
    confMatrix_bayes.append(compute_confMatrix_bayes (llr, labels, 0.5, 1, 1))
    confMatrix_bayes.append(compute_confMatrix_bayes (llr, labels, 0.1, 1, 1))
    confMatrix_bayes.append(compute_confMatrix_bayes (llr, labels, 0.9, 1, 1))
    #Binary task evaluation
    DCFu = []
    DCFu.append(compute_Bayes_risk (llr, labels, 0.5, 1, 1))
    DCFu.append(compute_Bayes_risk (llr, labels, 0.1, 1, 1))
    DCFu.append(compute_Bayes_risk (llr, labels, 0.9, 1, 1))
    #Binary task evaluation normalized
    DCF = []
    DCF.append(compute_Bayes_risk_normalized (llr, labels, 0.5, 1, 1))
    DCF.append(compute_Bayes_risk_normalized (llr, labels, 0.1, 1, 1))
    DCF.append(compute_Bayes_risk_normalized (llr, labels, 0.9, 1, 1))
    #Minimum Detection Costs
    minDCF = []
    minDCF.append(compute_Bayes_risk_minimal (llr, labels, 0.5, 1, 1, version)) 
    minDCF.append(compute_Bayes_risk_minimal (llr, labels, 0.1, 1, 1, version)) 
    minDCF.append(compute_Bayes_risk_minimal (llr, labels, 0.9, 1, 1, version)) 
    return confMatrix_bayes, DCFu, DCF, minDCF

''' used '''
def logreg_obj (v, DTR, LTR, l, pit=0):
    w, b = v[0:-1], v[-1]
    sum1 = 0
    sum0 = 0
    sum = 0
    for i in range (LTR.size):
        if (pit > 0):
            if (LTR[i] == 1):
                sum1 += numpy.log1p(numpy.exp(-numpy.dot(w.T, DTR[:, i]) - b))
            else:
                sum0 += numpy.log1p(numpy.exp(numpy.dot(w.T, DTR[:, i]) + b))
        else:
            sum += numpy.dot(LTR[i], numpy.log1p(numpy.exp(-numpy.dot(w.T, DTR[:, i]) - b))) + numpy.dot(1-LTR[i], numpy.log1p(numpy.exp(numpy.dot(w.T, DTR[:, i]) + b)))
    if (pit > 0):
        J = (numpy.dot(l/2, numpy.dot(abs(w), abs(w))) + numpy.dot(pit/LTR[LTR==1].size, sum1) + numpy.dot((1-pit)/LTR[LTR==0].size, sum0))
    else:
        J = (numpy.dot(l/2, numpy.dot(abs(w), abs(w))) + numpy.dot(1/LTR.size, sum))
    return J

''' used '''
def compute_llr_logreg (v, DTE, LTE):
    w, b = v[0:-1], v[-1]
    S = []
    for i in range(LTE.size):
        S.append(numpy.dot(w.T, DTE[:, i]) + b)
    S = numpy.array(S)
    return S

''' used '''
def bayes_Plots(llr, labels, version, lam):
    #Bayes plots
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    piTilde = numpy.zeros(effPriorLogOdds.size)
    for i in range(effPriorLogOdds.size):
        piTilde[i] = (1 / (1 + numpy.exp(-effPriorLogOdds[i])))  
    mindcf = numpy.zeros(piTilde.size)
    result = []
    for i in range(piTilde.size):
        if (piTilde[i] > 0.1 and len(result) == 0):
            mindcf[i] = compute_Bayes_risk_minimal(llr, labels, 0.1, 1, 1, version)
            result.append(mindcf[i])
        else:
            mindcf[i] = compute_Bayes_risk_minimal(llr, labels, piTilde[i], 1, 1, version)
        if (piTilde[i] == 0.5 and len(result) == 1):
            result.append(mindcf[i])
        if (piTilde[i] > 0.9 and len(result) == 2):
            mindcf[i] = compute_Bayes_risk_minimal(llr, labels, 0.9, 1, 1, version)
            result.append(mindcf[i])
        else:
            mindcf[i] = compute_Bayes_risk_minimal(llr, labels, piTilde[i], 1, 1, version)
         
    plt.plot(effPriorLogOdds, mindcf, label='λ='+str(numpy.format_float_scientific(lam, precision=1)))
    return result

def GAU_pdf (XGAU, mu, var):
    N = []
    const = (2*numpy.pi*var)**(-0.5);
    for x in XGAU:
        exp = ((x-mu)**2)/(2*var);
        N.append(const/numpy.exp(exp));
    return numpy.array(N);

''' used '''
def compute_Gaussianize (DTR, LTR):
    f = []
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            x = (DTR[i, :] < DTR[i][j]).astype(int).sum() + 1
            f.append(x)
    d = numpy.array(f).reshape(DTR.shape)/(DTR.shape[1] + 2)
    return scipy.stats.norm.ppf(d), LTR

''' used '''
def split_db_2to1(D, L):
    nTrain = int(D.shape[1]*80/100) #number of data training
    DTR = D[:, 0:nTrain] #data training
    DTV = D[:, nTrain::] #data test
    LTR = L[0:nTrain]   #label training
    LTV = L[nTrain::]    #label test
    return (DTR, LTR), (DTV, LTV)

''' used '''
def plot_heatmap (DTR, LTR):
    hm_Whole, hm_0class, hm_1class = compute_Pearson(DTR, LTR)
    plt.imshow(hm_Whole, cmap='gist_yarg')
    plt.savefig('./heatmap/heatmap_whole.png')
    plt.figure()
    plt.imshow(hm_0class, cmap='Blues')
    plt.savefig('./heatmap/heatmap_0class.png')
    plt.figure()
    plt.imshow(hm_1class, cmap='Greens')
    plt.savefig('./heatmap/heatmap_1class.png')
    
''' used '''
def compute_Results (DTR, LTR, DTV=0, LTV=0, gauss=False, m=0, kfold=False):
    t = PrettyTable(['type', 'pi=0.5', 'pi=0.1', 'pi=0.9'])
    if (kfold == False):
        llr_mvg = compute_loglikelihoodsRatios(DTR, LTR, DTV, LTV, 'mvg')
        _, _, _, minDCFmvg = compute_optimalDecisions (llr_mvg, LTV, 'mvg')
        llr_nb = compute_loglikelihoodsRatios(DTR, LTR, DTV, LTV, 'nb')
        _, _, _, minDCFnb = compute_optimalDecisions (llr_nb, LTV, 'nb')
        llr_tied = compute_loglikelihoodsRatios(DTR, LTR, DTV, LTV, 'tied')
        _, _, _, minDCFtied = compute_optimalDecisions (llr_tied, LTV, 'tied')
    else:
        llr_mvg = compute_loglikelihoodsRatios_Kfold(DTR, LTR, 'mvg')
        _, _, _, minDCFmvg = compute_optimalDecisions (llr_mvg, LTR, 'mvg')
        llr_nb = compute_loglikelihoodsRatios_Kfold(DTR, LTR, 'nb')
        _, _, _, minDCFnb = compute_optimalDecisions (llr_nb, LTR, 'nb')
        llr_tied = compute_loglikelihoodsRatios_Kfold(DTR, LTR, 'tied')
        _, _, _, minDCFtied = compute_optimalDecisions (llr_tied, LTR, 'tied')
    t.add_row(['Full-Cov', minDCFmvg[0], minDCFmvg[1], minDCFmvg[2]])
    t.add_row(['Diag-Cov', minDCFnb[0], minDCFnb[1], minDCFnb[2]])
    t.add_row(['Tied-Full-Cov', minDCFtied[0], minDCFtied[1], minDCFtied[2]])
    
    if (kfold == False):
        if (gauss == True):
            if (m > 0):
                t.title = 'Single fold - Gaussianized features - PCA (m='+str(m)+')'
            else:
                t.title = 'Single fold - Gaussianized features - no PCA'
        else:
            if (m > 0):
                t.title = 'Single fold - Raw features (no Gaussianization) - PCA (m='+str(m)+')'
            else:
                t.title = 'Single fold - Raw features (no Gaussianization) - no PCA'
    else:
        if (gauss == True):
            if (m > 0):
                t.title = '5-fold - Gaussianized features - PCA (m='+str(m)+')'
            else:
                t.title = '5-fold - Gaussianized features - no PCA'
        else:
            if (m > 0):
                t.title = '5-fold - Raw features (no Gaussianization) - PCA (m='+str(m)+')'
            else:
                t.title = '5-fold - Raw features (no Gaussianization) - no PCA'
    print(t)
    
    return llr_mvg, llr_nb, llr_tied

''' used '''
def split_db_kfold (DTR_or, LTR_or, fold):
    size = int(DTR_or.shape[1]/5) + 1
    init = size*fold
    final = size*(fold+1)
    DTR = numpy.concatenate((DTR_or[:, 0:init], DTR_or[:, final::]), axis=1)
    LTR = numpy.concatenate((LTR_or[0:init], LTR_or[final::]), axis=0)
    DTV = DTR_or[:, init:final]
    LTV = LTR_or[init:final]
    return (DTR, LTR), (DTV, LTV)
    
''' used '''
def compute_loglikelihoodsRatios_Kfold (DTR_or, LTR_or, version):
    for i in range(5):
        (DTR, LTR), (DTV, LTV) = split_db_kfold(DTR_or, LTR_or, i)
        if (i == 0):
            llr = compute_loglikelihoodsRatios(DTR, LTR, DTV, LTV, version)
        else:
            llr = numpy.concatenate((llr, compute_loglikelihoodsRatios(DTR, LTR, DTV, LTV, version)), axis=0)
    return llr
    

def compute_Pearson (DTR, LTR):
    pearson_Whole = numpy.corrcoef(DTR)
    pearson_0class = numpy.corrcoef(DTR[:, LTR==0])
    pearson_1class = numpy.corrcoef(DTR[:, LTR==1])
    return pearson_Whole, pearson_0class, pearson_1class

''' used '''
def randomize (DTR, LTR):
    numpy.random.seed(0)
    idx = numpy.random.permutation(DTR.shape[1])
    DTR = DTR[:, idx]
    LTR = LTR[idx]
    return DTR, LTR

''' used '''
def compute_Hext (DTR, LTR, K):
    new_row = numpy.array(numpy.full((1, DTR.shape[1]), K))
    Dext = numpy.append(DTR, new_row).reshape(DTR.shape[0]+1, DTR.shape[1])
    
    Hext = numpy.zeros(Dext.shape[1]**2).reshape(Dext.shape[1], Dext.shape[1])
    for i in range(Dext.shape[1]):
        if (LTR[i] == 1):
            zi = 1
        else:
            zi = -1
        for j in range(Dext.shape[1]):
            if (LTR[j] == 1):
                zj = 1
            else:
                zj = -1
            Hext[i][j] = zi*zj*numpy.dot(Dext[:, i].T, Dext[:, j])
    return Hext

''' used '''
def compute_LBFGSB (alpha, Hext):
    return ((1/2)*numpy.dot(numpy.dot(alpha.T, Hext), alpha)) - numpy.dot(alpha.T, numpy.ones(alpha.size)) , (numpy.dot(Hext, alpha) - numpy.ones(alpha.size)).reshape(alpha.size)

''' used '''    
def compute_Dual_and_Primal_loss (LTR, DTR, K, C, Hext, pit=0):
    new_row = numpy.array(numpy.full((1, DTR.shape[1]), K))
    Dext = numpy.append(DTR, new_row).reshape(DTR.shape[0]+1, DTR.shape[1])
    
    bounds = []
    for i in range(LTR.size):
        if (pit > 0):
            if (LTR[i] == 1):
                bounds.append((0, (C*pit)/(LTR.size/LTR[LTR==1].size)))
            else:
                bounds.append((0, (C*(1-pit))/(LTR.size/LTR[LTR==0].size)))
        else:
            bounds.append((0, C))
    alpha, L, d = scipy.optimize.fmin_l_bfgs_b(compute_LBFGSB, numpy.zeros(LTR.size), args=(Hext,), bounds=bounds, factr=1.0)
    
    w_star_ext = 0
    for i in range(alpha.size):
        if (LTR[i] == 1):
            w_star_ext += numpy.dot(alpha[i], Dext[:, i])
        else:
            w_star_ext -= numpy.dot(alpha[i], Dext[:, i])
            
    sum = 0
    S = numpy.dot(w_star_ext.T, Dext)
    for i in range(LTR.size):
        if (LTR[i] == 1):
            sum += max(0, 1-S[i])
        else:
            sum += max(0, 1+S[i])
    J = (1/2)*((w_star_ext*w_star_ext).sum()) + C*sum
    return L, J, w_star_ext

''' used '''
def compute_llr_svm (DTE, LTE, w_star_ext, K):
    new_row = numpy.array(numpy.full((1, DTE.shape[1]), K))
    Dext_t = numpy.append(DTE, new_row).reshape(DTE.shape[0]+1, DTE.shape[1])
    
    S = numpy.dot(w_star_ext.T, Dext_t)
    return numpy.array(S)

''' used '''       
def bayes_Plots_svm (llr, labels, version, K, C):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    piTilde = numpy.zeros(effPriorLogOdds.size)
    for i in range(effPriorLogOdds.size):
        piTilde[i] = (1 / (1 + numpy.exp(-effPriorLogOdds[i])))  
    mindcf = numpy.zeros(piTilde.size)
    result = []
    for i in range(piTilde.size):
        if (piTilde[i] > 0.1 and len(result) == 0):
            mindcf[i] = compute_Bayes_risk_minimal(llr, labels, 0.1, 1, 1, version)
            result.append(mindcf[i])
        else:
            mindcf[i] = compute_Bayes_risk_minimal(llr, labels, piTilde[i], 1, 1, version)
        if (piTilde[i] == 0.5 and len(result) == 1):
            result.append(mindcf[i])
        if (piTilde[i] > 0.9 and len(result) == 2):
            mindcf[i] = compute_Bayes_risk_minimal(llr, labels, 0.9, 1, 1, version)
            result.append(mindcf[i])
        else:
            mindcf[i] = compute_Bayes_risk_minimal(llr, labels, piTilde[i], 1, 1, version)
                
    plt.plot(effPriorLogOdds, mindcf, label='K='+str(K)+' C='+str(C))
    return result

''' used '''
def compute_means_gmm(DT, LT):
    list0 = numpy.array(DT[:, LT==0])
    list1 = numpy.array(DT[:, LT==1])
    
    mu0 = list0.mean(1)
    mu1 = list1.mean(1)
   
    return numpy.array([vcol(mu0), vcol(mu1)])

''' used '''
def compute_GMM (X, L, means, version):
    for i in range(means.shape[0]):
        C = compute_covariance(numpy.array(X[:, L==i]), means[i], version)
        U, s, _ = numpy.linalg.svd(C)
        s[s<0.01] = 0.01
        covNew = numpy.dot(U, vcol(s)*U.T)
        if (i == 0):
            gmm = numpy.array([[1.0, means[i], covNew]], dtype=object)
        else:
            gmm = numpy.vstack((gmm, numpy.array([[1.0, means[i], covNew]], dtype=object)))
    return gmm

''' used '''
def compute_S (X, gmm):
    list = []
    for g in range(gmm.shape[0]):
        list.append(GAU_ND_logpdf(X, gmm[g][1], gmm[g][2]))
    
    S = numpy.array(list)
    for g in range(gmm.shape[0]):
        S[g, :] += numpy.log(gmm[g][0])
    return S

''' used '''
def logpdf_GMM (X, gmm, version=0):
    S = compute_S(X, gmm)
    logdens = scipy.special.logsumexp(S, axis=0)
    if (version == 'P'):
        return S - logdens.reshape(1, logdens.size)
    else:
        return logdens.reshape(1, logdens.size)

''' used '''
def compute_EM (X, gmm, version, delta):
    P = numpy.exp(logpdf_GMM(X, gmm, 'P'))
    
    Z = numpy.zeros(gmm.shape[0])
    F = [numpy.zeros(X.shape[0])]
    S = [numpy.zeros((X.shape[0], X.shape[0]))]
    
    for g in range(gmm.shape[0]):
        F.append(numpy.zeros(X.shape[0]))
        S.append(numpy.zeros((X.shape[0], X.shape[0])))
    
    ll1 = logpdf_GMM(X, gmm)
    #print('ll1: ', ll1.mean())
    for g in range(gmm.shape[0]):
        for i in range(X.shape[1]):
            Z[g] += P[g][i]
            F[g] += numpy.dot(P[g][i], X[:, i])
            S[g] += numpy.dot(P[g][i], numpy.dot(vcol(X[:, i]), vcol(X[:, i]).T))
            
    for g in range(gmm.shape[0]):
        gmm[g][1] = vcol(numpy.array(F[g]/Z[g]))
        
        cov = (S[g]/Z[g]) - numpy.dot(gmm[g][1], gmm[g][1].T)
        
        #Part 5: Diagonal and tied-covariance GMMs
        if (version == 'nb'):
            cov = cov * numpy.eye(cov.shape[0])
        elif (version == 'tied'):
            covNew = 0
            for g in range(gmm.shape[0]):
                covNew += (1/X.shape[1])*(numpy.dot(Z[g], cov))
            cov = covNew
         
        #Part 4: Constraining the eigenvalues of the covariance matrices
        U, s, _ = numpy.linalg.svd(cov)
        s[s<0.01] = 0.01
        covNew = numpy.dot(U, vcol(s)*U.T)
        
        gmm[g][2] = covNew
        gmm[g][0] = Z[g]/(Z.sum())
    ll2 = logpdf_GMM(X, gmm)
    #print('ll2: ', ll2.mean())
    
    if ((ll2.mean() - ll1.mean()) < delta):
        return ll2
    else:
        ll2 = compute_EM(X, gmm, version, delta)
    return ll2

''' used '''
def compute_LBG (X, gmm, alpha, components, version, delta):
   if (gmm.shape[0] == 1):
       ll = compute_EM(X, gmm, version, delta)
   for g in range(gmm.shape[0]):
       U, s, Vh = numpy.linalg.svd(gmm[g][2])
       d = U[:, 0:1] * s[0]**0.5 * alpha
            
       if (g == 0):
           GMMnew = numpy.array([[gmm[g][0]/2, gmm[g][1]+d, gmm[g][2]]], dtype=object)
       else:
           GMMnew = numpy.vstack((GMMnew, numpy.array([[gmm[g][0]/2, gmm[g][1]+d, gmm[g][2]]], dtype=object)))
       GMMnew = numpy.vstack((GMMnew, numpy.array([[gmm[g][0]/2, gmm[g][1]-d, gmm[g][2]]], dtype=object)))
            
   ll = compute_EM(X, GMMnew, version, delta)
   if (GMMnew.shape[0] == components):
       return GMMnew, ll
   else:
       GMMnew, ll = compute_LBG(X, GMMnew, alpha, components, version, delta)
   return GMMnew, ll

''' used '''
def compute_err_gmm (X, gmm, components, classes, L):
    prob = numpy.zeros((classes, X.shape[1]));
    for x in range(X.shape[1]):
        for c in range(classes):
            score = 0
            for k in range(components):
                #score += numpy.log(gmm[c][k][0]) + logpdf_GAU_ND(X[:,x:x+1], gmm[c][k][1] , gmm[c][k][2])
                score += gmm[c][k][0]*numpy.exp(GAU_ND_logpdf(X[:,x:x+1], gmm[c][k][1] , gmm[c][k][2]))
            prob[c][x] = score*(1/2)
    PL = prob / prob.sum(axis=0).reshape((1,X.shape[1]))
    llr = log_likelihood_posteriors(PL)
    P = numpy.argmax(PL, 0)
    nCorrectPred = (P == L).sum() 
    acc = nCorrectPred / L.shape[0] 
    err = 1 - acc 
    return err, llr;

''' used '''
def log_likelihood_posteriors(posteriors):
    return numpy.log(posteriors[1] / posteriors[0])

def log_likelihood_log_posteriors(posteriors):
    return posteriors[1] - posteriors[0]

''' used '''
def extended_data_quad_lg(DTR_g):
    for i in range(DTR_g.shape[1]):
        vec = numpy.dot(vcol(DTR_g[:, i]), vrow(DTR_g[:, i].T))
        for j in range(vec.shape[1]):
            if (j == 0):
                vecnew = vec[:, j]
            else:
                vecnew = numpy.hstack((vecnew, vec[:, j]))
        vecfinal = vcol(numpy.hstack((vecnew, DTR_g[:, i])))
        if (i == 0):
            vectotal = vecfinal
        else:
            vectotal = numpy.hstack((vectotal, vecfinal))
    return vectotal

''' used '''
def compute_dcf_plots (llr, labels):
    #Bayes plots
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    piTilde = numpy.zeros(effPriorLogOdds.size)
    for i in range(effPriorLogOdds.size):
        piTilde[i] = (1 / (1 + numpy.exp(-effPriorLogOdds[i])))
    mindcf = numpy.zeros(piTilde.size)
    dcf = numpy.zeros(piTilde.size)
    result = []
    for i in range(piTilde.size):
        if(piTilde[i]>=0.09 and len(result)==0):
            mindcf[i] = compute_Bayes_risk_minimal(llr, labels, 0.1, 1, 1, 'mvg')
            dcf[i] = compute_Bayes_risk_normalized(llr, labels, 0.1, 1, 1)
            result.append(numpy.array([dcf[i], mindcf[i]]))
        if(piTilde[i]>=0.5 and len(result)==1):
            mindcf[i] = compute_Bayes_risk_minimal(llr, labels, 0.5, 1, 1, 'mvg')
            dcf[i] = compute_Bayes_risk_normalized(llr, labels, 0.5, 1, 1)
            result.append(numpy.array([dcf[i], mindcf[i]]))
        if(piTilde[i]>=0.89 and len(result)==2):
            mindcf[i] = compute_Bayes_risk_minimal(llr, labels, 0.9, 1, 1, 'mvg')
            dcf[i] = compute_Bayes_risk_normalized(llr, labels, 0.9, 1, 1)
            result.append(numpy.array([dcf[i], mindcf[i]]))
        mindcf[i] = compute_Bayes_risk_minimal(llr, labels, piTilde[i], 1, 1, 'mvg')
        dcf[i] = compute_Bayes_risk_normalized(llr, labels, piTilde[i], 1, 1)
    return effPriorLogOdds, mindcf, dcf, result

''' used '''
def generate_roc_axis (llr, labels, pi, Cfn, Cfp):
    thresholds = numpy.concatenate(([sys.float_info.min], numpy.sort(llr), [sys.float_info.max]))
    list = []
    x = []
    y = []
    for t in thresholds:
        Cstar = numpy.zeros(llr.size, dtype=numpy.int32)
        for i in range(Cstar.size):
            if (llr[i] > t):
                Cstar[i] = 1
            else:
                Cstar[i] = 0
        M = numpy.zeros((2,2));
        for i in range(Cstar.size):
            M[Cstar[i]][labels[i]] += 1;
        FNR = M[0, 1] / M[:, 1].sum()
        FPR = M[1, 0] / M[:, 0].sum()
        x.append(FPR)
        y.append(1-FNR)
        DCFu = (pi*Cfn*FNR) + ((1-pi)*Cfp*FPR)
        Bdummy = min(pi*Cfn, (1-pi)*Cfp)
        list.append(DCFu / Bdummy)
    return x, y

''' used '''
def compute_Gaussianize_test (DTR, LTR, DTE, LTE):
    f = []
    for i in range(DTE.shape[0]):
        for j in range(DTE.shape[1]):
            x = (DTR[i, :] < DTE[i][j]).astype(int).sum() + 1
            f.append(x)
    d = numpy.array(f).reshape(DTE.shape)/(DTE.shape[1] + 2)
    return scipy.stats.norm.ppf(d), LTE

#if __name__ == '__main__':
    
    