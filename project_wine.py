# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 14:58:51 2021

@author: Alberto Castrignanò, Stefano Rainò
"""

import mlfunctions as mlf
import numpy
import matplotlib.pyplot as plt
import scipy.optimize

if __name__ == '__main__':

    numpy.seterr('ignore') 
    DTR, LTR = mlf.load('./data/Train.txt')
    DTE, LTE = mlf.load ('./data/Test.txt')
    DTR, LTR = mlf.randomize (DTR, LTR)
    
    mlf.plot_scatter(DTR, LTR, 'raw')
    
    ### SINGLE FOLD APPROACH ###
    ### MVG NO PCA GAUSSIANIZED###
    DTR_g, LTR_g = mlf.compute_Gaussianize(DTR, LTR)
    
    mlf.plot_scatter(DTR_g, LTR_g, 'gauss')
    mlf.plot_heatmap(DTR_g, LTR_g)
    
    (DTR_sf_g, LTR_sf_g), (DTV_sf_g, LTV_sf_g) = mlf.split_db_2to1(DTR_g, LTR_g)
    mlf.compute_Results(DTR_sf_g, LTR_sf_g, DTV_sf_g, LTV_sf_g, gauss=True) #only GAUSS
    ### PCA=10 GAUSSIANIZED ###
    DTR_p, LTR_p = mlf.compute_PCA(DTR, LTR, 10)
    DTR_p_g, LTR_p_g = mlf.compute_Gaussianize(DTR_p, LTR_p)
    (DTR_sf_p_g, LTR_sf_p_g), (DTV_sf_p_g, LTV_sf_p_g) = mlf.split_db_2to1(DTR_p_g, LTR_p_g)
    mlf.compute_Results(DTR_sf_p_g, LTR_sf_p_g, DTV_sf_p_g, LTV_sf_p_g, m=10, gauss=True) #PCA m= 10 + GAUSS
    ### PCA=9 GAUSSIANIZED ###
    DTR_p, LTR_p = mlf.compute_PCA(DTR, LTR, 9)
    DTR_p_g, LTR_p_g = mlf.compute_Gaussianize(DTR_p, LTR_p)
    (DTR_sf_p_g, LTR_sf_p_g), (DTV_sf_p_g, LTV_sf_p_g) = mlf.split_db_2to1(DTR_p_g, LTR_p_g)
    mlf.compute_Results(DTR_sf_p_g, LTR_sf_p_g, DTV_sf_p_g, LTV_sf_p_g, m=9, gauss=True) #PCA m = 9 + GAUSS
    ### MVG NO PCA RAW ###
    (DTR_sf, LTR_sf), (DTV_sf, LTV_sf) = mlf.split_db_2to1(DTR, LTR)
    mlf.compute_Results(DTR_sf, LTR_sf, DTV_sf, LTV_sf, gauss=False) #only RAW
    
    ### 5-FOLD APPROACH ###
    ### MVG NO PCA GAUSSIANIZED###
    DTR_g, LTR_g = mlf.compute_Gaussianize(DTR, LTR)
    llr_mvg, _, _ = mlf.compute_Results(DTR_g, LTR_g, gauss=True, kfold=True) #only GAUSS
    ''' save '''
    #numpy.save("out/llr_mvg_5fold_full_cov_gauss", llr_mvg)
    ### PCA=10 GAUSSIANIZED ###
    DTR_p, LTR_p = mlf.compute_PCA(DTR, LTR, 10)
    DTR_p_g, LTR_p_g = mlf.compute_Gaussianize(DTR_p, LTR_p)
    mlf.compute_Results(DTR_p_g, LTR_p_g, m=10, gauss=True, kfold=True) #PCA m = 10 + GAUSS
    ### PCA=9 GAUSSIANIZED ### 
    DTR_p, LTR_p = mlf.compute_PCA(DTR, LTR, 9)
    DTR_p_g, LTR_p_g = mlf.compute_Gaussianize(DTR_p, LTR_p)
    mlf.compute_Results(DTR_p_g, LTR_p_g, m=9, gauss=True, kfold=True) #PCA m = 9 + GAUSS
    ### MVG NO PCA RAW ###
    mlf.compute_Results(DTR, LTR, kfold=True) #only RAW
    
    
    ### LINEAR LOGREG 5-FOLD ###
    
    ### GAUSSIANIZED UNBALANCED ### l=0, l=10**-6, l=10**-3
    
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    x0 = numpy.zeros(DTR_g.shape[0]+1)
    lam = numpy.array([0, 10**-6, 10**-3])
    for l in lam:
        llr = 0
        for i in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR_g, LTR_g, i)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_kg, LTR_kg, l), approx_grad=True)
            if (i == 0):
                llr = mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)
            else:
                llr = numpy.concatenate((llr, mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)))     
        rr = mlf.bayes_Plots(llr, LTR_g, version='mvg', lam=l)
        print(rr)  
    plt.legend()
    plt.savefig('./bayesPlots/5fold_logreg_gauss.png')
    
    ### GAUSSIANIZED BALANCED pt=0.5 ### l=0, l=10**-6, l=10**-3
        
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    x0 = numpy.zeros(DTR_g.shape[0]+1)
    lam = numpy.array([0, 10**-6, 10**-3])
    for l in lam:
        llr = 0
        for i in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR_g, LTR_g, i)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_kg, LTR_kg, l, 0.5), approx_grad=True)
            if (i == 0):
                llr = mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)
            else:
                llr = numpy.concatenate((llr, mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)))     
        rr = mlf.bayes_Plots(llr, LTR_g, version='mvg', lam=l)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_logreg_gauss_balanced05.png')
    
    
    ### GAUSSIANIZED BALANCED pt=0.1 ### l=0, l=10**-6, l=10**-3
    
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    x0 = numpy.zeros(DTR_g.shape[0]+1)
    lam = numpy.array([0, 10**-6, 10**-3])
    for l in lam:
        llr = 0
        for i in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR_g, LTR_g, i)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_kg, LTR_kg, l, 0.1), approx_grad=True)
            if (i == 0):
                llr = mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)
            else:
                llr = numpy.concatenate((llr, mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)))     
        rr = mlf.bayes_Plots(llr, LTR_g, version='mvg', lam=l)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_logreg_gauss_balanced01.png')
        
    ### GAUSSIANIZED BALANCED pt=0.9 ### l=0, l=10**-6, l=10**-3     
        
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    x0 = numpy.zeros(DTR_g.shape[0]+1)
    lam = numpy.array([0, 10**-6, 10**-3])
    for l in lam:
        llr = 0
        for i in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR_g, LTR_g, i)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_kg, LTR_kg, l, 0.9), approx_grad=True)
            if (i == 0):
                llr = mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)
            else:
                llr = numpy.concatenate((llr, mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)))     
        rr = mlf.bayes_Plots(llr, LTR_g, version='mvg', lam=l)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_logreg_gauss_balanced09.png')
        
    ### RAW UNBALANCED l=10**-6 ###
    
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    x0 = numpy.zeros(DTR_g.shape[0]+1)
    lam = numpy.array([10**-6])
    for l in lam:
        llr = 0
        for i in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR, LTR, i)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_kg, LTR_kg, l), approx_grad=True)
            if (i == 0):
                llr = mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)
            else:
                llr = numpy.concatenate((llr, mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)))     
        rr = mlf.bayes_Plots(llr, LTR, version='mvg', lam=l)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_logreg_raw_unbalanced.png')
        
    ### RAW BALANCED pt=0.5 l=10**-6 ###
        
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    x0 = numpy.zeros(DTR_g.shape[0]+1)
    lam = numpy.array([10**-6])
    for l in lam:
        llr = 0
        for i in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR, LTR, i)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_kg, LTR_kg, l, 0.5), approx_grad=True)
            if (i == 0):
                llr = mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)
            else:
                llr = numpy.concatenate((llr, mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)))     
        rr = mlf.bayes_Plots(llr, LTR, version='mvg', lam=l)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_logreg_raw_balanced05.png')
        
        
    ### RAW BALANCED pt=0.1 l=10**-6 ###
    
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    x0 = numpy.zeros(DTR_g.shape[0]+1)
    lam = numpy.array([10**-6])
    for l in lam:
        llr = 0
        for i in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR, LTR, i)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_kg, LTR_kg, l, 0.1), approx_grad=True)
            if (i == 0):
                llr = mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)
            else:
                llr = numpy.concatenate((llr, mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)))     
        rr = mlf.bayes_Plots(llr, LTR, version='mvg', lam=l)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_logreg_raw_balanced01.png')
    
    ### RAW BALANCED pt=0.9 l=10**-6 ###

    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    x0 = numpy.zeros(DTR_g.shape[0]+1)
    lam = numpy.array([10**-6])
    for l in lam:
        llr = 0
        for i in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR, LTR, i)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_kg, LTR_kg, l, 0.9), approx_grad=True)
            if (i == 0):
                llr = mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)
            else:
                llr = numpy.concatenate((llr, mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)))     
        rr = mlf.bayes_Plots(llr, LTR, version='mvg', lam=l)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_logreg_raw_balanced09.png')
        
    ### QUADRATIC LOGREG 5-FOLD ###
    
    ### GAUSSIANIZED UNBALANCED ### l=0, l=10**-6, l=10**-3
    
    DTR_ext_g = mlf.extended_data_quad_lg(DTR_g)
    
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    x0 = numpy.zeros(DTR_ext_g.shape[0]+1)
    lam = numpy.array([0, 10**-6, 10**-3])
    for l in lam:
        llr = 0
        for i in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR_ext_g, LTR_g, i)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_kg, LTR_kg, l), approx_grad=True)
            if (i == 0):
                llr = mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)
            else:
                llr = numpy.concatenate((llr, mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)))     
        rr = mlf.bayes_Plots(llr, LTR_g, version='mvg', lam=l)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_quadlogreg_gauss_unbalanced.png')
    
    '''
    ###USED TO RETRIEVE 10**-6 case ONLY###
    print()
    print("--------------------------------------------")
    x0 = numpy.zeros(DTR_ext_g.shape[0]+1)
    lam = numpy.array([10**-6])
    for l in lam:
        llr = 0
        for i in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR_ext_g, LTR_g, i)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_kg, LTR_kg, l), approx_grad=True)
            if (i == 0):
                llr = mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)
            else:
                llr = numpy.concatenate((llr, mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)))     
        rr = mlf.bayes_Plots(llr, LTR_g, version='mvg', lam=l, gauss=True, kfold=True)
        print(rr)
    numpy.save("out/llr_quad_logreg_5fold_unb_10m6", llr)
    '''     
   
    
    ### GAUSSIANIZED BALANCED pt=0.5 ### l=0, l=10**-6, l=10**-3
    
    DTR_ext_g = mlf.extended_data_quad_lg(DTR_g)
    
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    x0 = numpy.zeros(DTR_ext_g.shape[0]+1)
    lam = numpy.array([0, 10**-6, 10**-3])
    for l in lam:
        llr = 0
        for i in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR_ext_g, LTR_g, i)
            x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_kg, LTR_kg, l, 0.5), approx_grad=True)
            if (i == 0):
                llr = mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)
            else:
                llr = numpy.concatenate((llr, mlf.compute_llr_logreg(x, DTV_kg, LTV_kg)))     
        rr = mlf.bayes_Plots(llr, LTR_g, version='mvg', lam=l)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_quadlogreg_gauss_balanced05.png')
    
    ### LINEAR SVM 5-FOLD ###
    
    ### GAUSSIANIZED UNBALANCED ### K=1, K=10 - C=0.1, C=1, C=10 ###
    
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    K = numpy.array([1, 10])
    C = numpy.array([0.1, 1.0, 10])
    for k in range(K.size):
        for c in range(C.size):  
            for i in range(5):
                (DTR_gk, LTR_gk), (DTV_gk, LTV_gk) = mlf.split_db_kfold(DTR_g, LTR_g, i)
                Hext = mlf.compute_Hext (DTR_gk, LTR_gk, K=K[k])
                L, J, w_star_ext = mlf.compute_Dual_and_Primal_loss (LTR_gk, DTR_gk, Hext=Hext, K=K[k], C=C[c])
                if (i == 0):
                    llr = mlf.compute_llr_svm(DTV_gk, LTV_gk, w_star_ext, K=K[k])
                else:
                    llr = numpy.concatenate((llr, mlf.compute_llr_svm(DTV_gk, LTV_gk, w_star_ext, K=K[k])))
            rr = mlf.bayes_Plots_svm(llr, LTR_g, version='mvg', K=K[k], C=C[c])
            print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_svm_gauss_unbalanced.png')
            
    ### GAUSSIANIZED BALANCED pt=0.5 ### K=1, K=10 - C=0.1, C=1, C=10 ###
    
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    K = numpy.array([1, 10])
    C = numpy.array([0.1, 1.0, 10])
    for k in range(K.size):
        for c in range(C.size):  
            for i in range(5):
                (DTR_gk, LTR_gk), (DTV_gk, LTV_gk) = mlf.split_db_kfold(DTR_g, LTR_g, i)
                Hext = mlf.compute_Hext (DTR_gk, LTR_gk, K=K[k])
                L, J, w_star_ext = mlf.compute_Dual_and_Primal_loss (LTR_gk, DTR_gk, Hext=Hext, K=K[k], C=C[c], pit=0.5)
                if (i == 0):
                    llr = mlf.compute_llr_svm(DTV_gk, LTV_gk, w_star_ext, K=K[k])
                else:
                    llr = numpy.concatenate((llr, mlf.compute_llr_svm(DTV_gk, LTV_gk, w_star_ext, K=K[k])))
            rr = mlf.bayes_Plots_svm(llr, LTR_g, version='mvg', K=K[k], C=C[c])
            print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_svm_gauss_balanced05.png')
    
    
    ### GAUSSIAN MIXTURE MODEL ###
    
    ### GAUSSIANIZED 5-FOLD FULL COVARIANCE ###
    
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    
    components = [2, 4, 8, 16, 32, 64] #number of components
    classes = 2
    
    for comp in components:
        for j in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR_g, LTR_g, j)
            mu = numpy.array(mlf.compute_means_gmm(DTR_kg, LTR_kg))
            gmm_init = mlf.compute_GMM(DTR_kg, LTR_kg, mu, version='mvg')
            for i in range(classes):
                gmm_p, ll_p = mlf.compute_LBG(DTR_kg[:, LTR_kg==i], numpy.array([gmm_init[i]]), 0.1, comp, 0, 10**-6)
                if (i == 0): 
                    gmm = gmm_p
                else:
                    gmm = numpy.vstack((gmm, gmm_p))
                #ll_p = ll_p.reshape((ll_p.shape[1]))
            gmm = gmm.reshape(classes, comp, 3)
            err, llr = mlf.compute_err_gmm (DTV_kg, gmm, comp, 2, LTV_kg)
            if (j == 0):
                llrlist = llr
            else:
                llrlist = numpy.concatenate((llrlist, llr))
        rr = mlf.bayes_Plots(llrlist, LTR_g, version="mvg", lam=10**-6)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_gmm_gauss_full.png')
        
        
    ### GAUSSIANIZED 5-FOLD DIAG COVARIANCE ###
    
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    
    components = [2, 4, 8, 16, 32] #number of components
    classes = 2
    
    for comp in components:
        for j in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR_g, LTR_g, j)
            mu = numpy.array(mlf.compute_means_gmm(DTR_kg, LTR_kg))
            gmm_init = mlf.compute_GMM(DTR_kg, LTR_kg, mu, version='nb')
            for i in range(classes):
                gmm_p, ll_p = mlf.compute_LBG(DTR_kg[:, LTR_kg==i], numpy.array([gmm_init[i]]), 0.1, comp, 0, 10**-6)
                if (i == 0): 
                    gmm = gmm_p
                else:
                    gmm = numpy.vstack((gmm, gmm_p))
                #ll_p = ll_p.reshape((ll_p.shape[1]))
            gmm = gmm.reshape(classes, comp, 3)
            err, llr = mlf.compute_err_gmm (DTV_kg, gmm, comp, 2, LTV_kg)
            if (j == 0):
                llrlist = llr
            else:
                llrlist = numpy.concatenate((llrlist, llr))
        rr = mlf.bayes_Plots(llrlist, LTR_g, version="nb", lam=10**-6)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_gmm_gauss_diag.png')
        
    
    ### GAUSSIANIZED 5-FOLD TIED COVARIANCE ###

    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    
    components = [2, 4, 8, 16, 32] #number of components
    classes = 2
    
    for comp in components:
        for j in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR_g, LTR_g, j)
            mu = numpy.array(mlf.compute_means_gmm(DTR_kg, LTR_kg))
            gmm_init = mlf.compute_GMM(DTR_kg, LTR_kg, mu, version='tied')
            for i in range(classes):
                gmm_p, ll_p = mlf.compute_LBG(DTR_kg[:, LTR_kg==i], numpy.array([gmm_init[i]]), 0.1, comp, 0, 10**-6)
                if (i == 0): 
                    gmm = gmm_p
                else:
                    gmm = numpy.vstack((gmm, gmm_p))
                #ll_p = ll_p.reshape((ll_p.shape[1]))
            gmm = gmm.reshape(classes, comp, 3)
            err, llr = mlf.compute_err_gmm (DTV_kg, gmm, comp, 2, LTV_kg)
            if (j == 0):
                llrlist = llr
            else:
                llrlist = numpy.concatenate((llrlist, llr))
        rr = mlf.bayes_Plots(llrlist, LTR_g, version="tied", lam=10**-6)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_gmm_gauss_tied.png')

    ### RAW 5-FOLD FULL COVARIANCE ###

    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    
    components = [2, 4, 8, 16, 32] #number of components
    classes = 2
    
    for comp in components:
        for j in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR, LTR, j)
            mu = numpy.array(mlf.compute_means_gmm(DTR_kg, LTR_kg))
            gmm_init = mlf.compute_GMM(DTR_kg, LTR_kg, mu, version='mvg')
            for i in range(classes):
                gmm_p, ll_p = mlf.compute_LBG(DTR_kg[:, LTR_kg==i], numpy.array([gmm_init[i]]), 0.1, comp, 0, 10**-6)
                if (i == 0): 
                    gmm = gmm_p
                else:
                    gmm = numpy.vstack((gmm, gmm_p))
                #ll_p = ll_p.reshape((ll_p.shape[1]))
            gmm = gmm.reshape(classes, comp, 3)
            err, llr = mlf.compute_err_gmm (DTV_kg, gmm, comp, 2, LTV_kg)
            if (j == 0):
                llrlist = llr
            else:
                llrlist = numpy.concatenate((llrlist, llr))
        rr = mlf.bayes_Plots(llrlist, LTR, version="mvg", lam=10**-6)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_gmm_raw_full.png')
        
    '''
    ### ONLY FOR SAVING 
    
    plt.figure()
    print("--------------------------------------------")
    
    components = [8] #number of components
    classes = 2
    
    for comp in components:
        for j in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR, LTR, j)
            mu = numpy.array(mlf.compute_means_gmm(DTR_kg, LTR_kg))
            gmm_init = mlf.compute_GMM(DTR_kg, LTR_kg, mu, version='mvg')
            for i in range(classes):
                gmm_p, ll_p = mlf.compute_LBG(DTR_kg[:, LTR_kg==i], numpy.array([gmm_init[i]]), 0.1, comp, 0, 10**-6)
                if (i == 0): 
                    gmm = gmm_p
                else:
                    gmm = numpy.vstack((gmm, gmm_p))
                #ll_p = ll_p.reshape((ll_p.shape[1]))
            gmm = gmm.reshape(classes, comp, 3)
            err, llr = mlf.compute_err_gmm (DTV_kg, gmm, comp, 2, LTV_kg)
            if (j == 0):
                llrlist = llr
            else:
                llrlist = numpy.concatenate((llrlist, llr))
        rr = mlf.bayes_Plots(llrlist, LTR, version="mvg", lam=10**-6)
        print(rr)
    numpy.save("out/llr_GMM_5fold_raw_8_comp", llrlist)
    '''
      
        
    ### RAW 5-FOLD DIAG COVARIANCE ###
    
    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    
    components = [2, 4, 8, 16, 32] #number of components
    classes = 2
    
    for comp in components:
        for j in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR, LTR, j)
            mu = numpy.array(mlf.compute_means_gmm(DTR_kg, LTR_kg))
            gmm_init = mlf.compute_GMM(DTR_kg, LTR_kg, mu, version='nb')
            for i in range(classes):
                gmm_p, ll_p = mlf.compute_LBG(DTR_kg[:, LTR_kg==i], numpy.array([gmm_init[i]]), 0.1, comp, 0, 10**-6)
                if (i == 0): 
                    gmm = gmm_p
                else:
                    gmm = numpy.vstack((gmm, gmm_p))
                #ll_p = ll_p.reshape((ll_p.shape[1]))
            gmm = gmm.reshape(classes, comp, 3)
            err, llr = mlf.compute_err_gmm (DTV_kg, gmm, comp, 2, LTV_kg)
            if (j == 0):
                llrlist = llr
            else:
                llrlist = numpy.concatenate((llrlist, llr))
        rr = mlf.bayes_Plots(llrlist, LTR, version="nb", lam=10**-6)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_gmm_raw_diag.png')
        
   
    ### RAW 5-FOLD TIED COVARIANCE ###

    plt.figure()
    plt.title('minDCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('minDCF value')
    print()
    print("--------------------------------------------")
    
    components = [2, 4, 8, 16, 32] #number of components
    classes = 2
    
    for comp in components:
        for j in range(5):
            (DTR_kg, LTR_kg), (DTV_kg, LTV_kg) = mlf.split_db_kfold(DTR, LTR, j)
            mu = numpy.array(mlf.compute_means_gmm(DTR_kg, LTR_kg))
            gmm_init = mlf.compute_GMM(DTR_kg, LTR_kg, mu, version='tied')
            for i in range(classes):
                gmm_p, ll_p = mlf.compute_LBG(DTR_kg[:, LTR_kg==i], numpy.array([gmm_init[i]]), 0.1, comp, 0, 10**-6)
                if (i == 0): 
                    gmm = gmm_p
                else:
                    gmm = numpy.vstack((gmm, gmm_p))
                #ll_p = ll_p.reshape((ll_p.shape[1]))
            gmm = gmm.reshape(classes, comp, 3)
            err, llr = mlf.compute_err_gmm (DTV_kg, gmm, comp, 2, LTV_kg)
            if (j == 0):
                llrlist = llr
            else:
                llrlist = numpy.concatenate((llrlist, llr))
        rr = mlf.bayes_Plots(llrlist, LTR, version="tied", lam=10**-6)
        print(rr)
    plt.legend()
    plt.savefig('./bayesPlots/5fold_gmm_raw_tied.png')
        
    
    ''' BAR CHART TEST FOR OUR RESULTS
    rr01 = [0.735725938009788, 0.767536704730832, 0.7300163132137031, 0.6924959216965743, 0.699021207177814]
    rr05 = [0.30587275693311583, 0.300163132137031, 0.27814029363784665, 0.30750407830342574, 0.3172920065252855]
    rr09 = [0.6982055464926591, 0.7487765089722677, 0.7096247960848288,  0.6957585644371943, 0.8311582381729201]
    
    plt.figure()
    label = ['2', '4', '8', '16', '32']
    index = numpy.arange(5)
    bar_width = 0.25
    plt.bar(index, rr01, bar_width, alpha=0.8, color='b', label='pit=0.1')
    plt.bar(index + bar_width, rr05, bar_width, alpha=0.8, color='r', label='pit=0.5')
    plt.bar(index + (bar_width*2), rr09, bar_width, alpha=0.8, color='g', label='pit=0.9')
    
    plt.xticks(index + bar_width, label)
    plt.legend()
    plt.ylabel('min DCF')
    plt.xlabel('GMM components')
    plt.title('minDCF for whole components')
    '''
        
    ### minDCF - actualDCT best models ###
    
    plt.figure()
    plt.title('minDCF - DCF bayes plots')
    plt.grid(color='black', linestyle='--', linewidth=0.5)

    eff, dcf, mindcf, res = mlf.compute_dcf_plots(numpy.load('out/llr_GMM_5fold_raw_8_comp.npy'), LTR)
    plt.plot(eff, mindcf, color='r', label='GMM - raw - mindcf' )
    plt.plot(eff, dcf, color='r', linestyle='--', label='GMM - raw - dcf' )

    eff, dcf, mindcf, res = mlf.compute_dcf_plots(numpy.load('out/llr_mvg_5fold_full_cov_gauss.npy'), LTR) 
    plt.plot(eff, mindcf, color='b', label='MVG full - gauss. - mindcf' )
    plt.plot(eff, dcf, color='b', linestyle='--', label='MVG full - gauss. - dcf' )

    eff, dcf, mindcf, res = mlf.compute_dcf_plots(numpy.load('out/llr_quad_logreg_5fold_unb_10m6.npy'), LTR)
    plt.plot(eff, mindcf, color='g', label='QLR - gauss. - mindcf' )
    plt.plot(eff, dcf, color='g', linestyle='--', label='QLR - gauss. - dcf' )

    plt.legend()
    plt.ylim([0.2, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF value')

    
    ### ROC CURVES ###
    
    plt.figure()
    plt.title('ROC curve')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.xlabel('FNR')
    plt.ylabel('TNR')

    x, y = mlf.generate_roc_axis (numpy.load('llr_GMM_5fold_raw_8_comp.npy'), LTR, 0.5, 1, 1)
    plt.plot(numpy.sort(x), numpy.sort(y), marker='.', label='GMM - raw', linewidth=0.1)

    x, y = mlf.generate_roc_axis (numpy.load('llr_mvg_5fold_full_cov_gauss.npy'), LTR_g, 0.5, 1, 1)
    plt.plot(numpy.sort(x), numpy.sort(y), marker='.', label='MVG ful - gauss.', linewidth=0.1)

    x, y = mlf.generate_roc_axis (numpy.load('llr_quad_logreg_5fold_unb_10m6.npy'), LTR_g, 0.5, 1, 1)
    plt.plot(numpy.sort(x), numpy.sort(y), marker='.', label='QLR - gauss.', linewidth=0.1)
    plt.legend()
    plt.tight_layout()
    
    
    ### EVALUATION SET TESTS ###
    '''
    llr0 = numpy.load("out/llr_mvg_5fold_full_cov_gauss.npy") # mvg gauss
    llr1 = numpy.load("out/llr_quad_logreg_5fold_unb_10m6.npy") #quad logreg gauss
    llr2 = numpy.load("out/llr_GMM_5fold_raw_8_comp.npy") #gmm 8comp raw
    '''
    
    DTR_g, LTR_g = mlf.compute_Gaussianize(DTR, LTR)
    DTE_g, LTE_g = mlf.compute_Gaussianize_test(DTR, LTR, DTE, LTE)
    llr_mvg_t, _, _ = mlf.compute_Results(DTR_g, LTR_g, DTE_g, LTE_g, gauss=True) ## MVG
    numpy.save("out/llr_mvg_5fold_full_cov_gauss_test", llr_mvg)
    
    DTR_ext_g = mlf.extended_data_quad_lg(DTR_g)
    DTE_ext_g = mlf.extended_data_quad_lg(DTE_g)
    
    plt.figure()
    
    ### QUAD LOG REG

    x0 = numpy.zeros(DTR_ext_g.shape[0]+1)
    lam = numpy.array([10**-6])
    for l in lam:
        llr = 0
        x, f, d = scipy.optimize.fmin_l_bfgs_b(mlf.logreg_obj, x0, args=(DTR_ext_g, LTR_g, l), approx_grad=True)
        llr = mlf.compute_llr_logreg(x, DTE_ext_g, LTE_g)
        rr = mlf.bayes_Plots(llr, LTE_g, version='mvg', lam=l)
        print(rr)
        numpy.save("out/llr_quad_logreg_5fold_unb_10m6_test", llr)
        
    #GMM RAW 
    
    components = [8] #number of components
    classes = 2
    
    for comp in components:
        mu = numpy.array(mlf.compute_means_gmm(DTR, LTR))
        gmm_init = mlf.compute_GMM(DTR, LTR, mu, version='mvg')
        for i in range(classes):
            gmm_p, ll_p = mlf.compute_LBG(DTR[:, LTR==i], numpy.array([gmm_init[i]]), 0.1, comp, 0, 10**-6)
            if (i == 0): 
                gmm = gmm_p
            else:
                gmm = numpy.vstack((gmm, gmm_p))
            #ll_p = ll_p.reshape((ll_p.shape[1]))
        gmm = gmm.reshape(classes, comp, 3)
        err, llr = mlf.compute_err_gmm (DTE, gmm, comp, 2, LTE)
        rr = mlf.bayes_Plots(llr, LTE, version="mvg", lam=10**-6)
        print(rr)
        numpy.save("out/llr_GMM_5fold_raw_8_comp_test", llr)
        
    
    