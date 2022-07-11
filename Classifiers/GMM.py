
import numpy 
import scipy.special 
from mlFunc import empirical_covariance, gaussianize_features
from validators import compute_min_DCF, confusion_matrix_binary
import scipy.stats as stats
import pylab


#==============================================================================
# ------ BASIC FUNCTIONS --------

def mcol(v):
    #reshape a row vector in a column vector
    #!!!! if u write (v.size,) it will remain a ROW vector
    #So don't forget the column value "1"
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1, v.size))

def empirical_mean(D):
    return mcol(D.mean(1))

def plot_ROC(llrs, LTE, title):
    thresholds = numpy.array(llrs)
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
    FPR = numpy.zeros(thresholds.size)
    TPR = numpy.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = numpy.int32(llrs > t)
        conf = confusion_matrix_binary(Pred, LTE)
        TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
        FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
    pylab.plot(FPR, TPR)
    pylab.title(title)
    pylab.savefig('./images/ROC_' + title + '.png')
    pylab.show()
#==============================================================================

#==============================================================================
# ----------- LOG DENSITIES AND MARGINALS -------------------------------------

def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    
    for g in range(len(gmm)):
        #print (w,mu,C)
        (w, mu, C) = gmm[g]
        S[g, :] = logpdf_GAU_ND(X, mu, C) + numpy.log(w)
        
    logdens = scipy.special.logsumexp(S, axis=0)
    
    return S, logdens

def logpdf_GAU_ND(X,mu,C) :
    
    res = -0.5*X.shape[0]*numpy.log(2*numpy.pi)
    res += -0.5*numpy.linalg.slogdet(C)[1]
    res += -0.5*((X-mu)*numpy.dot(numpy.linalg.inv(C), (X-mu))).sum(0) #1
    return res

#==============================================================================
# -------------------------------- LBG ----------------------------------------
def split(GMM, alpha = 0.1): 
    size = len(GMM)
    splittedGMM = []
    for i in range(size):
        U, s, Vh = numpy.linalg.svd(GMM[i][2])
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]+d, GMM[i][2]))
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]-d, GMM[i][2]))
    return splittedGMM

#
def LBG( X,alpha,G,psi, typeOf='Full'):
    U, s, _ = numpy.linalg.svd(empirical_covariance(X,empirical_mean(X)))
    s[s<psi] = psi
    covNew = numpy.dot(U, mcol(s)*U.T)
    GMM = [(1,empirical_mean(X), covNew)]

    while len(GMM)<=G:
        #print('########################################## NEW ITER')
        if len(GMM) != 1:
            if typeOf=='full':
                GMM=GMM_EM(X,GMM,psi)
            if typeOf=='diag':
                GMM=GMM_EM_diag(X,GMM,psi)
            if typeOf=='tied_full':
                GMM=GMM_EM_tied(X,GMM,psi)
            if typeOf=='tied_diag':
                GMM=GMM_EM_tiedDiag(X,GMM,psi)
        #print('########################################## FIN ITER')
        if len(GMM)==G: 
            break

        gmmNew=[]
        for i in range (len(GMM)): 
            #nuove componenti
            (w,mu,sigma)=GMM[i]
            U,s,vh=numpy.linalg.svd(sigma)
            d=U[:,0:1]*s[0]**0.5*alpha
            gmmNew.append((w/2, mu + d, sigma))
            gmmNew.append((w/2, mu - d, sigma))
            #print("newGmm",gmmNew)
        GMM = gmmNew
            
    return GMM
#==============================================================================

#==============================================================================
# ---------------------- GMM _ EM, GMM _ FULL DIAGONAL,K FOLD -----------------
def GMM_EM(X, gmm, psi=0.01):
    '''
    EM algorithm for GMM full covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s<psi] = psi
            Sigma = numpy.dot(U, mcol(s)*U.T)
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew
        # print(llNew)
    #print(llNew-llOld)
    return gmm

def GMM_EM_tiedDiag(X,gmm,psi= 0.01): #X -> ev
    '''
    EM algorithm for GMM tied diagonal covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    sigma_array = []
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(
                X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        sigmaTied=numpy.zeros((X.shape[0],X.shape[0]))
        for g in range(G):
            # m step
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            #Sigma = S/Z - numpy.dot(mu, mu.T)
            #Sigma = Sigma * numpy.eye(Sigma.shape[0])
            #Sigma = Sigma * Z
            #gmmNew.append((w, mu, Sigma))
            sigma = S/Z - numpy.dot(mu, mu.T)
            sigmaTied += Z * sigma
            gmmNew.append((w, mu))  

        # calculate tied covariance
        gmm = gmmNew
        sigmaTied /= N
        sigmaTied *= numpy.eye(sigma.shape[0])
        U, s, _ = numpy.linalg.svd(sigmaTied)
        s[s<psi] = psi
        sigmaTied = numpy.dot(U, mcol(s)*U.T)
        
        newGmm = []
        for i in range(len(gmm)):
            (w, mu) = gmm[i]
            newGmm.append((w, mu, sigmaTied))
        
        gmm = newGmm
        #print(llNew,'llnew') #increase - if decrease problem
    #print(llNew-llOld,'llnew-llold')
    return gmm

def GMM_EM_tied(X, gmm, psi=0.01):
    '''
    EM algorithm for GMM tied full covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    sigma_array = []
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []

        sigmaTied = numpy.zeros((X.shape[0],X.shape[0]))
        for g in range(G):
            # m step
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            sigmaTied += Z*Sigma
            gmmNew.append((w,mu))
            #Sigma = Sigma * Z
            #gmmNew.append((w, mu, Sigma))

        # calculate tied covariance
        gmm=gmmNew
        sigmaTied /= N
        U,s,_ = numpy.linalg.svd(sigmaTied)
        s[s<psi]=psi 
        sigmaTied = numpy.dot(U, mcol(s)*U.T)

        gmmNew=[]
        for g in range(G):
            (w,mu)=gmm[g]
            gmmNew.append((w,mu,sigmaTied))
        gmm=gmmNew
        #print(llNew,'llnew') #increase - if decrease problem
    #print(llNew-llOld,'llnew-llold')
    return gmm

def GMM_Full(DTR,DTE,LTR,alpha, G, typeOf ,psi = 0.01):
    
    DTR0=DTR[:,LTR==0]
    gmm0=LBG(DTR0,alpha,G,psi,typeOf)
    _,llr0=logpdf_GMM(DTE,gmm0)

    DTR1=DTR[:,LTR==1]
    gmm1=LBG(DTR1,alpha,G,psi,typeOf)
    _,llr1=logpdf_GMM(DTE,gmm1)

    return llr1-llr0
        
def GMM_EM_diag(X, gmm, psi=0.01):
    '''
    EM algorithm for GMM diagonal covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)

        gmmNew = []
        for g in range(G):
            # m step
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            Sigma = Sigma * numpy.eye(Sigma.shape[0])
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < psi] = psi
            sigma = numpy.dot(U, mcol(s)*U.T)
            gmmNew.append((w, mu, sigma))
        gmm = gmmNew
        # print(llNew)
    # print(llNew)
    return gmm

'''
def k_fold(D,L,k, alpha, G, seed=0):
    nTrain = int(D.shape[1]*(k-1)/k)
    nTest = int(D.shape[1]) - nTrain
    numpy.random.seed(seed) 
    idx_ = numpy.random.permutation(D.shape[1])
    scores = []
    labels = []
    priors = [0.5,0.1,0.9]
    minDCF05Tied=numpy.zeros((1,3))
    minDCF05Diag=numpy.zeros((1,3))
    for pi in priors:
        scores = []
        scoresTied=[]
        labels = []
        idx = idx_
        for i in range(k):
            #print("Fold :", i+1)
            idxTrain = idx[0:nTrain] 
            idxTest = idx[nTrain:]
            DTR = D[:, idxTrain] 
            DTE = D[:, idxTest]
            LTR = L[idxTrain] 
            LTE = L[idxTest]
            scores.extend(GMM_Full(DTR,DTE,LTR,alpha,2**G,'Diag').tolist())
            scoresTied.extend(GMM_Full(DTR,DTE,LTR,alpha,2**G,'Tied').tolist())
            labels.append(LTE.tolist())
            idx = numpy.roll(idx,nTest,axis=0)
        #if gaussianized == 1:
            #print('minDCF LR with prior ', pi_T ,' and application ', pi,', ', 1,', ', 1 , ' with gaussianized features : ', compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
        #else :
            if pi == 0.5:
                minDCF05Tied[0][0]=compute_min_DCF(scoresTied, numpy.hstack(labels), pi, 1, 1)
                minDCF05Diag[0][0]=compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1)
        print('minDCF GMM with application DIAG 64 c', pi,', ', 1,', ', 1 , ' : ', "%.3f" % compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
        print('minDCF GMM with application TIED 64 c', pi,', ', 1,', ', 1 , ' : ', "%.3f" % compute_min_DCF(scoresTied, numpy.hstack(labels), pi, 1, 1))
        #if pi == 0.5 and pi_T == 0.5:
            #DCF.plot_minDCF(scores, numpy.hstack(labels))
   

    
    #print("Avarage error with cross-validation Tied PCA=7 prior: 0.1: ", "%.3f" % (acc2*100/k) , "%")
    #print("Avarage error with cross-validation Naive PCA=7 prior: 0.1: ", "%.3f" % (acc3*100/k) , "%")
    #"R" are the train, "E" are evaluations.
    return scores,minDCF05Tied,minDCF05Diag
'''
'''
def k_fold05(D,L,k, alpha, G, seed=0):
    nTrain = int(D.shape[1]*(k-1)/k)
    nTest = int(D.shape[1]) - nTrain
    numpy.random.seed(seed) 
    idx_ = numpy.random.permutation(D.shape[1])
    scoresDiag = []
    scoresTied=[]
    scoresFull=[]
    scoresTiedDiag=[]
    labels = []
    idx = idx_
    #priors = [0.5,0.1,0.9]
    minDCFTied05=None
    minDCFDiag05=None
    for i in range(k):
        #print("Fold :", i+1)
        idxTrain = idx[0:nTrain] 
        idxTest = idx[nTrain:]
        DTR = D[:, idxTrain] 
        DTE = D[:, idxTest]
        LTR = L[idxTrain] 
        LTE = L[idxTest]
        scoresDiag.extend(GMM_Full(DTR,DTE,LTR,alpha,2**G,'Diag').tolist())
        scoresTied.extend(GMM_Full(DTR,DTE,LTR,alpha,2**G,'Tied').tolist())
        scoresFull.extend(GMM_Full(DTR,DTE,LTR,alpha,2**G,'Full').tolist())
        scoresTiedDiag.extend(GMM_Full(DTR,DTE,LTR,alpha,2**G,'TiedDiag').tolist())
        labels.append(LTE.tolist())
        idx = numpy.roll(idx,nTest,axis=0)
    #if gaussianized == 1:
        #print('minDCF LR with prior ', pi_T ,' and application ', pi,', ', 1,', ', 1 , ' with gaussianized features : ', compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
    #else :
    minDCFDiag05=compute_min_DCF(scoresDiag, numpy.hstack(labels), 0.5, 1, 1)
    minDCFTied05=compute_min_DCF(scoresTied, numpy.hstack(labels), 0.5, 1, 1)
    minDCFFull05=compute_min_DCF(scoresFull, numpy.hstack(labels), 0.5, 1, 1)
    minDCFTiedDiag05=compute_min_DCF(scoresTiedDiag, numpy.hstack(labels), 0.5, 1, 1)
    print('minDCF GMM with application DIAG ', 0.5,', ', 1,', ', 1 , ' : ', "%.3f" % minDCFDiag05)
    print('minDCF GMM with application TIED', 0.5,', ', 1,', ', 1 , ' : ', "%.3f" % minDCFTied05)
    print('minDCF GMM with application FUKLL ', 0.5,', ', 1,', ', 1 , ' : ', "%.3f" % minDCFFull05)
    print('minDCF GMM with application TIEDDIAG', 0.5,', ', 1,', ', 1 , ' : ', "%.3f" % minDCFTiedDiag05)
    #if pi == 0.5 and pi_T == 0.5:
        #DCF.plot_minDCF(scores, numpy.hstack(labels))
   

    
    #print("Avarage error with cross-validation Tied PCA=7 prior: 0.1: ", "%.3f" % (acc2*100/k) , "%")
    #print("Avarage error with cross-validation Naive PCA=7 prior: 0.1: ", "%.3f" % (acc3*100/k) , "%")
    #"R" are the train, "E" are evaluations.
    return minDCFDiag05,minDCFTied05,minDCFFull05,minDCFTiedDiag05
#==============================================================================  
'''  
'''
if __name__ == '__main__':
    D, L = load('Train.txt')
    DE, LE = load('Test.txt')
    #_ = k_fold(D,L,5)
    ZD = stats.zscore(D, axis=1)
    GD = gaussianize_features(ZD, ZD)
    # ---------------------------
    vettoreMatriciDiagZ=[]
    vettoreMatriciTiedZ=[]
    vettoreMatriciFullZ=[]
    vettoreMatriciTiedDiagZ=[]
    # --------------------------
    vettoreMatriciDiagG=[]
    vettoreMatriciTiedG=[]
    vettoreMatriciFullG=[]
    vettoreMatriciTiedDiagG=[]
    # ----------------
    componentsToTry=[1,2,3,4,5,6,7] 
    exponents=[2,4,8,16,32,64,128]
    for i in componentsToTry:
        print("components: ",i)
        matriceTiedZ, matriceDiagZ, matriceFullZ, matriceTiedDiagZ = k_fold05(ZD,L,3,0.1,i)
        matriceTiedG, matriceDiagG, matriceFullG, matriceTiedDiagG = k_fold05(GD,L,3,0.1,i)
        # Z NORM
        vettoreMatriciDiagZ.append(matriceDiagZ)
        vettoreMatriciTiedZ.append(matriceTiedZ)
        vettoreMatriciFullZ.append(matriceFullZ)
        vettoreMatriciTiedDiagZ.append(matriceTiedDiagZ)
        # G NORM
        vettoreMatriciDiagG.append(matriceDiagG)
        vettoreMatriciTiedG.append(matriceTiedG)
        vettoreMatriciFullG.append(matriceFullG)
        vettoreMatriciTiedDiagG.append(matriceTiedDiagG)
        '''