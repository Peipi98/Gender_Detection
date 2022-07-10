import sys
import numpy
import scipy
import scipy.stats as stats
#from mlFunc import empirical_covariance, empirical_mean, mrow, mcol, logpdf_GAU_ND, get_DTRs
sys.path.append('../')
from validators import compute_min_DCF
from mlFunc import gaussianize_features


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

def empirical_cov(D, muc):
    DC = D - muc #class samples centered
    C = (numpy.dot(DC , DC.T))/D.shape[1]
    return C
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

#==============================================================================
# --------- LOAD FILE ---------------------------------------------------------
def load(fname):
    #we have 9 parameters per row
    #8 values and one label
    #we take the 8 values and put em in a column format into DList
    #we take the last argument and put it in the labelsList
    #so we have a 1:1 association index between the column and the label
    
    DList = []
    labelsList = []
    with open(fname) as f:
        for line in f:
            try:
               #i take the first 4 numbers as a vector
               attrs = line.split(",")[0:8]
               #i parse the number from string to float, then i
               #transform it in a column vector
               attrs = mcol(numpy.array([float(i) for i in attrs])) 
               label = int(line.split(",")[-1].strip())
               DList.append(attrs)
               labelsList.append(label)
            except:
                pass
    return numpy.hstack(DList), numpy.array(labelsList, dtype = numpy.int32)
#==============================================================================

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
    U, s, _ = numpy.linalg.svd(empirical_cov(X,empirical_mean(X)))
    s[s<psi] = psi
    covNew = numpy.dot(U, mcol(s)*U.T)
    GMM = [(1,empirical_mean(X), covNew)]

    while len(GMM)<=G:
        #print('########################################## NEW ITER')
        if len(GMM) != 1:
            if typeOf=='Full':
                GMM=GMM_EM(X,GMM,psi)
            if typeOf=='Diag':
                GMM=GMM_EM_diag(X,GMM,psi)
            if typeOf=='Tied':
                GMM=GMM_EM_tied(X,GMM,psi)
            if typeOf=='TiedDiag':
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
def GMM_EM(X,gmm,psi= 0.1): #X -> ev
    llNew=None
    llOld=None
    G=len(gmm)
    N=X.shape[1]
    while llOld is None or llNew-llOld>1e-6: #how much the likelihood increase
    #compute the matrix of joint density for sample and components
        llOld=llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew=SM.sum()/N #compute the log likelihood for all the data -> samples are independent
        #E STEP ----- compute posterior
        #print("E STEP")
        P=numpy.exp(SJ-SM) #posterior: join - marginal 
        gmmNew=[]
        #then we need to do the update and generate updated parameters
        #for each component of G we need to compute the mean, covariance and weight (we use the sufficient statistics)
        #M STEP -------- calcolo nuovi valori
        for g in range(G):
            #print("M STEP")
            gamma=P[g,:] #simple way to compute weighted sum
            Z=gamma.sum()
            F=(mrow(gamma)*X).sum(1) #broadcasted each element of matrix X with the corresponding gamma and then sum over al samples
            S=numpy.dot(X,(mrow(gamma)*X).T) #matrix matrix multiplication
            w=Z/N  #peso
            mu=mcol(F/Z) #media
            Sigma=S/Z-numpy.dot(mu,mu.T) #covariance
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s<psi] = psi
            Sigma = numpy.dot(U, mcol(s)*U.T)
            gmmNew.append((w,mu,Sigma))
        gmm=gmmNew
        #print(llNew,'llnew') #increase - if decrease problem
    #print(llNew-llOld,'llnew-llold')
    return gmm

def GMM_EM_tiedDiag(X,gmm,psi= 0.1): #X -> ev
    llNew=None
    llOld=None
    G=len(gmm)
    N=X.shape[1]
    while llOld is None or llNew-llOld>1e-6: #how much the likelihood increase
    #compute the matrix of joint density for sample and components
        llOld=llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew=SM.sum()/N #compute the log likelihood for all the data -> samples are independent
        #E STEP ----- compute posterior
        #print("E STEP")
        P=numpy.exp(SJ-SM) #posterior: join - marginal 
        gmmNew=[]
        #then we need to do the update and generate updated parameters
        #for each component of G we need to compute the mean, covariance and weight (we use the sufficient statistics)
        #M STEP -------- calcolo nuovi valori
        sigmaTied=numpy.zeros((X.shape[0],X.shape[0]))
        for g in range(G):
            #print("M STEP")
            gamma=P[g,:] #simple way to compute weighted sum
            Z=gamma.sum()
            F=(mrow(gamma)*X).sum(1) #broadcasted each element of matrix X with the corresponding gamma and then sum over al samples
            S=numpy.dot(X,(mrow(gamma)*X).T) #matrix matrix multiplication
            w=Z/N  #peso
            mu=mcol(F/Z) #media
            sigma = S/Z - numpy.dot(mu, mu.T)
            sigmaTied += Z * sigma
            gmmNew.append((w, mu))   
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

def GMM_EM_tied(X,gmm,psi=0.01):
    #print("TIED APPLICATION")
    llNew=None
    llOld=None
    G=len(gmm)
    N=X.shape[1]
    while llOld is None or llNew-llOld>1e-5: #how much the likelihood increase
    #compute the matrix of joint density for sample and components
        
        llOld=llNew
        SJ=numpy.zeros((G,N))
        #qui devo calcolare joint e marginal
        #SJ,SM=logpdf_GMM(D,gmm)
        #llOld=llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew=SM.sum()/N #compute the log likelihood for all the data -> samples are independent
        #E STEP ----- 
        #print("E STEP")
        P=numpy.exp(SJ-SM) #posterior: join - marginal 
        #Modello tied
        gmmNew=[]
        #then we need to do the update and generate updated parameters
        #for each component of G we need to compute the mean, covariance and weight (we use the sufficient statistics)
        #M STEP -------- calcolo nuovi valori
        sigmaTied = numpy.zeros((X.shape[0],X.shape[0]))
        for g in range(G):
            #print("M STEP")
            gamma=P[g,:] #simple way to compute weighted sum
            Z=gamma.sum()
            F=(mrow(gamma)*X).sum(1) #broadcasted each element of matrix X with the corresponding gamma and then sum over al samples
            S=numpy.dot(X,(mrow(gamma)*X).T) #matrix matrix multiplication
            w=Z/N  #peso
            mu=mcol(F/Z) #media
            Sigma=S/Z-numpy.dot(mu,mu.T) #covariance
            sigmaTied+=Z*Sigma
            gmmNew.append((w,mu))
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

def GMM_Full(DTR,DTE,LTR,alpha, G, typeOf ,psi = 0.1):
    
    DTR0=DTR[:,LTR==0]
    gmm0=LBG(DTR0,alpha,G,psi,typeOf)
    _,llr0=logpdf_GMM(DTE,gmm0)

    DTR1=DTR[:,LTR==1]
    gmm1=LBG(DTR1,alpha,G,psi,typeOf)
    _,llr1=logpdf_GMM(DTE,gmm1)

    return llr1-llr0
        
def GMM_EM_diag(X,gmm,psi=0.01):
    llNew = None
    llOld = None
    G=len(gmm)
    N = D.shape[1]
    while llOld == None or llNew-llOld>1e-5:
        llOld=llNew
        SJ=numpy.zeros((G,N))
        #qui devo calcolare joint e marginal
        #llOld=llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew=SM.sum()/N #compute the log likelihood for all the data -> samples are independent
        #E STEP ----- 
        #print("E STEP")
        P = numpy.exp(SJ - SM)
        #then we need to do the update and generate updated parameters
        #for each component of G we need to compute the mean, covariance and weight (we use the sufficient statistics)
        #M STEP -------- calcolo nuovi valori
        gmmNew = []
        for i in range(G):
            gamma = P[i, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            sigma = S/Z - numpy.dot(mu, mu.T)
            sigma *= numpy.eye(sigma.shape[0])
            U, s, _ = numpy.linalg.svd(sigma)
            s[s<psi] = psi
            sigma = numpy.dot(U, mcol(s)*U.T)
            gmmNew.append((w, mu, sigma))
        gmm = gmmNew
    return gmm

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

if __name__ == '__main__':
    D, L = load('../Train.txt')
    DE, LE = load('../Test.txt')
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
    # ----------- PLOTS -----------------------
    DCF.plot_DCF_GMM(exponents, numpy.hstack(vettoreMatriciDiagZ), numpy.hstack(vettoreMatriciDiagG), "GMM_Diag")
    DCF.plot_DCF_GMM(exponents, numpy.hstack(vettoreMatriciTiedZ), numpy.hstack(vettoreMatriciTiedG), "GMM_Tied")
    DCF.plot_DCF_GMM(exponents, numpy.hstack(vettoreMatriciFullZ), numpy.hstack(vettoreMatriciFullG), "GMM_Full")
    DCF.plot_DCF_GMM(exponents, numpy.hstack(vettoreMatriciTiedDiagZ), numpy.hstack(vettoreMatriciTiedDiagG), "GMM_tiedDiag")
# =============================================================================
#     print('z')
#     _, matriceTiedZ, matriceDiagZ = k_fold(ZD,L,3,0.1,1)
#     print('g')
#     _, matriceTiedG, matriceDiagG = k_fold(GD,L,3,0.1,1)
# =============================================================================
# =============================================================================
#     print('matriceTiedZ',matriceTiedZ)
#     print('matriceDiagZ',matriceDiagZ)
#     print('matriceDiagG',matriceDiagG)
#     print('matriceTiedG',matriceTiedG)
# =============================================================================
    