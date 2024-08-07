import numpy
import scipy
import scipy.special
import matplotlib.pyplot as plt
import bayesRisk

def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        '0': 0,
        '1': 1,
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = vcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def logpdf_GAU_ND(x, mu, C): # Fast version from Lab 4
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0) 

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    
    return (DTR, LTR), (DVAL, LVAL)

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GMM(X, gmm):

    S = []
    
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
        
    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens

def smooth_covariance_matrix(C, psi):

    U, s, Vh = numpy.linalg.svd(C)
    s[s<psi]=psi
    CUpd = U @ (vcol(s) * U.T)
    return CUpd

# X: Data matrix
# gmm: input gmm
# covType: 'Full' | 'Diagonal' | 'Tied'
# psiEig: factor for eignvalue thresholding
#
# return: updated gmm
def train_GMM_EM_Iteration(X, gmm, covType = 'Full', psiEig = None): 

    assert (covType.lower() in ['full', 'diagonal', 'tied'])
    
    # E-step
    S = []
    
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
        
    S = numpy.vstack(S) # Compute joint densities f(x_i, c), i=1...n, c=1...G
    logdens = scipy.special.logsumexp(S, axis=0) # Compute marginal for samples f(x_i)

    # Compute posterior for all clusters - log P(C=c|X=x_i) = log f(x_i, c) - log f(x_i)) - i=1...n, c=1...G
    # Each row for gammaAllComponents corresponds to a Gaussian component
    # Each column corresponds to a sample (similar to the matrix of class posterior probabilities in Lab 5, but here the rows are associated to clusters rather than to classes
    gammaAllComponents = numpy.exp(S - logdens)

    # M-step
    gmmUpd = []
    for gIdx in range(len(gmm)): 
    # Compute statistics:
        gamma = gammaAllComponents[gIdx] # Extract the responsibilities for component gIdx
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1)) # Exploit broadcasting to compute the sum
        S = (vrow(gamma) * X) @ X.T
        muUpd = F/Z
        CUpd = S/Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == 'diagonal':
            CUpd  = CUpd * numpy.eye(X.shape[0]) # An efficient implementation would store and employ only the diagonal terms, but is out of the scope of this script
        gmmUpd.append((wUpd, muUpd, CUpd))

    if covType.lower() == 'tied':
        CTied = 0
        for w, mu, C in gmmUpd:
            CTied += w * C
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]
        
    return gmmUpd

# Train a GMM until the average dela log-likelihood becomes <= epsLLAverage
def train_GMM_EM(X, gmm, covType = 'Full', psiEig = None, epsLLAverage = 1e-6):

    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType = covType, psiEig = psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        gmm = gmmUpd
        llOld = llUpd
        it = it + 1      
    return gmm
    
def split_GMM_LBG(gmm, alpha = 0.1):

    gmmOut = []
    for (w, mu, C) in gmm:
        U, s, Vh = numpy.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

# Train a full model using LBG + EM, starting from a single Gaussian model, until we have numComponents components. lbgAlpha is the value 'alpha' used for LBG, the otehr parameters are the same as in the EM functions above
def train_GMM_LBG_EM(X, numComponents, covType, psiEig = None, epsLLAverage = 1e-6, lbgAlpha = 0.1):

    mu, C = compute_mu_C(X)

    if covType.lower() == 'diagonal':
        C = C * numpy.eye(X.shape[0]) # We need an initial diagonal GMM to train a diagonal GMM
    
    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C, psiEig))] # 1-component model - if we impose the eignevalus constraint, we must do it for the initial 1-component GMM as well
    else:
        gmm = [(1.0, mu, C)] # 1-component model
    
    while len(gmm) < numComponents:
        # Split the components
        gmm = split_GMM_LBG(gmm, lbgAlpha)
        # Run the EM for the new GMM
        gmm = train_GMM_EM(X, gmm, covType = covType, psiEig = psiEig, epsLLAverage = epsLLAverage)
    return gmm

# Optimize the logistic regression loss
def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0 # Z: 1 if class=1, -1 if class=0

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w,b = v[:-1],v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b #compute the score (wT*x+b)

        loss = numpy.logaddexp(0, -ZTR * s) #this is log(1+e^-zi(wT*xi+b))

        G = -ZTR / (1.0 + numpy.exp(ZTR * s)) #computing the gradient to speed up computations
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()

        return loss.mean() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])#we return J(w,b)

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]
def plot_bayes_plot(actDCF,minDCF,msg):
    plt.figure(msg)
    plt.xscale('log', base=10)
    plt.plot(numpy.logspace(-4, 4, 21), actDCF, label="actDCF", color='r', marker= 'o')  
    plt.plot(numpy.logspace(-4, 4, 21), minDCF, label="minDCF", color='b', marker= 'o')   
    plt.title(msg)
    plt.grid()
    #plt.ylim(0,1)
    plt.xlabel("lambda")
    plt.ylabel("DCF")
    plt.legend()
    plt.show()

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)

    return rbfKernelFunc

# kernelFunc: function that computes the kernel matrix from two data matrices
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR)
    H = vcol(ZTR) * vrow(ZTR) * K

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE)
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore # we directly return the function to score a matrix of test samples
    
if __name__ == '__main__':
    D, L = load("trainData.txt")
    #we take a fraction of our data!
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    #now we train the training set for class 1 and class 2 with GMM of 32 components
    Dreal  = DTR[:, LTR == 1]
    Dfake = DTR[:, LTR == 0]
    #we first try the full MVG
    print("*"*40)
    print("GMM using Full covariance matrix")
    for m in ([1,2,4,8,16,32]):
        GMMreal = train_GMM_LBG_EM(Dreal, m,covType="Full",psiEig=0.01)
        for n in ([1,2,4,8,16,32]):
            GMMfake= train_GMM_LBG_EM(Dfake, n,covType="Full",psiEig=0.01)
            llr=logpdf_GMM(DVAL, GMMreal) -logpdf_GMM(DVAL,GMMfake)
            #print("LLR="+str(llr))
            actDCF=bayesRisk.compute_actDCF_binary_fast(llr, LVAL, 0.1, 1.0, 1.0)
            minDCF=bayesRisk.compute_minDCF_binary_fast(llr, LVAL, 0.1, 1.0, 1.0)
            print("-"*40)
            print("GMM Full covariance matrix components true class:"+str(m)+" components fake class:"+str(n))
            print("actDCF="+str(actDCF))
            print("minDCF="+str(minDCF))

    print()
    print("*"*40)
    print("GMM using diagonal covariance matrix")
    print()
    for m in ([1,2,4,8,16,32]):
        GMMreal = train_GMM_LBG_EM(Dreal, m,covType="Diagonal",psiEig=0.01)
        for n in ([1,2,4,8,16,32]):
            GMMfake= train_GMM_LBG_EM(Dfake, n,covType="Diagonal",psiEig=0.01)
            llr=logpdf_GMM(DVAL, GMMreal) -logpdf_GMM(DVAL,GMMfake)
            #print("LLR="+str(llr))
            actDCF=bayesRisk.compute_actDCF_binary_fast(llr, LVAL, 0.1, 1.0, 1.0)
            minDCF=bayesRisk.compute_minDCF_binary_fast(llr, LVAL, 0.1, 1.0, 1.0)
            print("-"*40)
            print("GMM diagonal covariance matrix components true class:"+str(m)+" components fake class:"+str(n))
            print("actDCF="+str(actDCF))
            print("minDCF="+str(minDCF))

#now let's plot the minDCF and actDCF of the 3 best classifier (quadratic LR,rbf kernel SVM and GMM)
#the best DCF was given by a small lambda value
    actDCF = []
    minDCF = []
    l=0.005
    effPriorLogOdds = numpy.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    for effPrior in effPriors:
        w, b = trainLogRegBinary(DTR, LTR,l) # Train model
        sVal = numpy.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        # Compute LLR-like scores - remove the empirical prior
        sValLLR = sVal - numpy.log(pEmp / (1-pEmp))
        # Compute optimal decisions for the prior 0.1
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, effPrior, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, effPrior, 1.0, 1.0))
    plot_bayes_plot(actDCF,minDCF,"quadratic LR with lambda=0.005")

    #the best SVM was the one with small regularization (big C value) and gamma=e^-3
    #so we take C=30, 
    C=30
    gamma=numpy.exp(-3)
    actDCF = []
    minDCF = []
    kernelFunc=rbfKernel(gamma)
    for effPrior in effPriors:
        fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc)
        SVAL = fScore(DVAL)
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, effPrior, 1.0, 1.0))
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, effPrior, 1.0, 1.0))
    plot_bayes_plot(actDCF,minDCF,"rbf SVM with C=30, gamma=e^-3")

    #now we plot the best GMM model: diagonal GMM with 8 components for the fake class and 32 components for the true class
    actDCF = []
    minDCF = []
    GMMreal = train_GMM_LBG_EM(Dreal, 32,covType="Diagonal",psiEig=0.01)
    GMMfake= train_GMM_LBG_EM(Dfake, 8,covType="Diagonal",psiEig=0.01)
    for effPrior in effPriors:
        llr=logpdf_GMM(DVAL, GMMreal) -logpdf_GMM(DVAL,GMMfake)
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(llr, LVAL, 0.1, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(llr, LVAL, 0.1, 1.0, 1.0))
    plot_bayes_plot(actDCF,minDCF,"GMM")
    

    