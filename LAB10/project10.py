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
    
    gmm = [(1.0, mu, C)] # 1-component model
    
    while len(gmm) < numComponents:
        # Split the components
        gmm = split_GMM_LBG(gmm, lbgAlpha)
        # Run the EM for the new GMM
        gmm = train_GMM_EM(X, gmm, covType = covType, psiEig = psiEig, epsLLAverage = epsLLAverage)
    return gmm

    
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
        GMMreal = train_GMM_LBG_EM(Dreal, m,"Full")
        for n in ([1,2,4,8,16,32]):
            GMMfake= train_GMM_LBG_EM(Dfake, n,"Full")
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
        GMMreal = train_GMM_LBG_EM(Dreal, m,"Diagonal")
        for n in ([1,2,4,8,16,32]):
            GMMfake= train_GMM_LBG_EM(Dfake, n,"Diagonal")
            llr=logpdf_GMM(DVAL, GMMreal) -logpdf_GMM(DVAL,GMMfake)
            #print("LLR="+str(llr))
            actDCF=bayesRisk.compute_actDCF_binary_fast(llr, LVAL, 0.1, 1.0, 1.0)
            minDCF=bayesRisk.compute_minDCF_binary_fast(llr, LVAL, 0.1, 1.0, 1.0)
            print("-"*40)
            print("GMM diagonal covariance matrix components true class:"+str(m)+" components fake class:"+str(n))
            print("actDCF="+str(actDCF))
            print("minDCF="+str(minDCF))