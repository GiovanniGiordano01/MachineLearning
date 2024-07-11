import scipy
import sklearn.datasets
import numpy
import matplotlib
import matplotlib.pyplot as plt

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)  

# Compute a dictionary of ML parameters for each class
def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

# Compute per-class log-densities. We assume classes are labeled from 0 to C-1. The parameters of each class are in hParams (for class i, hParams[i] -> (mean, cov))
def compute_log_likelihood_Gau(D, hParams):
    S = numpy.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
    return S

#i'm using the fast version
def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

# compute log-postorior matrix from log-likelihood matrix and prior array
def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(numpy.log(v_prior))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

# Compute a dictionary of ML parameters for each class - Naive Bayes version of the model
# We compute the full covariance matrix and then extract the diagonal. Efficient implementations would work directly with just the vector of variances (diagonal of the covariance matrix)
def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * numpy.eye(D.shape[0]))
    return hParams

# Compute a dictionary of ML parameters for each class - Tied Gaussian model
# We exploit the fact that the within-class covairance matrix is a weighted mean of the covraince matrices of the different classes
def Gau_Tied_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    hMeans = {}
    CGlobal = 0
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C_class = compute_mu_C(DX)
        CGlobal += C_class * DX.shape[1]
        hMeans[lab] = mu
    CGlobal = CGlobal / D.shape[1]
    for lab in labelSet:
        hParams[lab] = (hMeans[lab], CGlobal)
    return hParams
if __name__ == '__main__':
    #load the iris set
    D, L = load_iris()

    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)     

    #MVG classifier

    #separate DTR in the 3 classes
    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    for lab in [0,1,2]:
        print ('MVG - Class', lab)
        print(hParams_MVG[lab][0])
        print(hParams_MVG[lab][1])
        print()
    #compute ll for each class
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_MVG)

    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(3)/3.)

    PVAL = S_logPost.argmax(0)

    print("MVG - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 
    
    #tied covariance gaussian classifier
    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)
    for lab in [0,1,2]:
        print('Tied Gaussian - Class', lab)
        print(hParams_Tied[lab][0])
        print(hParams_Tied[lab][1])
        print()
        
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_Tied)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(3)/3.)
    PVAL = S_logPost.argmax(0)
    print("Tied Gaussian - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
    #naive bayes gaussian classifier
    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)
    for lab in [0,1,2]:
        print('Naive Bayes Gaussian - Class', lab)
        print(hParams_Naive[lab][0])
        print(hParams_Naive[lab][1])
        print()   

    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_Naive)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(3)/3.)
    PVAL = S_logPost.argmax(0)
    print("Naive Bayes Gaussian - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
        
    #binary classifier
    DIris, LIris = load_iris()
    D = DIris[:, L != 0]
    L = LIris[L != 0]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[2][0], hParams_MVG[2][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1])

    PVAL = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    TH = 0
    PVAL[LLR >= TH] = 2
    PVAL[LLR < TH] = 1
    print("MVG - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
