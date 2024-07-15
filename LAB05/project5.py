import scipy
import sklearn.datasets
import numpy
import matplotlib
import matplotlib.pyplot as plt
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

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

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

#i'm using the fast version
def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

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

# compute log-postorior matrix from log-likelihood matrix and prior array
def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(numpy.log(v_prior))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost

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

def MVG_classifier(DTR,LTR,DVAL,msg):
    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])
    PVAL = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print(msg + "%.2f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))

def tied_gaussian_classifier(DTR,LTR,DVAL,msg):
    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)     
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_Tied)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(2)/2.)
    PVAL = S_logPost.argmax(0)
    print(msg +"%.2f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))

def naive_bayes_classifier(DTR,LTR,DVAL,msg):
    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR) 
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_Naive)
    S_logPost = compute_logPosterior(S_logLikelihood, numpy.ones(2)/2.)
    PVAL = S_logPost.argmax(0)
    print(msg + "%.2f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))

def compute_pcaD(D,m):
    mu, C = compute_mu_C(D)
    s,U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return P

def apply_pca(P,D):
    return numpy.dot(P.T, D)

if __name__ == '__main__':
    D,L=load('trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    #MVG classifier
    MVG_classifier(DTR,LTR,DVAL,"MVG cassifier - Error rate: ")

    #tied gaussian model classifier
    tied_gaussian_classifier(DTR,LTR,DVAL,"Tied gaussian - Error rate: ")

    #naive bayes gaussian classifier
    naive_bayes_classifier(DTR,LTR,DVAL,"Naive Bayes Gaussian - Error rate: ")

    #data from previous projects
    print("LDA classifier - Error rate: 9.3%")
    print("LDA classifier with PCA pre processing - Error rate: 9.2%")
    
    #covariance matrixes
    Utrue,Ctrue=compute_mu_C(DTR[:,LTR==1])
    Ufake,Cfake=compute_mu_C(DTR[:,LTR==0])
    numpy.set_printoptions(linewidth=120)
    print("Covariance matrix true:")
    print(Ctrue)
    print("Covariance matrix fake:")
    print(Cfake)

    CorrTrue = Ctrue / ( vcol(Ctrue.diagonal()**0.5) * vrow(Ctrue.diagonal()**0.5) )
    CorrFake = Cfake / ( vcol(Cfake.diagonal()**0.5) * vrow(Cfake.diagonal()**0.5) )
    print("Pearson correlation for class true:")
    print(CorrTrue)
    print("Pearson correlation for class fake:")
    print(CorrFake)

    #gaussian models does not represent features 5 and 6 well enough
    #we should try to eliminate them and see if the classifier is more accurate
    D,L=load('trainData.txt')
    #print(D)
    Dnew=D[[0,1,2,3],:]    #print(Dnew)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(Dnew, L)

    #MVG classifier with only the first 4 features
    MVG_classifier(DTR,LTR,DVAL,"MVG classifier with 4 features - Error rate: ")

    #tied gaussian model classifier
    tied_gaussian_classifier(DTR,LTR,DVAL,"Tied gaussian with 4 features - Error rate: ")

    #naive bayes gaussian classifier
    naive_bayes_classifier(DTR,LTR,DVAL,"Naive Bayes Gaussian with 4 features - Error rate: ")

    #only feature 1-2
    D,L=load('trainData.txt')
    #print(D)
    Dnew=D[[0,1],:]
    #print(Dnew)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(Dnew, L)

    #MVG classifier with only the features 1-2
    MVG_classifier(DTR,LTR,DVAL,"MVG classifier with feature 1-2 - Error rate: ")

    #tied gaussian model classifier with only the features 1-2
    tied_gaussian_classifier(DTR,LTR,DVAL,"Tied gaussian with feature 1-2 - Error rate: ")

    #naive bayes gaussian classifier
    naive_bayes_classifier(DTR,LTR,DVAL,"Naive Bayes Gaussian with feature 1-2 - Error rate: ")

    #only feature 3-4
    D,L=load('trainData.txt')
    #print(D)
    Dnew=D[[2,3],:]
    #print(Dnew)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(Dnew, L)

    #MVG classifier with only the features 1-2
    MVG_classifier(DTR,LTR,DVAL,"MVG classifier with feature 3-4 - Error rate: ")

    #tied gaussian model classifier with only the features 1-2
    tied_gaussian_classifier(DTR,LTR,DVAL,"Tied gaussian with feature 3-4 - Error rate: ")

    #naive bayes gaussian classifier
    naive_bayes_classifier(DTR,LTR,DVAL,"Naive Bayes Gaussian with feature 3-4 - Error rate: ")

    D,L=load('trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    # PCA pre-processing with dimension m.
    for i in range(6):
        UPCA = compute_pcaD(DTR, m = i) # Estimated only on model training data
        DTR_pca = apply_pca(UPCA, DTR)   # Applied to original model training data
        DVAL_pca = apply_pca(UPCA, DVAL) # Applied to original validation data
        print("---------PCA (m=" + str(i) +")--------------")
        #MVG classifier with PCA pre processing
        MVG_classifier(DTR_pca,LTR,DVAL_pca,"MVG classifier with PCA (m=" + str(i) + ") pre processing - Error rate: ")

        #tied gaussian model classifier with PCA pre processing
        tied_gaussian_classifier(DTR_pca,LTR,DVAL_pca,"Tied gaussian with PCA (m=" + str(i) + ") pre processing - Error rate: ")

        #naive bayes gaussian classifier with PCA pre processing
        naive_bayes_classifier(DTR_pca,LTR,DVAL_pca,"Naive Bayes Gaussian with PCA (m=" + str(i) + ") pre processing - Error rate: ")
    


    
    