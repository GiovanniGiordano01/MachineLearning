import numpy
import scipy.special
import matplotlib
import matplotlib.pyplot
import sklearn.datasets

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

# compute matrix of posteriors from class-conditional log-likelihoods (each column represents a sample) and prior array
def compute_posteriors(log_clas_conditional_ll, prior_array):
    logJoint = log_clas_conditional_ll + vcol(numpy.log(prior_array))
    logPost = logJoint - scipy.special.logsumexp(logJoint, 0)
    return numpy.exp(logPost)

# Compute optimal Bayes decisions for the matrix of class posterior (each column refers to a sample)
def compute_optimal_Bayes(posterior, costMatrix):
    expectedCosts = costMatrix @ posterior
    return numpy.argmin(expectedCosts, 0)

# Assume that classes are labeled 0, 1, 2 ... (nClasses - 1)
def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

# Build uniform cost matrix with cost 1 for all kinds of error, and cost 0 for correct assignments
def uniform_cost_matrix(nClasses):
    return numpy.ones((nClasses, nClasses)) - numpy.eye(nClasses)

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

# Compute a dictionary of ML parameters for each class
def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

def compute_log_likelihood_Gau(D, hParams):
    S = numpy.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
    return S

#i'm using the fast version
def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

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
    #confusion matrixes of IRIS dataset
    #load the iris set
    D, L = load("trainData.txt")
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L) 
    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_MVG)
    IRIS_posteriors=compute_posteriors(S_logLikelihood,numpy.ones(2)/2.0)
    IRIS_predictions = compute_optimal_Bayes(IRIS_posteriors, uniform_cost_matrix(2))
    print("---confusion matrix trainData MVG classifier---")
    print( compute_confusion_matrix(IRIS_predictions, LVAL))

    hParams_tied = Gau_Tied_ML_estimates(DTR, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_tied)
    IRIS_posteriors=compute_posteriors(S_logLikelihood,numpy.ones(2)/2.0)
    IRIS_predictions = compute_optimal_Bayes(IRIS_posteriors, uniform_cost_matrix(2))
    print("---confusion matrix trainData tied covariance classifier---")
    print( compute_confusion_matrix(IRIS_predictions, LVAL))