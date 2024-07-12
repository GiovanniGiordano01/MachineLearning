import numpy
import scipy.special
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

def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * numpy.eye(D.shape[0]))
    return hParams

# compute log-postorior matrix from log-likelihood matrix and prior array
def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(numpy.log(v_prior))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost

def MVG_classifier_LLR(DTR,LTR,DVAL):
    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])
    return LLR
def tied_gaussian_classifier_LLR(DTR,LTR,DVAL):
    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)     
    LLR = logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1]) - logpdf_GAU_ND(DVAL, hParams_Tied[0][0], hParams_Tied[0][1])
    return LLR
def naive_bayes_classifier_LLR(DTR,LTR,DVAL):
    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR) 
    LLR = logpdf_GAU_ND(DVAL, hParams_Naive[1][0], hParams_Naive[1][1]) - logpdf_GAU_ND(DVAL, hParams_Naive[0][0], hParams_Naive[0][1])
    return LLR
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

# Optimal Bayes decisions for binary tasks with log-likelihood-ratio scores
def compute_optimal_Bayes(llr, prior, Cfn, Cfp):
    th = -numpy.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return numpy.int32(llr > th)

# Specialized function for binary problems (empirical_Bayes_risk is also called DCF or actDCF)
def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

# Auxiliary function, returns all combinations of Pfp, Pfn corresponding to all possible thresholds
# We do not consider -inf as threshld, since we use as assignment llr > th, so the left-most score corresponds to all samples assigned to class 1 already
def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs
    classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0 # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate

        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = numpy.concatenate([llrSorted, numpy.array([numpy.inf])])
    
    return numpy.array(Pfn), numpy.array(Pfp), llrSorted # we return also the corresponding thresholds

# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
def compute_minDCF(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / numpy.minimum(prior * Cfn, (1-prior)*Cfp) # We exploit broadcasting to compute all DCFs for all thresholds
    idx = numpy.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]
# Multiclass solution that works also for binary problems
def compute_empirical_Bayes_risk(predictedLabels, classLabels, prior_array, costMatrix, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    errorRates = M / vrow(M.sum(0))
    bayesError = ((errorRates * costMatrix).sum(0) * prior_array.ravel()).sum()
    if normalize:
        return bayesError / numpy.min(costMatrix @ vcol(prior_array))
    return bayesError
def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pcaD(D,m):
    mu, C = compute_mu_C(D)
    s,U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return P

def apply_pca(P,D):
    return numpy.dot(P.T, D)

def compute_bayes_risk(LLRMVG,LLRtied,LLRnaive):
    for prior, Cfn, Cfp in [ (0.5, 1, 1), (0.9, 1, 1), (0.1, 1, 1) ]:
        print("-"*40)
        print('Bayes risk for application set (',prior,',',Cfn,',',Cfp,')')
        predictions_binary = compute_optimal_Bayes(LLRMVG, prior, Cfn, Cfp)
        #print(compute_confusion_matrix(predictions_binary, LVAL))
        print('-MVG DCF (normalized): %.5f' % (compute_empirical_Bayes_risk_binary(predictions_binary, LVAL, prior, Cfn, Cfp)))
        minDCF, minDCFThreshold = compute_minDCF(LLRMVG, LVAL, prior, Cfn, Cfp, returnThreshold=True)
        print('-MVG MinDCF (normalized, fast): %.5f (@ th = %e)' % (minDCF, minDCFThreshold))
        predictions_binary = compute_optimal_Bayes(LLRtied, prior, Cfn, Cfp)
        #print(compute_confusion_matrix(predictions_binary, LVAL))
        print('-tied DCF (normalized): %.5f' % (compute_empirical_Bayes_risk_binary(predictions_binary, LVAL, prior, Cfn, Cfp)))
        minDCF, minDCFThreshold = compute_minDCF(LLRtied, LVAL, prior, Cfn, Cfp, returnThreshold=True)
        print('-tied MinDCF (normalized, fast): %.5f (@ th = %e)' % (minDCF, minDCFThreshold))
        predictions_binary = compute_optimal_Bayes(LLRnaive, prior, Cfn, Cfp)
        #print(compute_confusion_matrix(predictions_binary, LVAL))
        print('-naive DCF (normalized): %.5f' % (compute_empirical_Bayes_risk_binary(predictions_binary, LVAL, prior, Cfn, Cfp)))
        minDCF, minDCFThreshold = compute_minDCF(LLRnaive, LVAL, prior, Cfn, Cfp, returnThreshold=True)
        print('-naive MinDCF (normalized, fast): %.5f (@ th = %e)' % (minDCF, minDCFThreshold))

if __name__ == '__main__':
    #first of all,let's compute the effective prior for the last 2 applications
    effective_prior4=(0.5*1)/(0.5*1+(1-0.5)*9.0)
    effective_prior5=(0.5*9)/(0.5*9+(1-0.5)*1)
    print("effective prior application 4:",effective_prior4)
    print("effective prior application 5:",effective_prior5)
    #load tour data sets
    D, L = load('trainData.txt')
    # # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L) 

    #MVG classifier
    LLRMVG=MVG_classifier_LLR(DTR,LTR,DVAL)

    #tied gaussian model classifier
    LLRtied=tied_gaussian_classifier_LLR(DTR,LTR,DVAL)

    #naive bayes gaussian classifier
    LLRnaive=naive_bayes_classifier_LLR(DTR,LTR,DVAL)

    #first we try without pca pre-processing
    compute_bayes_risk(LLRMVG,LLRtied,LLRnaive)
    #now let's try with PCA pre processing
    for i in range(1,6):
        UPCA = compute_pcaD(DTR, m = i) # Estimated only on model training data
        DTR_pca = apply_pca(UPCA, DTR)   # Applied to original model training data
        DVAL_pca = apply_pca(UPCA, DVAL) # Applied to original validation data

        LLRMVG=MVG_classifier_LLR(DTR_pca,LTR,DVAL_pca)
        LLRtied=tied_gaussian_classifier_LLR(DTR_pca,LTR,DVAL_pca)
        LLRnaive=naive_bayes_classifier_LLR(DTR_pca,LTR,DVAL_pca)
        print('-'*20,'PCA m=',i,'-'*20)
        compute_bayes_risk(LLRMVG,LLRtied,LLRnaive)
    #best PCA: m=5
    UPCA = compute_pcaD(DTR, m = 5) # Estimated only on model training data
    DTR_pca = apply_pca(UPCA, DTR)   # Applied to original model training data
    DVAL_pca = apply_pca(UPCA, DVAL) # Applied to original validation data

    LLRMVG=MVG_classifier_LLR(DTR_pca,LTR,DVAL_pca)
    LLRtied=tied_gaussian_classifier_LLR(DTR_pca,LTR,DVAL_pca)
    LLRnaive=naive_bayes_classifier_LLR(DTR_pca,LTR,DVAL_pca)

    # Bayes error plot MVG cassifier
    effPriorLogOdds = numpy.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
        commedia_predictions_binary = compute_optimal_Bayes(LLRMVG, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(commedia_predictions_binary, LVAL, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF(LLRMVG, LVAL, effPrior, 1.0, 1.0))
    matplotlib.pyplot.figure("MVG classifier - DCF vs minDCF")
    plt.title("MVG classifier")
    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='DCF', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    plt.xlabel("prior log odds")
    plt.ylabel("DCF")
    plt.grid()
    matplotlib.pyplot.ylim([0, 1.1])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

    # Bayes error plot tied classifier
    effPriorLogOdds = numpy.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
        commedia_predictions_binary = compute_optimal_Bayes(LLRtied, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(commedia_predictions_binary, LVAL, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF(LLRtied, LVAL, effPrior, 1.0, 1.0))
    matplotlib.pyplot.figure("tied classifier - DCF vs minDCF")
    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='DCF', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    matplotlib.pyplot.ylim([0, 1.1])
    plt.title("naive Bayes classifier")
    plt.xlabel("prior log odds")
    plt.ylabel("DCF")
    plt.grid()
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

    # Bayes error plot naive cassifier
    effPriorLogOdds = numpy.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
        commedia_predictions_binary = compute_optimal_Bayes(LLRnaive, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(commedia_predictions_binary, LVAL, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF(LLRnaive, LVAL, effPrior, 1.0, 1.0))
    matplotlib.pyplot.figure("naive classifier - DCF vs minDCF")
    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='DCF', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    plt.title("tied covariance classifier")
    plt.xlabel("prior log odds")
    plt.ylabel("DCF")
    plt.grid()
    matplotlib.pyplot.ylim([0, 1.1])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
