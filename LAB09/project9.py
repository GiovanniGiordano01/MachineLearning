import numpy
import bayesRisk
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
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

# Optimize SVM
def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    
    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1])) * K])
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    
    # Primal loss
    def primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * numpy.linalg.norm(w_hat)**2 + C * numpy.maximum(0, 1 - ZTR * S).sum()

    # Compute primal solution for extended data matrix
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
    # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w'x + b * K

    primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
    print ('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, primalLoss, dualLoss, primalLoss - dualLoss))
    
    return w, b

# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

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
def plot_bayes_plot(actDCF,minDCF,msg):
    plt.figure("SVM - DCF as a function of C")
    plt.xscale('log', base=10)
    plt.plot(numpy.logspace(-5, 0, 11), actDCF, label="actDCF", color='r', marker= 'o')  
    plt.plot(numpy.logspace(-5, 0, 11), minDCF, label="minDCF", color='b', marker= 'o')   
    plt.title(msg)
    plt.grid()
    plt.xlabel("logC")
    plt.ylabel("DCF")
    #matplotlib.pyplot.ylim([0, 1.1])
    plt.legend()
    plt.show()

def plot_bayes_plot_rbf(actDCF,minDCF):
    plt.figure("SVM - DCF as a function of C - rbf kernel")
    plt.xscale('log', base=10)
    plt.plot(numpy.logspace(-3, 2, 11), actDCF[0], label="actDCF - gamma=e^(-4)", color='r', marker= 'o')  
    plt.plot(numpy.logspace(-3, 2, 11), minDCF[0], label="minDCF - gamma=e^(-4)", color='b', marker= 'o')   
    plt.plot(numpy.logspace(-3, 2, 11), actDCF[1], label="actDCF - gamma=e^(-3)", color='g', marker= 'o')  
    plt.plot(numpy.logspace(-3, 2, 11), minDCF[1], label="minDCF - gamma=e^(-3)", color='c', marker= 'o') 
    plt.plot(numpy.logspace(-3, 2, 11), actDCF[2], label="actDCF - gamma=e^(-2)", color='y', marker= 'o')  
    plt.plot(numpy.logspace(-3, 2, 11), minDCF[2], label="minDCF - gamma=e^(-2)", color='m', marker= 'o') 
    plt.plot(numpy.logspace(-3, 2, 11), actDCF[3], label="actDCF - gamma=e^(-1)", color='k', marker= 'o')  
    plt.plot(numpy.logspace(-3, 2, 11), minDCF[3], label="minDCF - gamma=e^(-1)", color='b', marker= 'o') 
    plt.title("DCF as a function of C - rbf kernel")
    plt.grid()
    plt.xlabel("logC")
    plt.ylabel("DCF")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    D, L = load("trainData.txt")
    #we take a fraction of our data!
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    # SVM takes time to train: let's reduce our data
    #DTR=DTR[:, ::50]
    #LTR=LTR[::50]
    K=1
    actDCF = []
    minDCF = []
    for C in numpy.logspace(-5, 0, 11):
        w, b = train_dual_SVM_linear(DTR, LTR, C, K)
        SVAL = (vrow(w) @ DVAL + b).ravel()
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0))
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0))
    plot_bayes_plot(actDCF,minDCF,"SVM - DCF as a funtion of C")

    #let's repeat with centered data
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    #DTR=DTR[:, ::50]
    #LTR=LTR[::50]
    mu=DTR.mean(1) #we compute the mean for every dimension
    DC=DTR-mu.reshape((mu.size,1))#we center the dataset
    DVALC=DVAL-mu.reshape((mu.size,1))#we center the dataset
    actDCF = []
    minDCF = []
    for C in numpy.logspace(-5, 0, 11):
        w, b = train_dual_SVM_linear(DC, LTR, C, K)
        SVAL = (vrow(w) @ DVALC + b).ravel()
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0))
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0))
    plot_bayes_plot(actDCF,minDCF,"SVM with centered data - DCF as a funtion of C")

    actDCF = []
    minDCF = []
    kernelFunc=polyKernel(2, 1)
    for C in numpy.logspace(-5, 0, 11):
        fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc)
        SVAL = fScore(DVAL)
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0))
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0))
    plot_bayes_plot(actDCF,minDCF,"SVM with polynomial kernel - DCF as a funtion of C")

    actDCFcollection=[]
    minDCFcollection=[]
    #let's try with RBF kernel function
    # DTR=DTR[:, ::50]
    # LTR=LTR[::50]
    for gamma in numpy.exp([-1,-2,-3,-4]):
        actDCF = []
        minDCF = []
        kernelFunc=rbfKernel(gamma)
        for C in numpy.logspace(-3, 2, 11):
            fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc)
            SVAL = fScore(DVAL)
            minDCF.append(bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0))
            actDCF.append(bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0))
        actDCFcollection.append(actDCF)
        minDCFcollection.append(minDCF)
plot_bayes_plot_rbf(actDCFcollection,minDCFcollection)
