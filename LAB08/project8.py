import bayesRisk
import numpy
import scipy.special
import matplotlib
import matplotlib.pyplot

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

# Optimize the weighted logistic regression loss
def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTrue = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wFalse = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b #compute the score

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTrue # Apply the weights to the loss computations
        loss[ZTR<0] *= wFalse

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTrue # Apply the weights to the gradient computations
        G[ZTR < 0] *= wFalse
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

def plot_bayes_plot(dcf,msg):
    matplotlib.pyplot.figure("Logistic regression - lambda and "+msg+" visualized")
    matplotlib.pyplot.xscale('log', base=10)
    matplotlib.pyplot.plot(numpy.logspace(-4, 2, 13), dcf, label=msg, color='r', marker= 'o')      
    #matplotlib.pyplot.ylim([0, 1.1])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
if __name__ == '__main__':
    #load our data set
    D, L = load('trainData.txt')
    # # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L) 

    actDCF = []
    minDCF = []
    for lamb in numpy.logspace(-4, 2, 13):
        w, b = trainLogRegBinary(DTR, LTR, lamb) # Train model
        sVal = numpy.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        # Compute LLR-like scores - remove the empirical prior
        sValLLR = sVal - numpy.log(pEmp / (1-pEmp))
        # Compute optimal decisions for the prior 0.1
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0))
        #print ('minDCF - pT = 0.5: %.4f' % bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0))
        #print ('actDCF - pT = 0.5: %.4f' % bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0))

    plot_bayes_plot(actDCF,"actDCF")
    plot_bayes_plot(minDCF,"minDCF")

    #too many samples: we reduce the dataset by taking only 1 out of 50 samples
    DTRreduced=DTR[:, ::50]
    LTRreduced=LTR[::50]
    #now we repeat the logistic regression classification with the reduced training sets

    actDCF = []
    minDCF = []
    for lamb in numpy.logspace(-4, 2, 13):
        w, b = trainLogRegBinary(DTRreduced, LTRreduced, lamb) # Train model
        sVal = numpy.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        # Compute empirical prior
        pEmp = (LTRreduced == 1).sum() / LTRreduced.size
        # Compute LLR-like scores - remove the empirical prior
        sValLLR = sVal - numpy.log(pEmp / (1-pEmp))
        # Compute optimal decisions for the prior 0.1
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0))
        #print ('minDCF - pT = 0.5: %.4f' % bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0))
        #print ('actDCF - pT = 0.5: %.4f' % bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0))

    plot_bayes_plot(actDCF,"actDCF - reduced sample set")
    plot_bayes_plot(minDCF,"minDCF - reduced sample set")

    #return to the entire data set: prior weighted algorithm
    actDCF = []
    minDCF = []
    for lamb in numpy.logspace(-4, 2, 13):
        pT = 0.1
        w, b = trainWeightedLogRegBinary(DTR, LTR, lamb, pT = pT) # Train model to print the loss
        sVal = numpy.dot(w.T, DVAL) + b
        sValLLR = sVal - numpy.log(pT / (1-pT))
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0))

    plot_bayes_plot(actDCF,"actDCF - prior weighted LR")
    plot_bayes_plot(minDCF,"minDCF - prior weighted LR")

