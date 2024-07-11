import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg

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

def apply_pca(P,D):
    return numpy.dot(P.T, D)

def apply_lda(W,D):
    return numpy.dot(W.T,D)

def Histograms(DP, fig_title):
    D0 = DP[:, L==0]
    D1 = DP[:, L==1]
    fig, axs = plt.subplots(2, 3)  # Create a figure with 2 rows and 3 columns of subplots
    for i, ax in enumerate(axs.flat):
        if i < len(D0):
            ax.hist(D0[i], label='fake', density=True, alpha=0.5, edgecolor='navy', bins=20)
            ax.hist(D1[i], label='real', density=True, alpha=0.5, edgecolor='darkorange', bins=20)
            ax.set_title("m="+str(i+1))
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(fig_title)
    plt.show()
    return

def plot_histogram(DP, L):

    D0 = DP[:, L==0]
    D1 = DP[:, L==1]


    plt.figure()
    plt.hist(D0[0, :], bins = 20, density = True, alpha = 0.5, edgecolor="navy",label = 'true')
    plt.hist(D1[0, :], bins = 20, density = True, alpha = 0.5,edgecolor="darkorange", label = 'fake')
    
    plt.title("LDA")       
    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.show()


def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in numpy.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L, m):
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

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
    mu = vcol(D.mean(1)) #compute the mean of the columns (axis 0 = row, axis 1 = columns)
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pcaD(D,m):
    mu, C = compute_mu_C(D)
    s,U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return P



if __name__ == '__main__':
    m=0
    #PCA
    D, L = load('trainData.txt') #load dei dati sulle impronte digitali
    P=compute_pcaD(D,m=6)
    DP=apply_pca(P,D)
    #Histograms(DP, "PCA")
        
    #LDA
    W = compute_lda_geig(D, L, m = 1)
    DW=apply_lda(W,D)
    #plot_histogram(DW,L)

    #LDA used as classification
    D, L = load('trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    ULDA = compute_lda_geig(DTR, LTR, m = 1)
    DTR_lda=apply_lda(ULDA,DTR)

    if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==0].mean():
        ULDA = -ULDA
        DTR_lda = apply_lda(ULDA, DTR)
    
    DVAL_lda  = apply_lda(ULDA, DVAL)

    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==0].mean()) / 2.0 # Estimated only on model training data
    print("threshold:"+str(threshold))
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 0
    PVAL[DVAL_lda[0] < threshold] = 1
    print('----------LDA without preprocessing----------')
    #print('Labels:     ', LVAL)
    #print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))

    # PCA pre-processing with dimension m.
    for m in range(1,6):
        UPCA = compute_pcaD(DTR, m = m) # Estimated only on model training data
        DTR_pca = apply_pca(UPCA, DTR)   # Applied to original model training data
        DVAL_pca = apply_pca(UPCA, DVAL) # Applied to original validation data

        ULDA = compute_lda_geig(DTR_pca, LTR, m = 1) # Estimated only on model training data, after PCA has been applied

        DTR_lda = apply_lda(ULDA, DTR_pca)   # Applied to PCA-transformed model training data, the projected training samples are required to check the orientation of the direction and to compute the threshold

        if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==0].mean():
            ULDA = -ULDA
            DTR_lda = apply_lda(ULDA, DTR_pca)

        DVAL_lda = apply_lda(ULDA, DVAL_pca) # Applied to PCA-transformed validation data

        threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==0].mean()) / 2.0 # Estimated only on model training data

        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL[DVAL_lda[0] >= threshold] = 0
        PVAL[DVAL_lda[0] < threshold] = 1
        print('----------LDA with PCA pre-processing, using m='+str(m)+'----------')
        #print('Labels:     ', LVAL)
        #print('Predictions:', PVAL)
        print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
        print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
