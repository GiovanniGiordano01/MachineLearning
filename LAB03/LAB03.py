import numpy
import pca
import lda
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


def load2():

    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def PCA_Projection_Matrix(D, m):
    
    mu = D.mean(1).reshape(D.shape[0], 1)   # Media
    DC = D - mu                             # Matrice dei dati centrata 
      
    C = (DC @ DC.T) / float(D.shape[1])     # Matrice della covarianza
    s, U = numpy.linalg.eigh(C)                # s:autovalori, U:autovettori
    P = U[:, ::-1][:, 0:m]                  
    s = s.reshape(s.size, 1)
    
    DP = numpy.dot(P.T, D)                     # Projection matrix
    
    return DP, P

def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    hFea = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',    
        }

    plt.figure()
    plt.xlabel(hFea[0])
    plt.hist(D0[0, :], bins = 10, density = True, alpha = 0.4, label = 'setosa')
    plt.hist(D1[0, :], bins = 10, density = True, alpha = 0.4, label = 'versicolor')
    plt.hist(D2[0, :], bins = 10, density = True, alpha = 0.4, label = 'virginica')
        
    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    #plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()

def plot_hist_classification(D, L):

    D1 = D[:, L==1]
    D2 = D[:, L==2]


    plt.figure()
    plt.hist(D1[0, :], bins = 5, density = True, alpha = 0.4, label = 'versicolor', color='orange')
    plt.hist(D2[0, :], bins = 5, density = True, alpha = 0.4, label = 'virginica', color='green')
        
    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    #plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()

def PCA_plots(DP, L):
    DP0 = DP[:, L == 0]

    DP1 = DP[:, L == 1]

    DP2 = DP[:, L == 2]
    
    plt.figure(0)
    plt.scatter(DP0[0, :], DP0[1, :], label='setosa')
    plt.scatter(DP1[0, :], DP1[1, :], label='versicolor')
    plt.scatter(DP2[0, :], DP2[1, :], label='virginica')
    plt.legend()

    # plt.figure(1)
    # plt.hist(DP0[0], label='setosa', density=DENSITY, alpha=0.5, edgecolor='navy', bins=5)
    # plt.hist(DP1[0], label='versicolor', density=DENSITY, alpha=0.5, edgecolor='darkorange', bins=5)
    # plt.hist(DP2[0], label='virginica', density=DENSITY, alpha=0.5, edgecolor='darkgreen', bins=5)
    # plt.legend()
    
    return


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

if __name__ == '__main__':
    #PCA
    
    D, L = load2() #load dei dati iris
    DP,P=PCA_Projection_Matrix(D,m=2)
    #PCA_plots(DP,L)
    #plot_hist(DP,L)

    #LDA
    W = compute_lda_geig(D, L, m = 1)
    DW=numpy.dot(W.T,D)
    #print(DW)
    #plot_hist(DW,L)

    #classification

    DIris, LIris = load2()
    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    ULDA = lda.compute_lda_geig(DTR, LTR, m=1)
    DTR_lda = lda.apply_lda(ULDA, DTR)

    # Check if the Virginica class samples are, on average, on the right of the Versicolor samples on the training set. If not, we reverse ULDA and re-apply the transformation.
    if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==2].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR)

    DVAL_lda  = lda.apply_lda(ULDA, DVAL)

    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0 # Estimated only on model training data

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))

    # Solution with PCA pre-processing with dimension m.
    m = 3
    UPCA = pca.compute_pca(DTR, m = m) # Estimated only on model training data
    DTR_pca = pca.apply_pca(UPCA, DTR)   # Applied to original model training data
    DVAL_pca = pca.apply_pca(UPCA, DVAL) # Applied to original validation data

    ULDA = lda.compute_lda_JointDiag(DTR_pca, LTR, m = 1) # Estimated only on model training data, after PCA has been applied

    DTR_lda = lda.apply_lda(ULDA, DTR_pca)   # Applied to PCA-transformed model training data, the projected training samples are required to check the orientation of the direction and to compute the threshold
    # Check if the Virginica class samples are, on average, on the right of the Versicolor samples on the training set. If not, we reverse ULDA and re-apply the transformation
    if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==2].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR_pca)

    DVAL_lda = lda.apply_lda(ULDA, DVAL_pca) # Applied to PCA-transformed validation data

    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0 # Estimated only on model training data

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))


    
