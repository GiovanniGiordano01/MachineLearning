# Iris setosa will be indicated with value 0, iris versicolor with value 1 and iris virginica with value 2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import scipy.linalg
import sklearn
import sklearn.datasets

DENSITY = True

class PRINTS(Enum):
    Dimensione_1 = 0
    Dimensione_2 = 1
    Dimensione_3 = 2
    Dimensione_4 = 3
    Dimensione_5 = 4
    Dimensione_6 = 5


def main():
    m=6
    D, L = load_dataset('trainData.txt')

    # PCA
    UPCA = PCA(D, m)
    D_pca = np.dot(UPCA.T, D)
    
    
    D0 = D_pca[:, L == 0] #Fake
    D1 = D_pca[:, L == 1] #True
    
    print("*** Istogrammi per PCA ***")
    Histograms(D0, D1, "PCA")


    # LDA
    ULDA = LDA(D, L, 1)
    D_lda=np.dot(ULDA.T, D)
    print(ULDA)
    D0 = D_lda[:, L == 0] #Fake
    D1 = D_lda[:, L == 1] #True
        
    print("*** Istogrammi per LDA ***")
    Histograms(D0, D1, "LDA")


    # Apply LDA as a classifier
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    print("LDA")
    LDA_classifier(DTR, LTR, DVAL, LVAL, 1)

    
    print("LDA+PCA")
    UPCA = PCA(DTR, m)
    D_pca = np.dot(UPCA.T, DTR)
    DVAL_pca = np.dot(UPCA.T, DVAL)
    
    LDA_classifier(D_pca, LTR, DVAL_pca, LVAL, 1)
    

    plt.show()
    return


def LDA_classifier (DTR, LTR, DVAL, LVAL, m):
    
    ULDA = LDA(DTR, LTR, m)
    DTR_lda=np.dot(ULDA.T, DTR)

    if DTR_lda[0, LTR==0].mean() > DTR_lda[0, LTR==1].mean():
        ULDA = -ULDA
        DTR_lda = ULDA.T @ DTR
        
    Histograms(DTR_lda[:, LTR == 0], DTR_lda[:, LTR == 1], "training data")
    
    DVAL_lda = np.dot(ULDA.T, DVAL)
    Histograms(DVAL_lda[:, LVAL == 0], DVAL_lda[:, LVAL == 1], "validation data")
    
    threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0
    print(threshold)
    # threshold = -0.1 #leggermente migliore
    
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0
    
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
    
    return


def load_dataset(filename):
    data = []
    labels = []

    file = open(filename, "r")
    for line in file:

        parts = line.strip().split(',')
        features = list(map(float, parts[:6]))
        data.append(features)
        labels.append(int(parts[6]))
    
    # Converto data and labels (liste) in NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data.T, labels


def Histograms(D0, D1, fig_title):
    fig, axs = plt.subplots(2, 3)  # Create a figure with 2 rows and 3 columns of subplots

    for i, ax in enumerate(axs.flat):
        if i < len(D0):
            ax.hist(D0[i], label='fake', density=DENSITY, alpha=0.5, edgecolor='navy', bins=20)
            ax.hist(D1[i], label='real', density=DENSITY, alpha=0.5, edgecolor='darkorange', bins=20)
            ax.set_title(PRINTS(i).name)
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(fig_title)
    
    
    return


def PCA(D, m):
    return PCA_Projection_Matrix(D, m)  
    

def LDA(D, L, m):

    Sb, Sw = Covariance_matrixes(D, L)

    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
   
    return W

def Covariance_matrixes(D, L):
    mu = D.mean(1).reshape(D.shape[0], 1)
    Sb = 0
    Sw = 0
    for i in [0, 1]:
        Di = D[:, L==i]
        cm = Di.mean(1).reshape(Di.shape[0], 1)
        DC = Di - cm
        C = (DC @ DC.T)
        
        Sb = Sb + Di.shape[1] * np.dot(cm-mu, (cm-mu).T)
        Sw = Sw + C
    
    return Sb / float(D.shape[1]), Sw / float(D.shape[1])



def mcol(v):
    return v.reshape((v.size, 1))

def PCA_Projection_Matrix(D, m):
    
    mu = D.mean(1).reshape(D.shape[0], 1)   # Media
    DC = D - mu                             # Matrice dei dati centrata 
      
    C = (DC @ DC.T) / float(D.shape[1])     # Matrice della covarianza
    s, U = np.linalg.eigh(C)                # s:autovalori, U:autovettori
    P = U[:, ::-1][:, 0:m]                  
    
    return P

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)


if __name__ == "__main__":
    main()
