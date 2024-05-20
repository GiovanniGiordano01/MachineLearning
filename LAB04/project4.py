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

#i'm using the fast version
def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

def compute_ll(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

if __name__ == '__main__':

    # ML estimates 
    D,L = load('trainData.txt')
    Dreal  = D[:, L == 1]
    Dfake = D[:, L == 0]

    for i in range (6):
        m_MLreal, C_MLreal = compute_mu_C(vrow(Dreal[i,:]))
        m_MLfake, C_MLfake = compute_mu_C(vrow(Dfake[i,:]))
       
        plt.figure('dimension' + str(i+1))
        plt.title("dimension "+str(i+1))
        plt.hist(Dreal[i,:].ravel(), bins=50, density=True, alpha = 0.4,label = 'real')    
        plt.hist(Dfake[i,:].ravel(), bins=50, density=True, alpha = 0.4,label = 'fake')
        XPlot = numpy.linspace(-4, 4, 1000)
        plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m_MLreal, C_MLreal)), color='Blue')
        plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m_MLfake, C_MLfake)), color='Orange')
        plt.legend()
        plt.show()

