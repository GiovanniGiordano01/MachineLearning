############################################################################################
# Copyright (C) 2024 by Sandro Cumani                                                      #
#                                                                                          #
# This file is provided for didactic purposes only, according to the Politecnico di Torino #
# policies on didactic material.                                                           #
#                                                                                          #
# Any form of re-distribution or online publication is forbidden.                          #
#                                                                                          #
# This file is provided as-is, without any warranty                                        #
############################################################################################

import numpy

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

# Compute log-density for a single sample x (column vector). The result is a 1-D array with 1 element
def logpdf_GAU_ND_singleSample(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu).T @ P @ (x-mu)).ravel()

# Compute log-densities for N samples, arranged as a MxN matrix X (N stacked column vectors). The result is a 1-D array with N elements, corresponding to the N log-densities
def logpdf_GAU_ND_slow(X, mu, C):
    ll = [logpdf_GAU_ND_singleSample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return numpy.array(ll).ravel()


# Compute log-densities for N samples, arranged as a MxN matrix X (N stacked column vectors). The result is a 1-D array with N elements, corresponding to the N log-densities
def logpdf_GAU_ND_fast(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

logpdf_GAU_ND = logpdf_GAU_ND_slow

def compute_ll(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

if __name__ == '__main__':

    # Plot
    import matplotlib.pyplot as plt
    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    m = numpy.ones((1,1)) * 1.0
    C = numpy.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    #plt.show()

    # Check pdf - we check both the fast and slow functions
    pdfSol = numpy.load('llGAU.npy')
    pdfGau = logpdf_GAU_ND_fast(vrow(XPlot), m, C)
    print (numpy.abs(pdfSol - pdfGau).max())
    pdfGau = logpdf_GAU_ND_slow(vrow(XPlot), m, C)
    print (numpy.abs(pdfSol - pdfGau).max())

    # Check multi-dimensional pdf - we check both the fast and slow functions    
    XND = numpy.load('XND.npy')
    mu = numpy.load('muND.npy')
    C = numpy.load('CND.npy')

    pdfSol = numpy.load('llND.npy')
    pdfGau = logpdf_GAU_ND_fast(XND, mu, C)
    print(numpy.abs(pdfSol - pdfGau).max())
    pdfGau = logpdf_GAU_ND_slow(XND, mu, C)
    print(numpy.abs(pdfSol - pdfGau).max())

    # ML estimates - XND
    m_ML, C_ML = compute_mu_C(XND)
    print(m_ML)
    print(C_ML)
    print(compute_ll(XND, m_ML, C_ML))
    
    # ML estimates - X1D
    X1D = numpy.load('X1D.npy')
    m_ML, C_ML = compute_mu_C(X1D)
    print(m_ML)
    print(C_ML)

    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m_ML, C_ML)))
    #plt.show()

    print(compute_ll(X1D, m_ML, C_ML))
    #Trying other values
    print(compute_ll(X1D, numpy.array([[1.0]]), numpy.array([[2.0]])))
    print(compute_ll(X1D, numpy.array([[0.0]]), numpy.array([[1.0]])))
    print(compute_ll(X1D, numpy.array([[2.0]]), numpy.array([[6.0]]))) # Values close to ML estimates
    
