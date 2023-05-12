
import numpy
import matplotlib.pyplot as plt
from test_snakeML.numpy_transformations import mrow, mcol
import math

def logpdf_GAU_ND_unoptimized(x, mu, C):
    C_inv=numpy.linalg.inv(C)
    det = numpy.linalg.slogdet(C)[1]
    M=x.shape[0]
    log_pi=math.log(2*math.pi)
    ab=(-M*log_pi-det)/2
    result=[]
    for i in range(x.shape[1]):
        x_mu=numpy.subtract(mcol(x[:,i]),mu)
        c_p=numpy.dot(x_mu.T,C_inv)
        c_d=numpy.dot(c_p,x_mu)
        c=-c_d/2
        y=numpy.add(ab,c)[0][0]
        result.append(y)
    return numpy.array(result)

def logpdf_GAU_ND(x, mu, C):
    C_inv=numpy.linalg.inv(C)
    det = numpy.linalg.slogdet(C)[1]
    M=x.shape[0]
    log_pi=math.log(2*math.pi)
    x_mu=numpy.subtract(x,mu)
    r1=numpy.dot(x_mu.T,C_inv)
    r2=numpy.diagonal(numpy.dot(r1,x_mu))
    result=(-M*log_pi-det-r2)/2
    return result

def logpdf_GAU_ND_error(Solution, Result):
    print("Error: ",numpy.abs(Solution - Result).max())
 
def logpdf_GAU_ND_visualization(Data, result):
    plt.figure()
    plt.plot(Data.ravel(), numpy.exp(result))
    plt.show()

def loglikelihood_visualization(Data, XPlot, Result):
    plt.figure()
    plt.hist(Data.ravel(), bins=50, density=True)
    plt.plot(XPlot.ravel(), numpy.exp(Result))
    plt.show()

def loglikelihood(x, m_ML=False, C_ML=False, visualize=numpy.array([])):
    if (not m_ML and not C_ML):
        mu=mcol(numpy.mean(x,axis=1))
        c=numpy.cov(x)
        if c.size==1:
            c=numpy.reshape(numpy.array(c),(c.size,-1))
        gau=logpdf_GAU_ND(x,mu,c)
        if visualize.size>0:
            loglikelihood_visualization(x, visualize, logpdf_GAU_ND(mrow(visualize),mu,c))
    else:
        gau=logpdf_GAU_ND(x,m_ML,C_ML)
        if visualize.size>0:
            loglikelihood_visualization(x, visualize, logpdf_GAU_ND(mrow(visualize),m_ML,C_ML))
    result=numpy.sum(gau)
    return result

# Example 1

XPlot = numpy.linspace(-8, 12, 1000)
m = numpy.ones((1,1)) * 1.0
C = numpy.ones((1,1)) * 2.0
ans=logpdf_GAU_ND(mrow(XPlot),m,C)
logpdf_GAU_ND_visualization(XPlot,ans)

# Example 2
XND = numpy.load('Solutions/XND.npy')
mu = numpy.load('Solutions/muND.npy')
C = numpy.load('Solutions/CND.npy')
pdfGau = logpdf_GAU_ND(XND, mu, C)

# Example 3
XND = numpy.load('Solutions/X1D.npy')
XPlot = numpy.linspace(-8, 12, 1000)
res=loglikelihood(XND, visualize=XPlot)
print(res)