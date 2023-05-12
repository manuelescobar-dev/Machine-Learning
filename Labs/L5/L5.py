import scipy
import sklearn.datasets 
import numpy
from test_snakeML.validation import matrix_max_error, accuracy, error
import math
from test_snakeML.density_estimation import loglikelihood, logpdf_GAU_ND, logpdf_GAU_ND_error
from test_snakeML.numpy_transformations import mcol, mean_cov, mrow, wc_cov, mean

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris() ['target']
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    x_train = D[:, idxTrain]
    x_test = D[:, idxTest]
    y_train = L[idxTrain]
    y_test = L[idxTest]
    return (x_train, y_train), (x_test, y_test)

#---------MVG----------

def SJoint_MVG(Pc, x_train, y_train, x_test, tied=False):
    SJoint=[]
    if tied:
        C=wc_cov(x_train,y_train)
    for i in numpy.unique(y_train):
        if tied:
            mu=mean(x_train[:,y_train==i])
        else:
            mu, C= mean_cov(x_train[:,y_train==i])
        gau= logpdf_GAU_ND(x_test,mu, C, exp=True)
        SJoint.append(gau)
    SJoint=numpy.array(SJoint)
    SJoint=SJoint*(Pc)
    return SJoint

def logSJoint_MVG(Pc, x_train, y_train, x_test, tied=False):
    SJoint=[]
    if tied:
        C=wc_cov(x_train,y_train)
    for i in numpy.unique(y_train):
        if tied:
            mu=mean(x_train[:,y_train==i])
        else:
            mu, C= mean_cov(x_train[:,y_train==i])
        gau= logpdf_GAU_ND(x_test,mu, C)
        SJoint.append(gau)
    SJoint=numpy.array(SJoint)
    SJoint=SJoint+math.log(Pc)
    return SJoint

def SPost_MVG(SJoint, SMarginal):
    P=SJoint/SMarginal
    return P

def logSPost_MVG(logSJoint,logMarginal, exp=True):
    logSPost = logSJoint - logMarginal
    if exp:
        return numpy.exp(logSPost)
    else:
        return logSPost

def SMarginal_MVG(SJoint):
    return mrow(SJoint.sum(0))

def logSMarginal_MVG(logSJoint):
    return  mrow(scipy.special.logsumexp(logSJoint, axis=0))

def MVG(Pc, x_train, y_train, x_test, y_test, tied=False):
    #print("---MVG---")
    SJoint=SJoint_MVG(Pc,x_train,y_train,x_test, tied)
    Marginal=SMarginal_MVG(SJoint)
    Posterior=SPost_MVG(SJoint,Marginal)
    predictions=numpy.argmax(Posterior, axis=0)
    acc=accuracy(predictions,y_test)
    err=error(predictions,y_test)
    return predictions, acc, err

def logMVG(Pc, x_train, y_train, x_test, y_test, tied=False):
    #print("---logMVG---")
    logSJoint=logSJoint_MVG(Pc,x_train,y_train,x_test, tied)
    logMarginal=logSMarginal_MVG(logSJoint)
    p=logSPost_MVG(logSJoint,logMarginal)
    predictions=numpy.argmax(p, axis=0)
    acc=accuracy(predictions,y_test)
    err=error(predictions,y_test)
    return predictions, acc, err

#---------NBG----------

def SJoint_NBG(Pc, x_train, y_train, x_test, tied=False):
    SJoint=[]
    if tied:
        C=wc_cov(x_train,y_train)
        C=C*numpy.identity(C.shape[0])
    for i in numpy.unique(y_train):
        if tied:
            mu=mean(x_train[:,y_train==i])
        else:
            mu, C= mean_cov(x_train[:,y_train==i])
            C=C*numpy.identity(C.shape[0])
        gau= logpdf_GAU_ND(x_test,mu, C, exp=True)
        SJoint.append(gau)
    SJoint=numpy.array(SJoint)
    SJoint=SJoint*(Pc)
    return SJoint

def logSJoint_NBG(Pc, x_train, y_train, x_test, tied=False):
    SJoint=[]
    if tied:
        C=wc_cov(x_train,y_train)
        C=C*numpy.identity(C.shape[0])
    for i in numpy.unique(y_train):
        if tied:
            mu=mean(x_train[:,y_train==i])
        else:
            mu, C= mean_cov(x_train[:,y_train==i])
            C=C*numpy.identity(C.shape[0])
        gau= logpdf_GAU_ND(x_test,mu, C)
        SJoint.append(gau)
    SJoint=numpy.array(SJoint)
    SJoint=SJoint+math.log(Pc)
    return SJoint

def NBG(Pc, x_train, y_train, x_test, y_test, tied=False):
    #print("---NBG---")
    SJoint=SJoint_NBG(Pc,x_train,y_train,x_test, tied)
    Marginal=SMarginal_MVG(SJoint)
    Posterior=SPost_MVG(SJoint, Marginal)
    predictions=numpy.argmax(Posterior, axis=0)
    acc=accuracy(predictions,y_test)
    err=error(predictions,y_test)
    return predictions, acc, err

def logNBG(Pc, x_train, y_train, x_test, y_test, tied=False):
    #print("---logNBG---")
    logSJoint=logSJoint_NBG(Pc,x_train,y_train,x_test, tied)
    logMarginal=logSMarginal_MVG(logSJoint)
    p=logSPost_MVG(logSJoint,logMarginal)
    predictions=numpy.argmax(p, axis=0)
    acc=accuracy(predictions,y_test)
    err=error(predictions,y_test)
    return predictions, acc, err

def generativeClassifier(x_train, y_train, x_test, y_test, model, Pc=False):
    classes=numpy.unique(y_train)
    if not Pc:
        Pc=1/classes.size
    match(model):
        case("MVG"):
            return logMVG(Pc,x_train,y_train,x_test,y_test)
        case("logMVG"):
            return logMVG(Pc,x_train,y_train,x_test,y_test)
        case("NBG"):
            return NBG(Pc,x_train,y_train,x_test,y_test)
        case("logNBG"):
            return logNBG(Pc,x_train,y_train,x_test,y_test)
        case("TiedMVG"):
            return MVG(Pc,x_train,y_train,x_test,y_test, tied=True)
        case("logTiedMVG"):
            return logMVG(Pc,x_train,y_train,x_test,y_test, tied=True)
        case("TiedNBG"):
            return NBG(Pc,x_train,y_train,x_test,y_test, tied=True)
        case("logTiedNBG"):
            return logNBG(Pc,x_train,y_train,x_test,y_test, tied=True)

def kfold_generativeClassifier(D, L, model):
    error=0
    indexes = numpy.arange(D.shape[1])
    for i in range(D.shape[1]):
        x_train = D[:, indexes!=i]
        y_train = L[indexes!=i]
        x_test = D[:, indexes==i]
        y_test = L[indexes==i]
        pred, acc, err=generativeClassifier(x_train,y_train,x_test,y_test, model)
        error+=err
    print(error/D.shape[1],"%")

D, L = load_iris()
# DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
(x_train, y_train), (x_test, y_test) = split_db_2to1(D, L)

#SJoint_MVG
""" SJoint=SJoint_MVG(1/3,x_train,y_train,x_test,L)
Marginal=SMarginal_MVG(SJoint)
SJointSol = numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/SJoint_MVG.npy')
PosteriorSol = numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/Posterior_MVG.npy')
Posterior=SPost_MVG(SJoint, Marginal)
predictions=numpy.argmax(Posterior, axis=0)
accuracy(predictions,y_train)
error(predictions,y_train)
matrix_max_error(SJoint,SJointSol)
matrix_max_error(Posterior,PosteriorSol)
 """
#logSJoint_MVG
""" print("---logMVG---")
logSJoint=logSJoint_MVG(1/3,x_train,y_train,x_test,y_train)
logMarginal=logSMarginal_MVG(logSJoint)
p=logSPost_MVG(logSJoint,logMarginal)
logPosterior=logSPost_MVG(logSJoint,logMarginal, exp=False)
predictions=numpy.argmax(p, axis=0)
accuracy(predictions,y_test)
error(predictions,y_test)
logSJointSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logSJoint_MVG.npy')
matrix_max_error(logSJoint,logSJointSol)
logMarginalSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logMarginal_MVG.npy')
matrix_max_error(logMarginalSol,logMarginal)
logPosteriorSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logPosterior_MVG.npy')
matrix_max_error(logPosteriorSol,logPosterior) """

#SJoint_NBG
""" print("---NBG---")
SJoint=SJoint_NBG(1/3,x_train,y_train,x_test,y_train)
Marginal=SMarginal_MVG(SJoint)
Posterior=SPost_MVG(SJoint,Marginal)
predictions=numpy.argmax(Posterior, axis=0)
accuracy(predictions,y_test)
error(predictions,y_test)
SJointSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/SJoint_NaiveBayes.npy')
matrix_max_error(SJoint,SJointSol)
PosteriorSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/Posterior_NaiveBayes.npy')
matrix_max_error(Posterior,PosteriorSol) """

#logSJoint_NBG
""" print("---logNBG---")
logSJoint=logSJoint_NBG(1/3,x_train,y_train,x_test,y_train)
logMarginal=logSMarginal_MVG(logSJoint)
p=logSPost_MVG(logSJoint,logMarginal)
logPosterior=logSPost_MVG(logSJoint,logMarginal, exp=False)
predictions=numpy.argmax(p, axis=0)
accuracy(predictions,y_test)
error(predictions,y_test)
logSJointSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logSJoint_NaiveBayes.npy')
matrix_max_error(logSJoint,logSJointSol)
logMarginalSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logMarginal_NaiveBayes.npy')
matrix_max_error(logMarginalSol,logMarginal)
logPosteriorSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logPosterior_NaiveBayes.npy')
matrix_max_error(logPosteriorSol,logPosterior) """

#TiedMVG
""" SJoint=SJoint_MVG(1/3,x_train,y_train,x_test,L, tied=True)
Marginal=SMarginal_MVG(SJoint)
SJointSol = numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/SJoint_TiedMVG.npy')
PosteriorSol = numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/Posterior_TiedMVG.npy')
Posterior=SPost_MVG(SJoint, Marginal)
predictions=numpy.argmax(Posterior, axis=0)
accuracy(predictions,y_test)
error(predictions,y_test)
matrix_max_error(SJoint,SJointSol)
matrix_max_error(Posterior,PosteriorSol) """

#TiedNBG
""" SJoint=SJoint_NBG(1/3,x_train,y_train,x_test,L, tied=True)
Marginal=SMarginal_MVG(SJoint)
SJointSol = numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/SJoint_TiedNaiveBayes.npy')
PosteriorSol = numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/Posterior_TiedNaiveBayes.npy')
Posterior=SPost_MVG(SJoint, Marginal)
predictions=numpy.argmax(Posterior, axis=0)
accuracy(predictions,y_test)
error(predictions,y_test)
matrix_max_error(SJoint,SJointSol)
matrix_max_error(Posterior,PosteriorSol) """

#logTiedMVG
""" print("---logTiedMVG---")
logSJoint=logSJoint_MVG(1/3,x_train,y_train,x_test,y_train,tied=True)
logMarginal=logSMarginal_MVG(logSJoint)
p=logSPost_MVG(logSJoint,logMarginal)
logPosterior=logSPost_MVG(logSJoint,logMarginal, exp=False)
predictions=numpy.argmax(p, axis=0)
accuracy(predictions,y_test)
error(predictions,y_test)
logSJointSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logSJoint_TiedMVG.npy')
matrix_max_error(logSJoint,logSJointSol)
logMarginalSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logMarginal_TiedMVG.npy')
matrix_max_error(logMarginalSol,logMarginal)
logPosteriorSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logPosterior_TiedMVG.npy')
matrix_max_error(logPosteriorSol,logPosterior)  """

#logTiedNBG
""" print("---logTiedNBG---")
logSJoint=logSJoint_NBG(1/3,x_train,y_train,x_test,y_train, tied=True)
logMarginal=logSMarginal_MVG(logSJoint)
p=logSPost_MVG(logSJoint,logMarginal)
logPosterior=logSPost_MVG(logSJoint,logMarginal, exp=False)
predictions=numpy.argmax(p, axis=0)
accuracy(predictions,y_test)
error(predictions,y_test)
logSJointSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logSJoint_TiedNaiveBayes.npy')
matrix_max_error(logSJoint,logSJointSol)
logMarginalSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logMarginal_TiedNaiveBayes.npy')
matrix_max_error(logMarginalSol,logMarginal)
logPosteriorSol=numpy.load('/Users/manuelescobar/Documents/POLITO/2023-1/ML/Labs/L5/Solutions/logPosterior_TiedNaiveBayes.npy')
matrix_max_error(logPosteriorSol,logPosterior) """

# ----- COMPARISON ------
models=['MVG','logMVG','NBG','logNBG','TiedMVG','logTiedMVG','TiedNBG','logTiedNBG']

#Kfold
for i in models:
    print("Kfold - ",i)
    kfold_generativeClassifier(D,L,i)
