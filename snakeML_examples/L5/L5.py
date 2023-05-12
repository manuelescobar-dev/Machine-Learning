from testlib.validation import accuracy, error, matrix_max_error
from testlib.density_estimation import logpdf_GAU_ND
from testlib.numpy_transformations import mean_cov, mrow, wc_cov, mean
from testlib.loads import load_iris, db_train_test_split
from testlib.gaussian import kfold_generativeClassifier, generativeClassifier

D, L = load_iris()

# DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
(x_train, y_train), (x_test, y_test) = db_train_test_split(D, L)

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
    print("---",i,"---")
    pred, acc, err=generativeClassifier(x_train,y_train, x_test, y_test,i)
    print("Accuracy:",acc,"Error:",err)
    kfold=kfold_generativeClassifier(D,L,i)
    print("Kfold:", round(kfold,1))
