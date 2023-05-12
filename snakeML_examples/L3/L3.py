


from testlib.dimensionality_reduction import scatter_LDA, scatter_PCA
from testlib.loads import loadEncodedData


D,L,L_names=loadEncodedData('iris.csv')
scatter_PCA(D,L,L_names,2)
scatter_LDA(D,L,L_names,2)