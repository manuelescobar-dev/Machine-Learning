


from test_snakeML.dimensionality_reduction import scatter_LDA
from test_snakeML.loads import loadEncodedData


D,L,L_names=loadEncodedData('iris.csv')
scatter_LDA(2,D,L,L_names)