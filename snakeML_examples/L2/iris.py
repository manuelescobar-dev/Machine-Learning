import numpy
from snakeML.numpy_transformations import mcol
from snakeML.visualization import  histogram_attributeVSfrequency, scatter_attributeVSattribute
from snakeML.loads import loadEncodedData

if __name__ == '__main__':
    data, labels, label_names = loadEncodedData("iris.csv", row_attributes=True, numpyDataType=numpy.float32)
    attributes=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    histogram_attributeVSfrequency(data, labels, features=attributes,row_attributes=True, label_names=label_names, is_label_dict=True, dense=True, center_data=True)
    scatter_attributeVSattribute(data, labels, features=attributes,row_attributes=True, label_names=label_names, is_label_dict=True, dense=True, center_data=True)



        

