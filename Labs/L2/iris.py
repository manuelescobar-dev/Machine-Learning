import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    data=[]
    labels=[]
    with open(filename, 'r') as f:
        for line in f:
            record=line.strip()
            record=record.split(",")
            data.append(record[:4])
            label=record[-1]
            if label=='Iris-setosa':
                labels.append(0)
            elif label == 'Iris-versicolor':
                labels.append(1)
            else:
                labels.append(2)
    data=np.array(data,dtype=np.float32).T
    labels=np.array(labels,dtype=np.int32)
    return data, labels

def histograms(data, labels):
    setosa = data[:,labels==0]
    versicolor = data[:,labels==1]
    virginica = data[:,labels==2]
    attributes=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    for i in range(data.shape[0]):
        #plt.figure()
        plt.hist(setosa[i,:],density=True,label='Setosa')
        plt.hist(versicolor[i,:],density=True,label='Versicolor')
        plt.hist(virginica[i,:],density=True,label='Virginica')
        plt.legend()
        plt.xlabel(attributes[i])
        plt.show()

def scatters(data, labels):
    setosa = data[:,labels==0]
    versicolor = data[:,labels==1]
    virginica = data[:,labels==2]
    attributes=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i!=j:
                #plt.figure()
                plt.plot(setosa[i,:],setosa[j,:],linestyle='',marker='.',label='Setosa')
                plt.plot(versicolor[i,:],versicolor[j,:],linestyle='',marker='.',label='Versicolor')
                plt.plot(virginica[i,:],virginica[j,:],linestyle='',marker='.',label='Virginica')
                plt.legend()
                plt.xlabel(attributes[i])
                plt.ylabel(attributes[j])
                plt.show()

def mcol(row):
    return row.reshape(row.size,1)

def mrow(col):
    return col.reshape(1,col.size)


if __name__ == '__main__':
    data,labels = load_data("iris.csv")
    #histograms(data,labels)
    #scatters(data,labels)
    print(mcol(data.mean(axis=1)).shape)
    DC= data-mcol(data.mean(axis=1))
    histograms(DC,labels)
    scatters(DC,labels)  