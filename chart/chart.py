import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("../OurDataset/result_thresMean-1Class.dat", delimiter="\t")
#Get first column
X = data[:,:1].astype(int)
X = X.reshape(X.shape[0],)

#get second column
Y = data[:,1:2]
Y = Y.reshape(Y.shape[0],)

for a,b in zip(X,Y):
    plt.text(a-0.3,b+0.5,str(b)+'%')
plt.xlabel("Número de entradas por RAM")
plt.ylabel("Acertos no conjunto de dados (%)")
plt.title("Porcentagem de acerto em relação ao dataset (84 imagens)")
plt.bar(X,Y, color=(0.4,0.4,0.4))
plt.savefig('OURDATASET_thresMean-1Class.png')
plt.show()