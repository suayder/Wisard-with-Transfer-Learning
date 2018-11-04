import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("../results/result_thresMean-1Class.dat", delimiter="\t")
data2 = np.loadtxt("../results/WisardWithOutTL.dat", delimiter="\t")
#Get first column
X = data[:,:1].astype(int)
X = X.reshape(X.shape[0],)
X2 = data2[:,:1].astype(int)
X2 = X2.reshape(X2.shape[0],)

#get second column
Y = data[:,1:2]
Y = Y.reshape(Y.shape[0],)
Y2 = data2[:,1:2]
Y2 = Y2.reshape(Y2.shape[0],)

for a,b in zip(X,Y):
    plt.text(a-0.42,b+0.5,str(b)+'%', size=8)

for a,b in zip(X2,Y2):
    plt.text(a-0.02,b+0.5,str(b)+'%', size=8)
plt.xlabel("Número de entradas por RAM")
plt.ylabel("Acertos no conjunto de dados (%)")
plt.title("Porcentagem de acerto em relação ao dataset (77 imagens)")
#plt.bar(X,Y)
plt.bar(X-0.2, Y, label='Arq. 1', color=(0.4,0.4,0.4), width=0.4)
plt.bar(X2+0.2, Y2, label='Wisard', width=0.4)
plt.legend(loc='best')
plt.savefig('wisardWithouttl.pdf')
#plt.show()