import sys
sys.path.append('../')
from Lib_WISARD.Wisard import Wisard
import json
import math
import numpy as np

#--------Reading hiper-parameters from json--------#

p_datasetTrain = np.loadtxt("featuresFromxCeption/positiveTrain.txt", delimiter=" ")
n_datasetTrain = np.loadtxt("featuresFromxCeption/negativeTrain.txt", delimiter=" ")
p_datasetTest = np.loadtxt("featuresFromxCeption/positiveTest.txt", delimiter=" ")
n_datasetTest = np.loadtxt("featuresFromxCeption/negativeTest.txt", delimiter=" ")

#------------------------varying RAM Routine------------------------#
for inp in range(8,41):
    n_inputs_ram = inp
    size_Wisard = math.floor(2048/n_inputs_ram)
    results = []
    precision = []

    #------------------------Training Routine------------------------#
    #Instance of neural network
    wisard_0 = Wisard(size_Wisard, n_inputs_ram)
    wisard_1 = Wisard(size_Wisard, n_inputs_ram)

    for i in p_datasetTrain:
        wisard_0.train(i)
    for i in n_datasetTrain:
        wisard_1.train(i)

    #------------------------Testing Routine------------------------#

    #print(p_Y.shape,n_Y.shape)
    for i in p_datasetTest:
        triggered_rams_0 = wisard_0.classify(i)
        triggered_rams_1 = wisard_1.classify(i)
        
        if (triggered_rams_0>=triggered_rams_1):
            results.append(True)
            #print(triggered_rams_0*100/size_Wisard)
            precision.append(triggered_rams_0*100/size_Wisard)
        else:
            results.append(False)
            print('ERRO\tClass positive: ',triggered_rams_0*100/size_Wisard, 'class negative: ',triggered_rams_1*100/size_Wisard)

    for i in n_datasetTest:
        triggered_rams_0 = wisard_0.classify(i)
        triggered_rams_1 = wisard_1.classify(i)

        if (triggered_rams_0<=triggered_rams_1):
            results.append(True)
            #print(triggered_rams_1*100/size_Wisard)
            precision.append(triggered_rams_1*100/size_Wisard)
        else:
            results.append(False)
            print('ERRO\tClass positive: ',triggered_rams_0*100/size_Wisard, 'class negative: ',triggered_rams_1*100/size_Wisard)

    del wisard_0, wisard_1

    np_precision = np.array(precision)
    #mean = np_results.mean()
    count=0
    failure = 0
    for i in results:
        if i==True:
            count+=1
        else:
            failure+=1

    print("Hits: ", count, 'Failures: ', failure)

    del results, precision
    result = open('result1.dat', 'a')
    result.write("{0}\t{1}\t{2}\n".format(n_inputs_ram, format(count*100/40, '.2f'),format(failure*100/40,'.2f')))
    result.close()