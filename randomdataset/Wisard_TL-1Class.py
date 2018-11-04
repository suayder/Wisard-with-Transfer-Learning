import sys
sys.path.append('../')
from Lib_WISARD.Wisard import Wisard
import json
import math
import numpy as np

#--------Reading hiper-parameters from json--------#

p_datasetTrain = np.loadtxt("featuresFromxCeption/positiveTrain.txt", delimiter=" ")
p_datasetTest = np.loadtxt("featuresFromxCeption/positiveTest.txt", delimiter=" ")
n_datasetTest = np.loadtxt("featuresFromxCeption/negativeTest.txt", delimiter=" ")

#------------------------varying RAM Routine------------------------#
for inp in range(8,41):
    n_inputs_ram = inp
    size_Wisard = math.floor(2048/n_inputs_ram)
    results = []
    negative_results = []
    #------------------------Training Routine------------------------#
    #Instance of neural network
    wisard = Wisard(size_Wisard, n_inputs_ram)

    for i in p_datasetTrain:
        wisard.train(i)
    #------------------------Testing Routine------------------------#

    for i in p_datasetTest:
        triggered_rams = wisard.classify(i)

        results.append(triggered_rams*100/size_Wisard)
        print("Percent triggered RAMs: ", triggered_rams*100/size_Wisard)

    for i in n_datasetTest:
        triggered_rams = wisard.classify(i)

        negative_results.append(triggered_rams*100/size_Wisard)
        print("Percent triggered RAMs: ", triggered_rams*100/size_Wisard)

    del wisard

    np_results = np.array(results)
    mean = np_results.mean()
    count=0
    failure = 0
    fp = 0
    fn = 0
    for i in np_results:
        if i>mean:
            count+=1
        else:
            fn+=1
            failure+=1
    for i in negative_results:
        if i>mean:
            failure+=1
            fp+=1
        else:
            count+=1

    print("Hits: ", count, 'FN: ', fn, 'Failures: ', failure, 'FP: ', fp)
    print("Mean: ", mean)

    del results, np_results, negative_results
    result = open('results/result_1discriminator1.dat', 'a')
    result.write("{0}\t{1}\t{2}\n".format(n_inputs_ram,format((count*100)/40,'.2f'),format((failure*100)/40,'.2f'),))
    result.close()
