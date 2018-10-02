import sys
sys.path.append('../')
from Lib_WISARD.Wisard import Wisard
import json
import math
import numpy as np

#--------Reading hiper-parameters from json--------#

p_dataset = np.loadtxt("dataSet/positiveFeatures.txt", delimiter=" ")
n_dataset = np.loadtxt("dataSet/negativeFeatures.txt", delimiter=" ")

p_slice_size = int(p_dataset.shape[0]/21)
n_slice_size = int(n_dataset.shape[0]/5)
#------------------------varying RAM Routine------------------------#
for inp in range(9,15):
    n_inputs_ram = inp
    size_Wisard = math.floor(2048/n_inputs_ram)
    results = []
    precision = []
    for it in range(21):
        #------------------------Training Routine------------------------#
        #Instance of neural network
        wisard_0 = Wisard(size_Wisard, n_inputs_ram)
        wisard_1 = Wisard(size_Wisard, n_inputs_ram)

        #Training dataset

        p_X = p_dataset[:(it*p_slice_size),:]
        p_X = np.append(p_X, p_dataset[(it*p_slice_size+(p_slice_size)):,:], axis=0)
        n_X = n_dataset[:((it%5)*n_slice_size),:]
        n_X = np.append(n_X, n_dataset[((it%5)*n_slice_size+n_slice_size):,:], axis=0)
        #print((it*p_slice_size), (it*p_slice_size+(p_slice_size)))
        #print(((it%5)*n_slice_size), ((it%5)*n_slice_size+n_slice_size))
        #print(p_X.shape,n_X.shape)

        for i in p_X:
            wisard_0.train(i)
        for i in n_X:
            wisard_1.train(i)


        #------------------------Testing Routine------------------------#

        p_Y = p_dataset[(it*p_slice_size):(it*p_slice_size+(p_slice_size)),:]
        n_Y = n_dataset[((it%5)*n_slice_size):((it%5)*n_slice_size+(n_slice_size)),:]
        #print(p_Y.shape,n_Y.shape)
        for i in p_Y:
            triggered_rams_0 = wisard_0.classify(i)
            triggered_rams_1 = wisard_1.classify(i)
            
            if (triggered_rams_0>=triggered_rams_1):
                results.append(True)
                #print(triggered_rams_0*100/size_Wisard)
                precision.append(triggered_rams_0*100/size_Wisard)
            else:
                results.append(False)
                print('Class positive: ',triggered_rams_0*100/size_Wisard, triggered_rams_1*100/size_Wisard)

        for i in n_Y:
            triggered_rams_0 = wisard_0.classify(i)
            triggered_rams_1 = wisard_1.classify(i)

            if (triggered_rams_0<triggered_rams_1):
                results.append(True)
                #print(triggered_rams_1*100/size_Wisard)
                precision.append(triggered_rams_1*100/size_Wisard)
            else:
                results.append(False)
                print('Class negative: ',triggered_rams_0*100/size_Wisard, triggered_rams_1*100/size_Wisard)

        del wisard_0, wisard_1
        del p_X, n_X
        del p_Y,n_Y

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
    """ result = open('result_thres0.dat', 'a')
    result.write("{0}\t{1}\t{2}\t{3}\n".format(n_inputs_ram, count,failure, format(np_precision.mean(),'.2f')))
    result.close() """