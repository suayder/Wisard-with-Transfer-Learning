import sys
sys.path.append('../')
from Lib_WISARD.Wisard import Wisard
import json
import math
import numpy as np

#--------Reading hiper-parameters from json--------#

p_dataset = np.loadtxt("dataSet/MeanPositiveFeatures.txt", delimiter=" ")
n_dataset = np.loadtxt("dataSet/MeanNegativeFeatures.txt", delimiter=" ")

p_slice_size = int(p_dataset.shape[0]/21)
n_slice_size = int(n_dataset.shape[0]/15)
#------------------------varying RAM Routine------------------------#
for inp in range(9,15):
    n_inputs_ram = inp
    size_Wisard = math.floor(2048/n_inputs_ram)
    results = []
    negative_results = []
    for it in range(21):
        #------------------------Training Routine------------------------#
        #Instance of neural network
        wisard = Wisard(size_Wisard, n_inputs_ram)

        #Training dataset

        p_X = p_dataset[:(it*p_slice_size),:]
        p_X = np.append(p_X, p_dataset[(it*p_slice_size+(p_slice_size)):,:], axis=0)
        n_X = n_dataset[:((it%5)*n_slice_size),:]
        n_X = np.append(n_X, n_dataset[((it%5)*n_slice_size+n_slice_size):,:], axis=0)
        #print((it*p_slice_size), (it*p_slice_size+(p_slice_size)))
        #print(((it%5)*n_slice_size), ((it%5)*n_slice_size+n_slice_size))
        #print(p_X.shape,n_X.shape)

        for i in p_X:
            wisard.train(i)
        #------------------------Testing Routine------------------------#

        p_Y = p_dataset[(it*p_slice_size):(it*p_slice_size+(p_slice_size)),:]
        n_Y = n_dataset[((it%5)*n_slice_size):((it%5)*n_slice_size+(n_slice_size)),:]
        #print(p_Y.shape,n_Y.shape)
        for i in p_Y:
            triggered_rams = wisard.classify(i)

            results.append(triggered_rams*100/size_Wisard)

        for i in n_Y:
            triggered_rams = wisard.classify(i)

            negative_results.append(triggered_rams*100/size_Wisard)

        del wisard
        del p_X, n_X
        del p_Y,n_Y

    np_results = np.array(results)
    mean = np_results.mean()
    count=0
    failure = 0
    for i in np_results:
        if i>mean or i>70:
            count+=1
        else:
            failure+=1
    for i in negative_results:
        if i>mean or i>70:
            failure+=1
        else:
            count+=1

    print("Hits: ", count, 'Failures: ', failure)
    print("Mean: ", mean)

    del results, np_results, negative_results
    result = open('../results/result_thresMean-1Class.dat', 'a')
    result.write("{0}\t{1}\t{2}\n".format(n_inputs_ram,format((count*100)/84,'.2f'),format((failure*100)/84,'.2f')))
    result.close()

#OBS: Metodologia para teste: crossvalidation pega 3 positivas e 1 negativa