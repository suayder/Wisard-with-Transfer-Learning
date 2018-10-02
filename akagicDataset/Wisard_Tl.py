import sys
sys.path.append('../')
from Lib_WISARD.Wisard import Wisard
import json
import math
import numpy as np
import matplotlib.pyplot as plt

#--------Reading hiper-parameters from json--------#

with open('Lib_WISARD/Network_Settings.json') as jsonfile:
    settings = json.load(jsonfile)

n_inputs_ram = settings['n_inputs_ram']
path_train = settings['training_dataset']
result = settings['result']
train_exemples = settings['examples_samples']

dataset = np.loadtxt("dataFeaturesForTest.txt", delimiter=" ")

slice_size = int(dataset.shape[0]/18)
#------------------------varying RAM Routine------------------------#
precisions = []
for inp in range(9,15):
    n_inputs_ram = inp
    size_Wisard = math.floor(2048/n_inputs_ram)
    results = []
    for it in range(18):
        #------------------------Training Routine------------------------#
        #Instance of neural network
        wisard = Wisard(size_Wisard, n_inputs_ram)

        #Training dataset

        X = dataset[:(it*slice_size),:]
        X = np.append(X, dataset[(it*slice_size+(slice_size)):,:], axis=0)
        #print((it*slice_size), (it*slice_size+(slice_size)))
        #print(X.shape)

        for i in X:
            wisard.train(i)

        #------------------------Testing Routine------------------------#

        Y = dataset[(it*slice_size):(it*slice_size+(slice_size)),:]
        for i in Y:
            triggered_rams = wisard.classify(i)
            print((triggered_rams*100)/(math.floor(2048/n_inputs_ram)),'%')
            results.append(((triggered_rams*100)/(math.floor(2048/n_inputs_ram))))

        del wisard
        del X
        del Y

    np_results = np.array(results)
    mean = np_results.mean()
    count=0
    failure = 0
    for i in np_results:
        if i>mean or i>70:
            count+=1
        else:
            failure+=1

    print("MEAN: ", mean)
    print("number: ", count)

    del results, np_results
    result = open('result_thres0.dat', 'a')
    result.write("{0}\t{1}\t{2}\n".format(n_inputs_ram,format((count*100)/90,'.2f'),format((failure*100)/90,'.2f')))
    result.close()
    precisions.append(format((count*100)/90,'.2f'))
