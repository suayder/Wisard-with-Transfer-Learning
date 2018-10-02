from Lib_WISARD.Ram import *

class Wisard:

    def __init__(self, n_rams, n_inputs_ram, retine = 0):
        self.__n_rams = n_rams
        self.__n_inputs_ram = n_inputs_ram
        self.__discriminator = {}
        for i in range(n_rams):
            self.__discriminator[i] = Ram(n_inputs_ram)
        if retine == 0:
            self.__retine = np.random.permutation(int(n_inputs_ram*n_rams))
        else:
            self.__retine = retine

    def train(self, array):

        tuple_info = np.empty([self.__n_inputs_ram], dtype=np.int64)
        onedarray = array.flatten()
        start = 0
        count = 0
        for element in self.__retine:
            
            tuple_info[count%self.__n_inputs_ram] = onedarray[element]
            if(count%self.__n_inputs_ram == self.__n_inputs_ram-1):
                self.__discriminator[int(count/(self.__n_inputs_ram))].insertSet(np.array2string(tuple_info)) #insert into Ram
            count+=1

    def print_discriminator(self):
        keys = self.__discriminator.keys()
        for i in keys:
            print(self.__discriminator[i].getLook_up_table(), "\n\n")

    '''The input is the rectangle box and return of this below method is the number of triggered rams'''
    def classify(self, array):

        tuple_info = np.empty([self.__n_inputs_ram], dtype=np.int64)
        onedarray = array.flatten()
        triggered = 0
        count = 0

        for element in self.__retine:
            
            tuple_info[count%self.__n_inputs_ram] = onedarray[element]

            if(count%self.__n_inputs_ram == (self.__n_inputs_ram-1)):
                response = self.__discriminator[int(count/(self.__n_inputs_ram))].searchSet(np.array2string(tuple_info)) #search into Ram
                if response == True:
                    triggered+=1
            count+=1
        return triggered

    def getN_Rams(self):
        return self.__n_rams