# n is a number
import numpy as np
class Ram:

    def __init__(self, n_inputs_ram):
        self.__n_inputs_ram = n_inputs_ram
        self.__look_up_table = {}

    def insertSet(self, pattern):
        try:
            self.__look_up_table[pattern] = True
        except:
            print ("Error when inserting in dictionary: Ram::insetSet\n")

    def searchSet(self, pattern):
        try:
            self.__look_up_table[pattern]
            return True
        except KeyError:
            return False
    
    def getLook_up_table(self):
        return self.__look_up_table