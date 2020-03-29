import numpy as np
import random
from scipy.stats import norm
from math import sqrt


class BrownianMotion(object):
    def __init__(self, start_point, time_increment, init_time=0):
        # check format
        if not isinstance(start_point, list) and not isinstance(start_point, float) and not isinstance(start_point, int):
                raise AssertionError("__init__: start_point must be either a float or a list.")
        if not (time_increment > 0):
                raise AssertionError("__init__: time_increment must be a value biggerthan 0.")
        if not isinstance(init_time, float) and not isinstance(init_time, int):
                raise AssertionError("__init__: init_time must be an integer or a float.")

        # init memory
        self.start_point = start_point
        self.time_increment = time_increment
        self.init_time = init_time
        self.time = init_time
        self.bm = np.array([start_point]) #if type(start_point) in [float, int] else np.array([start_point])
        self.dim = self.bm.shape
        self.flag_multidimensional_bm = False if self.dim == (1,) else True 

    def current_val(self):
        return self.time, self.bm[-1]

    def next_val(self):
        self.time += self.time_increment
        new_val = self.bm[-1] + np.random.normal(0, sqrt(self.time_increment), size=self.dim)
        self.bm = np.concatenate((self.bm, new_val))

        if self.flag_multidimensional_bm:
            return self.time, new_val
        else:
            return self.time, new_val[-1]

    def reset(self):
        # init memory
        self.time = init_time
        self.bm = np.array([start_point]) #if type(start_point) in [float, int] else np.array([start_point])

    def run(self, val):
        assert(val >= 0)
        for _ in range(val):
            self.next_val()

        return self.bm
    
# TESTING UNIT
if __name__ == "__main__":
    pass    
    
    