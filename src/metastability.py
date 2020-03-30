import numpy as np
import random
import math
import logging
import time


class DoubleWellDynamics(object):
    def __init__(self, h=1, w=1, sig=1, beta=1):
        # meta vars
        assert(h >= 0 and w >= 0)
        self.h = h
        self.w = w
        self.sig = sig
        self.r0 = 2 ** (1/6) * self.sig
        self.beta = beta

        # setting memory
        self.x = list

    def potential(self, dim, x):
        if dim == 1:
            return x ** 2 - 1  
        elif dim == 2:
            assert(isinstance(x, list))
            r = sum([i**2 for i in x]) 

            # page 44 book
            return self.h * (1 - (r - self.r0 - self.w) ** 2 / self.w**2) ** 2
        else:
            raise NotImplementedError("potential: variable 'dim' is implemented for dimension 1 or 2 only.")

    def grad_potential(self, dim, x):
        """
        Expression of the grad from p.44
        """
        if dim == 1:
            return 2 * x 
        elif dim == 2:
            r = math.sqrt(sum([i**2 for i in x]))
            grads = [2 * self.h * (1 - (r - self.r0 - self.w) ** 2 / self.w **2) * 2 * (r - self.r0 - self.w) / (self.w ** 2) * 2 * x[i] / math.sqrt(r) for i in range(dim)]
            return np.array(grads)
        else:
            raise NotImplementedError("grad_potential: variable 'dim' is implemented for dimension 1 or 2 only.")

    def run(self, x0, end_time, time_increment=1, start_time=0):
        """
        Dynamics: dX_t = - \nabla V(X_t)dt + \sqrt(2 \beta ^-1) dW_t
        Discretisation: X_t = X_t-1 + -\nabla V(X_t-1) * \Delta t + \sqrt(2 \beta ^-1) \Delta W_{t-1}
        """
        # check format x0 and get dim
        if type(x0) is int or x0 is float:
            dim = 1
        else:
            try:
                if len(x0) == 2:
                    dim = 2
                    x0 = np.array(x0, dtype=float)
            except:
                raise AssertionError("run: only dynamics of dimension 1 and 2 are supported.")

        logging.warning(f"run: running dynamics in dimension {dim}.")

        # # init bm
        # bm = BrownianMotion(start_point=[0,0], 
        #                     dim=dim, 
        #                     time_increment=time_increment,
        #                     init_time=start_time) 

        # run
        t = start_time
        self.x = [(start_time, x0)]
        x_previous = x0
        cste = math.sqrt(2 * self.beta)

        while t < end_time:
            x = x_previous - self.grad_potential(dim, x_previous) * time_increment + cste * np.random.normal(0, math.sqrt(time_increment), size=dim)
            t += time_increment

            #print(f"x: {x}")
            #print(f"grad: {self.grad_potential(dim, x)}")

            if dim == 2:
                self.x.append((t, list(x)))
            elif dim == 1:
                self.x.append((t, x[0]))
            x_previous = x

        return np.array(self.x)

if __name__ == "__main__":
    # Testing unit
    obj = DoubleWellDynamics()
    dyn = obj.run(1, 1000, time_increment=0.1)

    dyn_x = [x[1] for x in dyn]

    import matplotlib.pyplot as plt

    plt.plot(dyn_x)
    plt.show()
