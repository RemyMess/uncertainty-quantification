#%%
import numpy as np
import random
import math
import logging
import sys
import time

#%%
class DoubleWellDynamics(object):
    """
    Computational experience for the dynamics of a particle in a 1D/2D double well potential whose dynamics
    is prescribed by
    dX_t = - nabla V(X_t)dt + \sqrt(2 \ beta ^-1) dW_t
    """
    def __init__(self, h=1, w=100, sig=100, beta=100):
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
            return self.h * (1 - (abs(x) - self.r0 - self.w) ** 2 / self.w**2) ** 2  
        
        elif dim == 2:
            # eq page 45 book
            assert(isinstance(x, list))
            return 1/6 * (4 * (1 - x[0] **2 - x[1] **2) ** 2 + 2 * (x[0] ** 2 - 2) ** 2 + ((x[0] + x[1]) ** 2 - 1) ** 2 + ((x[0] - x[1]) ** 2 - 1) **2)
        else:
            raise NotImplementedError("potential: variable 'dim' is implemented for dimension 1 or 2 only.")

    def grad_potential(self, dim, x):
        """
        Expression of the double well potential can be found in "T. Lelièvre, M. Rousset and G. Stoltz, Free energy computations: A mathematical perspective,
        Imperial College Press, 2010." at page 44.
        """
        if dim == 1:
            return 2 * self.h * (1 - (abs(x) - self.r0 - self.w) ** 2 / (self.w **2) ) * (-2) * (abs(x) - self.r0 - self.w) / (self.w ** 2) * 2
        elif dim == 2:
            # eq page 45 book
            partial_x = 1/6 * (-16 * (1 - x[0]**2 - x[1]**2) * x[0] + 8 * x[0] * (x[0] ** 2 - 2) + 4 * (x[0] + x[1]) * ((x[0] + x[1])**2 - 1) + 4 * (x[0] - x[1]) * ((x[0] - x[1])**2 - 1) 
            )
            partial_y = 1/6 * (-16 * (1 - x[0]**2 - x[1]**2) * x[1]  + 4 * (x[0] + x[1]) * ((x[0] + x[1])**2 - 1) - 4 * (x[0] - x[1]) * ((x[0] - x[1])**2 - 1) 
            )

            return np.array([partial_x, partial_y])
        else:
            raise NotImplementedError("grad_potential: variable 'dim' is implemented for dimension 1 or 2 only.")

    def run(self, x0, end_time, time_increment=1, start_time=0):
        """
        Dynamics: dX_t = - nabla V(X_t)dt + \sqrt(2 \ beta ^-1) dW_t
        Discretisation: X_t = X_t-1 + -nabla V(X_t-1) * Delta t + \sqrt(2 beta ^-1) Delta W_{t-1}
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

        logging.warning(f"run: running dynamics.")

        # run
        t = start_time
        self.x = [(start_time, x0)]
        x_previous = x0
        cste = math.sqrt(2 * self.beta ** -1)

        while t < end_time:
            x = x_previous - self.grad_potential(dim, x_previous) * time_increment + cste * np.random.normal(0, math.sqrt(time_increment), size=dim)
            t += time_increment

            logging.debug(f"x: {x}")
            logging.debug(f"grad: {self.grad_potential(dim, x)}")

            if dim == 2:
                self.x.append((t, list(x)))
            elif dim == 1:
                self.x.append((t, x[0]))
            x_previous = x

        return np.array(self.x)

    def plot_1d(self, interval=50, plot_pot=True, plot_grad=True, plot_dynamics=True):
        # plot potential 1D
        if plot_pot == True:
            pot = [self.potential(dim=1, x=i) for i in range(-500, 500, 1)]
            plt.plot(pot)
            plt.title("Double well potential in 1D")
            plt.show()

        # plot gradients 1D
        if plot_grad == True:
            grad_pot = [self.grad_potential(dim=1, x=i) for i in range(-500, 500, 1)]
            plt.plot(grad_pot)
            plt.title("Gradient of the potential in 1D")
            plt.show()

        # plot dynamics 1D

        if plot_dynamics == True:
            start_pos1 = 0
            dyn = self.run(start_pos1, 1000000, time_increment=60)
            dyn = [i[1] for i in dyn]

            plt.plot(dyn)
            plt.title(f"Metastability in 1D for particle starting at x={start_pos1}")
            plt.show()

            start_pos2 = 250
            dyn = self.run(start_pos2, 1000000, time_increment=60)
            dyn = [i[1] for i in dyn]
            plt.plot(dyn)
            plt.title(f"Metastability in 1D for particle starting at x={start_pos2}")
            plt.show()

    def plot_2d(self, interval=10, plot_pot=True, plot_grad=True, plot_dynamics=True, plot_dynamics2=True):
        # plot grad 2D
        if plot_pot == True:
            x = np.linspace(-interval, interval, 30)
            y = np.linspace(-interval, interval, 30)
            X, Y = np.meshgrid(x,y)

            def func_wrapper(x1, x2):
                return self.potential(dim=2, x=[x1,x2])
        
            Z = func_wrapper(X, Y)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.contour3D(X, Y, Z, 50, cmap='binary')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.title("Double well potential in 2D")
            plt.show()

        # plot gradients 2D
        if plot_grad == True:
            # Make the grid
            x, y = np.meshgrid(np.arange(-interval, interval, interval/10),
                                np.arange(-interval, interval, interval/10))

            def func_wrapper(x1, x2):
                return self.grad_potential(dim=2, x=[x1,x2])

            U,V = func_wrapper(x, y)
            fig, ax = plt.subplots()
            ax.quiver(x, y, U, V)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_ylabel('z')
            plt.title("Gradient of the potential in 2D")

            plt.show()

        if plot_dynamics == True:
            start_pos = [0.1, 0.1]
            iterations = 1000
            logging.warning(f"plot2d: simulation of {iterations} time steps about to start. This might take up to minute.")
            time.sleep(2)
            dyn = obj.run(start_pos, iterations, time_increment=0.2)
            dyn = [i[1] for i in dyn]
            x, y = zip(*dyn)
            plt.plot(x,y)
            plt.title(f"Metastability for particle 2D")
            plt.show()
        
        # plot metastable behaviour
        if plot_dynamics2 == True:
            start_pos = [0.1, 0.1]
            iterations = 1000
            logging.warning(f"plot2d: simulation of {iterations} time steps about to start. This might take up to minute.")
            time.sleep(2)
            dyn = obj.run(start_pos, iterations, time_increment=0.2)
            dyn = [i[1] for i in dyn]
            x, y = zip(*dyn)
            plt.plot(iterations*0.2,x)
            plt.title(f"Metastability for particle 2D")
            plt.show()
#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    obj = DoubleWellDynamics()
    #obj.plot_1d(1.2, plot_pot=True, plot_grad=True)
    #obj.plot_2d(1.2, plot_pot=True, plot_grad=True)
    obj.plot_2d(1.2, plot_dynamics2=True)

# %%
