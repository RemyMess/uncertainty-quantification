import numpy as np
import random
import math
import logging
import sys
import time
import matplotlib.pyplot as plt


class AdaptiveBiasingForce(object):
    """Class wrapping methods for the computation of the dynamics of a biased 
    Langevin equation using the ABF method for a double well potential in 2D.
    """
    def __init__(self, init_cond, l_bound, h_bound, dim=2, beta=100):
        # meta vars
        self.beta = beta
        self.l_bound = l_bound
        self.h_bound = h_bound

        if dim == 2:
            self.dim = dim
            assert(isinstance(init_cond, list))
            if len(init_cond) == dim:
                self.init_cond = init_cond
            else:
                raise AssertionError("_init_: init_cond size does not match "
                                     "the dimension.") 
        else:
            raise NotImplementedError("Dimensions other than 2 are not "
                                      "implemented.")
        
        # memory
        self.paths = np.array([])
        self.history = []

    def potential(self, x):
        """implementation of a double well potential according to equation (11)
        
        Args:
            x (list): coordinates
        
        Raises:
            NotImplementedError: if dimension is not 2
        
        Returns:
            float: value of the potential at x
        """
        if self.dim == 2:
            x1, x2 = x
            return 1/6 * (4 * (1 - x1 **2 - x2 **2) ** 2 
                          + 2 * (x1 ** 2 - 2) ** 2 
                          + ((x1 + x2) ** 2 - 1) ** 2 
                          + ((x1 - x2) ** 2 - 1) **2)

        else:
            raise NotImplementedError

    def grad_potential(self, x):
        """Implementation of the gradient of the double well potential 
        prescribed by equation (11).
        
        Args:
            x (list): coordinates
        
        Returns:
            np.array: gradient of the potential
        """
        if self.dim == 2:
            x1, x2 = x
            
            partial_x = 4/3*x1*(4*x1**2 + 5*x2**2 -5)
            partial_y = 4/3*x2*(5*x1**2 + 3*x2**2 -3)
            
            return np.array([partial_x, partial_y]) 

    def f(self, x):
        """Implementing the local mean force associated to reaction 
        coordinate xi(x,y)=x
        
        Args:
            x (list, np.array): coordinatesSpen
        
        Returns:
            float: local mean force at coordinate x
        """
        if self.dim == 2:
            x1, x2 = x
            return 2/self.beta/x1**3 - 8*x1/3 * (4 * x1 **2 + 5*x2**2-5) / (x1**2)

    def reac_coord(self, x):
        """Reaction coordinate xi (dimensionality reducting transformation)
        
        Args:
            x (list): coordinates
        
        Returns:
            list, np.array: reduced coordinates
        """
        logging.debug(f"reac_coord: type data: {type(x)}")
        return x[0]

    def _select_zl(self, i, k, j, u, m, z):
        """select the space discretisation point if the i-th path is 
        within the boundary [zl; z_{l+1}] when evaluated at kDs + jDt
        
        Args:
            path (np.array): array of np.arrays of paths
            i (int): index
            l (int): index
            k (int): index
            j (int): index
        
        Raises:
            AssertionError: when no value has been found
        
        Returns:
            [int]: index being integer part of z \ (b-a)/u 
        """
        for l in range(u):
            if z[l] <= self.reac_coord(self.paths[i]) < z[l+1]:
                return l
        if self.reac_coord(self.paths[i]) < z[0]:
            return 0
        elif z[-1] < self.reac_coord(self.paths[i]):
            return len(z) - 1
        
        raise Exception(f"_select_zl: no value found; z={z},q={self.reac_coord(self.paths[i])}")
        
    def run(self, N, T, n, m, u):
        """Run the biased dynamics following the ABF method
        
        Args:
            N (int): number of replicas of paths
            T (int): terminal time
            n (int): number of total time steps in the discretisation
            m (int): biasing force updating factor (i.e. number of times steps 
            after which an update is done)
            u (int): number of total space steps in the discretisation
        """
        # init vars
        Dt = float(T/n)
        Ds = m * Dt
        m_hat = int(T / Ds)
        
        # individual path incremented by one = +dt
        self.paths = [np.array(self.init_cond) for _ in range(N)]
        self.paths_history = [[] for _ in range(N)]
        
        # incremented by 1 = + b-a/u
        z = [self.l_bound + i * (self.h_bound - self.l_bound) / u
              for i in range(u+1)]
    
        # incremented by 1 = +Ds in time and by (b-a)/u in space
        gamma = np.ones(u+1) * self.f(self.init_cond)
        gamma_history = [[] for _ in range(m_hat)] 
        
        # run ABF
        for k in range(m_hat):
            N_in = np.zeros(u)  
            for j in range(m):
                for i in range(1, N):
                    try:
                        z_ijk = self._select_zl(i, k, j, u, m, z)
                        extra_term = np.array([gamma[z_ijk-1], 0]) * Dt
                    except:
                        extra_term = np.array([0,0])

                    new_val = self.paths[i-1] - self.grad_potential(self.paths[i-1])*Dt + extra_term + math.sqrt(Dt) * np.random.normal(0, 1, size=2)
                    logging.warning(f"new_val={new_val}")

                    # extend path
                    self.paths_history[i-1].append(new_val)
                    self.paths[i-1] = new_val
                    
            for l in range(u):
                # counting
                for n_path in range(N):
                    try:
                        n_bin = int((self.paths[n_path][0] - self.l_bound) - l * (self.h_bound - self.l_bound)/u)
                    except:
                        n_bin = 0
                        
                    if n_bin == 0:
                        N_in[l] += 1  
                    
                # Update gamma
                if N_in[0] != 0:
                    gamma[l] = sum([self.f(self.paths[i]) 
                                    for i in range(N)]) / float(N_in[l])
                    logging.warning(f"Updated value for gamma: {gamma[l]}")
                else:
                    gamma[l] = 0
                
        return self.paths   
    
    def plot(self, interval=1.4, dim=2, plot_pot=False, plot_grad=False, 
             plot_dynamics=True):
        # plot grad 2D
        if plot_pot == True:
            x = np.linspace(-interval, interval, 30)
            y = np.linspace(-interval, interval, 30)
            X, Y = np.meshgrid(x,y)

            def func_wrapper(x1, x2):
                return self.potential(x=[x1,x2])
        
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
                return self.grad_potential(x=[x1,x2])

            U,V = func_wrapper(x, y)
            fig, ax = plt.subplots()
            ax.quiver(x, y, U, V)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_ylabel('z')
            plt.title("Gradient of the potential in 2D")
            plt.xlabel("x position")
            plt.ylabel("y position")

            plt.show()
        
        # plot dynamics
        if plot_dynamics == True:
            start_pos = [0.1, 0.1]
            iterations = 1000
            end_time = 500
            logging.warning(f"plot2d: simulation of {iterations} time steps about to start. This might take up to a minute.")
            time.sleep(2)
            obj.run(N=10, T=end_time, n=iterations, m=1, u=1000)
            
            def get_path(i):
                return zip(*self.paths_history[i])

            for i in range(len(self.paths_history) - 1):
                x,y = get_path(i)
                plt.plot(x,y)
                plt.title(f"2D RBF method in 2D; {iterations} iterations, delta_t = {end_time/iterations}")
                plt.xlabel("x position")
                plt.ylabel("y position")
                plt.show()
                
            # plot dynamics projected
            for i in range(len(self.paths_history) - 1):
                x1, x2 = get_path(i)
                t = np.array(range(iterations)) /iterations * end_time
                plt.plot(t, x1)
                plt.title(f"2D RBF method projected on the x-axis; {iterations} iterations")
                plt.xlabel("Time")
                plt.ylabel("x-position")
                plt.show()
    
    
if __name__ == "__main__":
    obj = AdaptiveBiasingForce(init_cond=[0.5, 0], l_bound=-3000, h_bound=3000)
    obj.plot(plot_dynamics=True)
    