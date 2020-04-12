import numpy as np
import random
import math
import logging
import sys
import time


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

            partial_x = - 4/3 * (4 * x1**3 + x2*(x1**2 + x2**2 - 1) 
                            + x1*(3*x2**2 - 5) )
            partial_y = 1/6 * (-16 * (1 - x1**2 - x2**2) * x2  
                               + 4 * (x1 + x2) * ((x1 + x2)**2 - 1) 
                               - 4 * (x1 - x2) * ((x1 - x2)**2 - 1))
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
            return 2/self.beta/x1**3 - 8/3 * (4 * x1 **3 + x1**2*x2 
                                              + 3*x1*x2**2-5*x1 
                                              + x2**3 - x2) / (x1**2)

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
        raise Exception(f"_select_zl: no value found; z={z}, q={self.reac_coord(self.paths[i])}")
        
    def run(self, N, T, n, m, u):
        """Run the biased dynamics following the ABF method
        
        Args:
            N (int): number of replicas of paths
            T (int): terminal time
            n (int): number of total time steps in the discretisation
            m (int): biaising force updating factor (i.e. number of times steps 
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
        gamma = np.ones(m_hat) * self.f(self.init_cond)
        gamma_history = [[] for _ in range(m_hat)] 
        
        # Note: N_in[i][j] = N_{ij}
        N_in = [np.zeros(u) for _ in range(int(m_hat))]  
              
        # run ABF
        for k in range(m_hat):
            for j in range(m):
                for i in range(1, N):
                    z_ijk = self._select_zl(i, k, j, u, m, z)
                    z_ijk_old = z_ijk
                    
                    # except Exception as e:
                    #     logging.warning(f"run: _select_zl, {e}.")
                    
                    
                    # print(f"grad: {self.grad_potential(self.paths[i-1][k*m + j])*Dt }")
                    term1 = self.grad_potential(self.paths[i-1])*Dt 
                    term2 = np.array([gamma[z_ijk-1] * Dt, 0]) 
                    term3 = math.sqrt(Dt) * np.random.normal(0, 1, size=2)
                                        
                    new_val = self.paths[i-1] - self.grad_potential(self.paths[i-1])*Dt + np.array([gamma[z_ijk-1], 0]) * Dt + math.sqrt(Dt) * np.random.normal(0, 1, size=2)
                    logging.warning(f"new_val={new_val}")
                    
                    # extend path
                    self.paths_history[i-1].append(new_val)
                    self.paths[i-1] = new_val
                    
            for l in range(u):
                n_bin = (self.paths[l][-1][0] 
                         - self.l_bound) // (self.h_bound - self.l_bound)/u
                N_in[k][n_bin] +=1  
                
                # Update gamma
                gamma[k+1][l] = sum([self.f(self.paths[i][-1]) 
                                     for i in range(N+1)]) / N_in[k][l]
             
        return self.paths   
    
    def plot(self):
        pass
    
    
if __name__ == "__main__":
    obj = AdaptiveBiasingForce(init_cond=[0.2, 0.2], l_bound=-3000, h_bound=3000)
    print(obj.run(N=10, T=10, n=10000, m=1000, u=10))