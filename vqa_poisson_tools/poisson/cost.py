"""
Prepare Qiskit circuits for estimating Poisson cost functions.
"""


from abc import ABC, abstractmethod
from math import pi
import numpy as np

from qiskit.providers.aer import StatevectorSimulator, UnitarySimulator



class PoissonVQACost(ABC):
    """Abstract class for VQA cost function drivers.
    Basic functionality is to prepare circuits for estimating cost/energy functionals.
    """
    
    def __init__(self):
        
        pass
    
    
    @abstractmethod
    def get_circuits(self):
        """Assemble circuits for estimating cost functional.

        Returns
        -------
        object
            Circuit or collection of circuits
        """
        
        pass
    
    
    @abstractmethod
    def energy_from_measurements(self, measurements):
        """Estimate energy/cost functional from measurements.

        Parameters
        ----------
        measurements : object
            Counts or collection of counts. Assumed to be in probability form.

        Returns
        -------
        float
            Estimate for energy/cost functional.
        """
        
        pass
    
    
    cost_from_measurements = energy_from_measurements
    
    

class PoissonVQACost_OpInnerp(PoissonVQACost):
    """Class preparing circuits for estimating cost functionals by combining estimates of operator norms and inner products.
    
    Parameters
    ----------
    op_driver : PoissonVQAOperator 
        Instance of driver for preparing operator circuits.
    innerp_driver : InnerProduct
        Instance of driver for preparing inner product circuits.
    """
    
    def __init__(self, op_driver, innerp_driver):
        
        self._op_driver     = op_driver
        self._innerp_driver = innerp_driver
        
        # private variables
        self._slice_op       = 0
        self._slice_innerp   = 0
        self._num_parameters = op_driver.ansatz.num_parameters
        
        # call this to initialize slice variables
        self.get_circuits()
        
        
    def get_circuits(self):
        """Assemble circuits for estimating cost functional.

        Returns
        -------
        circuits : list[QuantumCircuit]
            List of required circuits.
        """
        
        # get circuits
        op_circs     = self._op_driver.get_circuits()
        if not isinstance(op_circs, list):
            op_circs = [op_circs]

        innerp_circs = self._innerp_driver.get_circuits()
        if not isinstance(innerp_circs, list):
            innerp_circs = [innerp_circs]
            
        num_op     = len(op_circs)
        num_innerp = len(innerp_circs)
        circuits   = op_circs + innerp_circs
        
        # set indices
        if num_op == 1:
           self._slice_op = 0
        else:
           self._slice_op = slice(0, num_op)
           
        if num_innerp == 1:
           self._slice_innerp = num_op
        else:
           self._slice_innerp = slice(num_op, num_op+num_innerp)

        return circuits
    
    
    @property
    def op_driver(self):
        """Driver for preparing operator circuits (PoissonVQAOperator)."""
        return self._op_driver
    
    
    @property
    def innerp_driver(self):
        """Driver for preparing inner product circuits (InnerProduct)."""
        return self._innerp_driver
    
    
    @op_driver.setter
    def op_driver(self, op_driver_):
        self._op_driver = op_driver_
        
        
    @innerp_driver.setter
    def innerp_driver(self, innerp_driver_):
        self._innerp_driver = innerp_driver_
    
    
    
class Poisson1dVQACost_Sato21_Innerp(PoissonVQACost_OpInnerp):
    """Class preparing circuits for estimating the Dirichlet energy as in [1] via estimates of operator norms and inner products.
  
    References
    ----------
    [1] Yuki Sato, Ruho Kondo, Satoshi Koide, Hideki Takamatsu and Nobuyuki Imoto,
        "Variational quantum algorithm based on the minimum potential energy
        for solving the Poisson equation", PRA 104, 052409 (2021). 
    """
    
    def energy_from_measurements(self, measurements=[None, None]):
        """Estimate energy/cost functional from measurements.

        Parameters
        ----------
        measurements : list[counts]
            Measurements in probability form. Default is empty list.

        Returns
        -------
        energy : float
            Energy functional estimate.
        """
        
        E_A     = self.op_driver.energy_from_measurements(measurements[self._slice_op])
        E_inner = self.innerp_driver.dot_from_counts(measurements[self._slice_innerp],
                                                     as_prob=True)
        
        energy  = -0.5*E_inner**2/E_A
        
        return energy
    
    
    cost_from_measurements = energy_from_measurements
    
    
    def get_circuits_energy(self, og_circuits, theta):
        """Circuits to estimate energy.

        Parameters
        ----------
        og_circuits : list[QuantumCircuit]
            List of original unbound energy circuits.
        theta : numpy array
            Parameters.

        Returns
        -------
        grad_circs : list[QuantumCircuit]
            List of circuits required for estimating the energy.
        """
        
        # bind og_circuits
        num_circs      = len(og_circuits)
        bound_circuits = [None]*num_circs
        
        for c in range(num_circs):
            qc = og_circuits[c]
            if qc is None:
                bound_circuits[c] = None
            else:
                bound_circuits[c] = qc.bind_parameters(theta)
            
        return bound_circuits
    
        
    get_circuits_cost = get_circuits_energy
    
    
    def get_circuits_gradient(self, og_circuits, theta):
        """Circuits to estimate gradient.

        Parameters
        ----------
        og_circuits : list[QuantumCircuit]
            List of original unbound cost circuits.
        theta : numpy array
            Parameters.

        Returns
        -------
        grad_circs : list[QuantumCircuit]
            List of circuits required for estimating the gradient.
        """
        
        # parameters
        num_parameters = self._num_parameters
        
        # get OG circuits
        op_circuits     = og_circuits[self._slice_op]
        innerp_circuits = og_circuits[self._slice_innerp]
        
        # list format
        if not isinstance(op_circuits, list):
            op_circuits = [op_circuits]
        if not isinstance(innerp_circuits, list):
            innerp_circuits = [innerp_circuits]
        
        len_op      = len(op_circuits)
        len_innerp  = len(innerp_circuits)
        total_og    = len_op + len_innerp
        
        total_grad  = total_og+len_op*2*num_parameters+len_innerp*num_parameters
        
        grad_circs  = [None]*(total_grad)
        
        # estimate for operator and inner product
        idx = 0
        for i in range(total_og):
            if og_circuits[i] is not None:
                grad_circs[idx] = og_circuits[i].bind_parameters(theta)
            idx += 1
        
        # estimate for gradient of operator
        for p in range(num_parameters):
            theta_     = theta.copy()
            theta_[p] += pi/2.
            for i in range(len_op):
                if op_circuits[i] is not None:
                    grad_circs[idx] = op_circuits[i].bind_parameters(theta_)
                idx += 1
                
        for p in range(num_parameters):
            theta_     = theta.copy()
            theta_[p] -= pi/2.
            for i in range(len_op):
                if op_circuits[i] is not None:
                    grad_circs[idx] = op_circuits[i].bind_parameters(theta_)
                idx            += 1
            
        # estimate for gradient of inner product
        for p in range(num_parameters):
            theta_     = theta.copy()
            theta_[p] += pi
            for i in range(len_innerp):
                if innerp_circuits[i] is not None:
                    grad_circs[idx] = innerp_circuits[i].bind_parameters(theta_)
                idx            += 1
                
        return grad_circs
    
    
    def gradient_from_measurements(self, measurements=[None, None, None]):
        """Estimate gradient from experimental measurements.

        Parameters
        ----------
        measurements : list[Counts]
            List of experimental counts in probability form.
            The default is [None, None, None].

        Returns
        -------
        grad : numpy array
            Gradient.
        """
        
        # parameters
        num_parameters = self._num_parameters
        
        # get OG circuits
        op_meas     = measurements[self._slice_op]
        innerp_meas = measurements[self._slice_innerp]
        
        len_op = len_innerp = 1
        if isinstance(op_meas, list):
            len_op = len(op_meas)
        if isinstance(innerp_meas, list):
            len_innerp = len(innerp_meas)

        total_og    = len_op + len_innerp
        total_grad  = total_og+len_op*2*num_parameters+len_innerp*num_parameters
   
        if measurements is None:
            measurements = [None]*total_grad
        
        E_A     = self.op_driver.energy_from_measurements(op_meas)
        E_inner = self.innerp_driver.dot_from_counts(innerp_meas, as_prob=True)
        
        # get shifted expectations
        E_A_pl     = np.empty(num_parameters)
        E_A_min    = np.empty(num_parameters)
        E_inner_pl = np.empty(num_parameters)
        grad       = np.empty(num_parameters)
        
        # shifted operator plus
        idx0 = len_op+len_innerp
        for p in range(num_parameters):
            idx       = None
            idx_start = idx0 + p*len_op
            idx_end   = idx0 + (p+1)*len_op
            if len_op == 1:
                idx = idx_start
            else:
                idx = slice(idx_start, idx_end)

            E_A_pl[p] = self.op_driver.energy_from_measurements(measurements[idx])
            
        # shifted operator minus
        idx0 = len_op+len_innerp+num_parameters*len_op
        for p in range(num_parameters):
            idx       = None
            idx_start = idx0 + p*len_op
            idx_end   = idx0 + (p+1)*len_op
            if len_op == 1:
                idx = idx_start
            else:
                idx = slice(idx_start, idx_end)
            E_A_min[p] = self.op_driver.energy_from_measurements(measurements[idx])
            
        # shifted inner product plus
        idx0 = len_op+len_innerp+2*num_parameters*len_op
        for p in range(num_parameters):
            idx       = None
            idx_start = idx0 + p*len_innerp
            idx_end   = idx0 + (p+1)*len_innerp
            if len_innerp == 1:
                idx = idx_start
            else:
                idx = slice(idx_start, idx_end)

            E_inner_pl[p] = self.innerp_driver.dot_from_counts(measurements[idx],
                                                               as_prob=True)
            
        # put into gradient
        for p in range(num_parameters):
            E_A_prime     = 0.5*(E_A_pl[p] - E_A_min[p])
            E_inner_prime = E_inner_pl[p]
            
            grad[p]       = -0.5*(E_inner*E_inner_prime)/E_A + 0.5*(E_inner**2*E_A_prime)/E_A**2
            
            
        return grad
    
    
    def energy_exact(self, theta):
        """Numerically exact energy.

        Parameters
        ----------
        theta : numpy array
            Parameters.

        Returns
        -------
        energy : float
            Energy value.
        """
        
        # parameters
        unbound        = self.op_driver.ansatz.copy()
        
        # bind
        self.op_driver.ansatz  = unbound.bind_parameters(theta)
        self.innerp_driver.lhs = unbound.bind_parameters(theta)
        
        # estimate
        E_A     = self.op_driver.energy_from_measurements()
        E_inner = self.innerp_driver.dot_from_counts()
        
        # compute
        energy  = -0.5*E_inner**2/E_A
        
        # unbind
        self.op_driver.ansatz  = unbound
        self.innerp_driver.lhs = unbound
        
        return energy
    
    
    cost_exact = energy_exact
    
    
    def gradient_exact(self, theta):
        """Numerically exact gradient.

        Parameters
        ----------
        theta : numpy array
            Parameters.

        Returns
        -------
        grad : numpy array
            Gradient.
        """
        
        # parameters
        unbound        = self.op_driver.ansatz.copy()
        num_parameters = unbound.num_parameters
        
        # operator and innerp
        # bind
        self.op_driver.ansatz  = unbound.bind_parameters(theta)
        self.innerp_driver.lhs = unbound.bind_parameters(theta)
        
        E_A     = self.op_driver.energy_from_measurements()
        E_inner = self.innerp_driver.dot_from_counts()
        
        # get shifted expectations
        E_A_pl     = np.empty(num_parameters)
        E_A_min    = np.empty(num_parameters)
        E_inner_pl = np.empty(num_parameters)
        grad       = np.empty(num_parameters)
        
        # shifted operator plus
        for p in range(num_parameters):
            theta_                = theta.copy()
            theta_[p]            += pi/2.
            self.op_driver.ansatz = unbound.bind_parameters(theta_)
            
            E_A_pl[p]             = self.op_driver.energy_from_measurements()
            
        # shifted operator minus
        for p in range(num_parameters):
            theta_                = theta.copy()
            theta_[p]            -= pi/2.
            self.op_driver.ansatz = unbound.bind_parameters(theta_)
            
            E_A_min[p]            = self.op_driver.energy_from_measurements()
            
        # shifted inner product plus
        for p in range(num_parameters):
            theta_                 = theta.copy()
            theta_[p]             += pi
            self.innerp_driver.lhs = unbound.bind_parameters(theta_)
            
            E_inner_pl[p]          = self.innerp_driver.dot_from_counts()
            
        # put into gradient
        for p in range(num_parameters):
            E_A_prime     = 0.5*(E_A_pl[p] - E_A_min[p])
            E_inner_prime = E_inner_pl[p]
            
            grad[p]       = -0.5*(E_inner*E_inner_prime)/E_A + 0.5*(E_inner**2*E_A_prime)/E_A**2
            
        # unbind
        self.op_driver.ansatz  = unbound
        self.innerp_driver.lhs = unbound
        
        return grad
        
        
        
class Poisson1dVQACost_Sato21_Overlap(PoissonVQACost_OpInnerp):
    """Class preparing circuits for estimating the Dirichlet energy as in [1] via estimates of operator norms and overlaps.
  
    References
    ----------
    [1] Yuki Sato, Ruho Kondo, Satoshi Koide, Hideki Takamatsu and Nobuyuki Imoto,
        "Variational quantum algorithm based on the minimum potential energy
        for solving the Poisson equation", PRA 104, 052409 (2021). 
    """
    
    def energy_from_measurements(self, measurements=[None, None]):
        """Estimate energy/cost functional from measurements.

        Parameters
        ----------
        measurements : list[op_measurements, innerp_measurements]
            Measurements in probability form, in the specified order.
            Default is empty list.

        Returns
        -------
        energy : float
            Energy functional estimate.
        """
        
        E_A     = self.op_driver.energy_from_measurements(measurements[self._slice_op])
        E_inner = self.innerp_driver.dot_from_counts(measurements[self._slice_innerp],
                                                     as_prob=True)
        
        energy  = -0.5*E_inner/E_A
        
        return energy
    
    
    cost_from_measurements = energy_from_measurements
    
    
    def get_circuits_energy(self, og_circuits, theta):
        """Circuits to estimate energy.

        Parameters
        ----------
        og_circuits : list[QuantumCircuit]
            List of original unbound energy circuits.
        theta : numpy array
            Parameters.

        Returns
        -------
        grad_circs : list[QuantumCircuit]
            List of circuits required for estimating the energy.
        """
        
        # bind og_circuits
        num_circs      = len(og_circuits)
        bound_circuits = [None]*num_circs
        
        for c in range(num_circs):
            qc = og_circuits[c]
            if qc is None:
                bound_circuits[c] = None
            else:
                bound_circuits[c] = qc.bind_parameters(theta)
            
        return bound_circuits
    
        
    get_circuits_cost = get_circuits_energy
    
    
    def energy_exact(self, theta):
        """Numerically exact energy.

        Parameters
        ----------
        theta : numpy array
            Parameters.

        Returns
        -------
        energy : float
            Energy value.
        """
        
        # parameters
        unbound        = self.op_driver.ansatz.copy()
        num_parameters = unbound.num_parameters
        
        # bind
        self.op_driver.ansatz  = unbound.bind_parameters(theta[0:num_parameters])
        self.innerp_driver.lhs = unbound.bind_parameters(theta[0:num_parameters])
        
        # estimate
        E_A     = self.op_driver.energy_from_measurements()
        E_inner = self.innerp_driver.dot_from_counts()
        
        # compute
        energy  = -0.5*E_inner/E_A
        
        # unbind
        self.op_driver.ansatz  = unbound
        self.innerp_driver.lhs = unbound
        
        return energy
    
    
    cost_exact = energy_exact
    
    
class Poisson1dVQACost_Sato21_Nonnorm(PoissonVQACost_OpInnerp):
    """Class preparing circuits for estimating the Dirichlet energy as in [1], but without normalization.
    
    References
    ----------
    [1] Yuki Sato, Ruho Kondo, Satoshi Koide, Hideki Takamatsu and Nobuyuki Imoto,
        "Variational quantum algorithm based on the minimum potential energy
        for solving the Poisson equation", PRA 104, 052409 (2021). 
    """
    
    def energy_from_measurements(self, measurements=[None, None, None]):
        """Estimate energy/cost functional from measurements.

        Parameters
        ----------
        measurements : list[counts, norm]
            Measurements in probability form, in the specified order.
            Norm is an estimate/guess for the square root of the norm, can be a positive or negative float.
            Default is empty list.

        Returns
        -------
        energy : float
            Energy functional estimate.
        """
        
        E_A       = self.op_driver.energy_from_measurements(measurements[self._slice_op])
        E_inner   = self.innerp_driver.dot_from_counts(measurements[self._slice_innerp],
                                                         as_prob=True)
        norm_sqrt = measurements[-1]
        energy    = 0.5*norm_sqrt*(norm_sqrt*E_A - 2.*E_inner)
        
        return energy
    
    
    cost_from_measurements = energy_from_measurements
    
    
    def get_circuits_energy(self, og_circuits, theta):
        """Circuits to estimate energy.

        Parameters
        ----------
        og_circuits : list[QuantumCircuit]
            List of original unbound energy circuits.
        theta : numpy array
            Parameters.

        Returns
        -------
        grad_circs : list[QuantumCircuit]
            List of circuits required for estimating the energy.
        """
        
        # parameters
        num_parameters = self._num_parameters
        
        # bind og_circuits
        num_circs      = len(og_circuits)
        bound_circuits = [None]*num_circs
        
        for c in range(num_circs):
            qc = og_circuits[c]
            if qc is None:
                bound_circuits[c] = None
            else:
                bound_circuits[c] = qc.bind_parameters(theta[0:num_parameters])
            
        return bound_circuits
    
        
    get_circuits_cost = get_circuits_energy
    
    
    def get_circuits_gradient(self, og_circuits, theta):
        """Estimate gradient.

        Parameters
        ----------
        og_circuits : list[QuantumCircuit]
            List of original unbound cost circuits.
        theta : numpy array
            Parameters.

        Returns
        -------
        grad_circs : list[QuantumCircuit]
            List of circuits required for estimating the gradient.
        """
        
        # parameters
        num_parameters  = self._num_parameters
        
        # get OG circuits
        op_circuits     = og_circuits[self._slice_op]
        innerp_circuits = og_circuits[self._slice_innerp]
        
        # list format
        if not isinstance(op_circuits, list):
            op_circuits = [op_circuits]
        if not isinstance(innerp_circuits, list):
            innerp_circuits = [innerp_circuits]
        
        len_op      = len(op_circuits)
        len_innerp  = len(innerp_circuits)
        total_og    = len_op + len_innerp
        
        total_grad  = total_og+len_op*2*num_parameters+len_innerp*num_parameters
        
        grad_circs  = [None]*(total_grad)
        
        # estimate for operator and inner product
        idx = 0
        for i in range(total_og):
            if og_circuits[i] is not None:
                grad_circs[idx] = og_circuits[i].bind_parameters(theta[0:num_parameters])
            idx += 1
        
        # estimate for gradient of operator
        for p in range(num_parameters):
            theta_     = theta.copy()
            theta_[p] += pi/2.
            for i in range(len_op):
                if op_circuits[i] is not None:
                    grad_circs[idx] = op_circuits[i].bind_parameters(theta_[0:num_parameters])
                idx += 1
                
        for p in range(num_parameters):
            theta_     = theta.copy()
            theta_[p] -= pi/2.
            for i in range(len_op):
                if op_circuits[i] is not None:
                    grad_circs[idx] = op_circuits[i].bind_parameters(theta_[0:num_parameters])
                idx            += 1
            
        # estimate for gradient of inner product
        for p in range(num_parameters):
            theta_     = theta.copy()
            theta_[p] += pi
            for i in range(len_innerp):
                if innerp_circuits[i] is not None:
                    grad_circs[idx] = innerp_circuits[i].bind_parameters(theta_[0:num_parameters])
                idx            += 1
                
        return grad_circs
    
    
    def gradient_from_measurements(self, measurements=[None, None, None]):
        """Estimate gradient from experimental measurements.

        Parameters
        ----------
        measurements : list[Counts]
            List of experimental counts in probability form.
            Last is parameter value (for this method).
            The default is [None, None, None].

        Returns
        -------
        grad : numpy array
            Gradient.
        """
        
        # parameters
        num_parameters = self._num_parameters
        norm_sqrt      = measurements[-1]
        
        # get OG circuits
        op_meas     = measurements[self._slice_op]
        innerp_meas = measurements[self._slice_innerp]
        
        len_op = len_innerp = 1
        if isinstance(op_meas, list):
            len_op = len(op_meas)
        if isinstance(innerp_meas, list):
            len_innerp = len(innerp_meas)

        total_og    = len_op + len_innerp
        total_grad  = total_og+len_op*2*num_parameters+len_innerp*num_parameters
   
        if measurements is None:
            measurements = [None]*total_grad
        
        E_A     = self.op_driver.energy_from_measurements(op_meas)
        E_inner = self.innerp_driver.dot_from_counts(innerp_meas, as_prob=True)
        
        # get shifted expectations
        E_A_pl     = np.empty(num_parameters)
        E_A_min    = np.empty(num_parameters)
        E_inner_pl = np.empty(num_parameters)
        grad       = np.empty(num_parameters+1)
        grad[-1]   = norm_sqrt*E_A - E_inner
        
        # shifted operator plus
        idx0 = len_op+len_innerp
        for p in range(num_parameters):
            idx       = None
            idx_start = idx0 + p*len_op
            idx_end   = idx0 + (p+1)*len_op
            if len_op == 1:
                idx = idx_start
            else:
                idx = slice(idx_start, idx_end)

            E_A_pl[p] = self.op_driver.energy_from_measurements(measurements[idx])
            
        # shifted operator minus
        idx0 = len_op+len_innerp+num_parameters*len_op
        for p in range(num_parameters):
            idx       = None
            idx_start = idx0 + p*len_op
            idx_end   = idx0 + (p+1)*len_op
            if len_op == 1:
                idx = idx_start
            else:
                idx = slice(idx_start, idx_end)
            E_A_min[p] = self.op_driver.energy_from_measurements(measurements[idx])
            
        # shifted inner product plus
        idx0 = len_op+len_innerp+2*num_parameters*len_op
        for p in range(num_parameters):
            idx       = None
            idx_start = idx0 + p*len_innerp
            idx_end   = idx0 + (p+1)*len_innerp
            if len_innerp == 1:
                idx = idx_start
            else:
                idx = slice(idx_start, idx_end)

            E_inner_pl[p] = self.innerp_driver.dot_from_counts(measurements[idx],
                                                               as_prob=True)
            
        # put into gradient
        for p in range(num_parameters):
            E_A_prime     = 0.5*(E_A_pl[p] - E_A_min[p])
            E_inner_prime = 0.5*E_inner_pl[p]
            grad[p]       = 0.5*norm_sqrt**2*E_A_prime - norm_sqrt*E_inner_prime
            
            
        return grad
    
    
    def energy_exact(self, theta):
        """Numerically exact energy.

        Parameters
        ----------
        theta : numpy array
            Parameters.

        Returns
        -------
        energy : float
            Energy value.
        """
        
        # parameters
        unbound        = self.op_driver.ansatz.copy()
        num_parameters = unbound.num_parameters
        norm_sqrt      = theta[-1]
        
        # bind
        self.op_driver.ansatz  = unbound.bind_parameters(theta[0:num_parameters])
        self.innerp_driver.lhs = unbound.bind_parameters(theta[0:num_parameters])
        
        # estimate
        E_A     = self.op_driver.energy_from_measurements()
        E_inner = self.innerp_driver.dot_from_counts()
        
        # compute
        energy  = 0.5*norm_sqrt*(norm_sqrt*E_A - 2.*E_inner)
        
        # unbind
        self.op_driver.ansatz  = unbound
        self.innerp_driver.lhs = unbound
        
        return energy
    
    
    cost_exact = energy_exact
    
    
    def gradient_exact(self, theta):
        """Numerically exact gradient.

        Parameters
        ----------
        theta : numpy array
            Parameters.

        Returns
        -------
        grad : numpy array
            Gradient.
        """
        
        # parameters
        unbound        = self.op_driver.ansatz.copy()
        num_parameters = unbound.num_parameters
        norm_sqrt      = theta[-1]
        
        # operator and innerp
        # bind
        self.op_driver.ansatz  = unbound.bind_parameters(theta[0:num_parameters])
        self.innerp_driver.lhs = unbound.bind_parameters(theta[0:num_parameters])
        
        E_A     = self.op_driver.energy_from_measurements()
        E_inner = self.innerp_driver.dot_from_counts()
        
        # get shifted expectations
        E_A_pl     = np.empty(num_parameters)
        E_A_min    = np.empty(num_parameters)
        E_inner_pl = np.empty(num_parameters)
        grad       = np.empty(num_parameters+1)
        
        grad[-1]   = norm_sqrt*E_A - E_inner
        
        # shifted operator plus
        for p in range(num_parameters):
            theta_                = theta.copy()
            theta_[p]            += pi/2.
            self.op_driver.ansatz = unbound.bind_parameters(theta_[0:num_parameters])
            
            E_A_pl[p]             = self.op_driver.energy_from_measurements()
            
        # shifted operator minus
        for p in range(num_parameters):
            theta_                = theta.copy()
            theta_[p]            -= pi/2.
            self.op_driver.ansatz = unbound.bind_parameters(theta_[0:num_parameters])
            
            E_A_min[p]            = self.op_driver.energy_from_measurements()
            
        # shifted inner product plus
        for p in range(num_parameters):
            theta_                 = theta.copy()
            theta_[p]             += pi
            self.innerp_driver.lhs = unbound.bind_parameters(theta_[0:num_parameters])
            
            E_inner_pl[p]          = self.innerp_driver.dot_from_counts()
            
        # put into gradient
        for p in range(num_parameters):
            E_A_prime     = 0.5*(E_A_pl[p] - E_A_min[p])
            E_inner_prime = 0.5*E_inner_pl[p]
            grad[p]       = 0.5*norm_sqrt**2*E_A_prime - norm_sqrt*E_inner_prime
        
        # unbind
        self.op_driver.ansatz  = unbound
        self.innerp_driver.lhs = unbound
        
        return grad
            
    
    
class PoissonVQACost_BravoPrieto20_Global(PoissonVQACost_OpInnerp):
    """Class for numerically exact evaluation of the global cost function from [1] for the Poisson problem.
    
    References
    ----------
    [1] Carlos Bravo-Prieto, Ryan LaRose, M. Cerezo, Yigit Subasi, Lukasz Cincio and Patrick J. Coles,
        "Variational quantum linear solver", arXiv:1909.05820v2 (2020)
    """
    
    
    def __init__(self, op_driver, innerp_driver):
        
        self._Uf = None
        self._PG = None
        self._A  = None
        
        super(PoissonVQACost_BravoPrieto20_Global, self).__init__(op_driver, innerp_driver)
    
    
    def get_circuits(self):
        return [None]
    
    
    def energy_from_measurements(self, measurements=None):
        return None
    
    
    cost_from_measurements = energy_from_measurements
    
    
    def energy_exact(self, theta):
        
        ansatz     = self.op_driver.ansatz
        rhs        = self.innerp_driver.rhs
        bound      = ansatz.bind_parameters(theta)
        num_qubits = ansatz.num_qubits
        N          = 2**num_qubits
        
        psi    = self._simulate_ansatz(bound)
        Uf     = self._simulate_Uf(rhs)
        
        PG     = self._assemble_global_projector(num_qubits)
        A      = self._assemble_dense_Poisson_matrix(num_qubits)
        
        HG     = A @ Uf @ (np.eye(N) - PG) @ Uf.transpose() @ A
        nom    = np.real(np.vdot(psi, HG @ psi))
        denom  = np.real(np.vdot(psi, A @ A @ psi))
        energy = nom/denom
        
        return energy
    
    
    cost_exact = energy_exact
    
    
    def gradient_exact(self, theta):
        
        ansatz         = self.op_driver.ansatz
        rhs            = self.innerp_driver.rhs
        bound          = ansatz.bind_parameters(theta)
        num_qubits     = ansatz.num_qubits
        num_parameters = ansatz.num_parameters
        grad           = np.empty(num_parameters)
        N              = 2**num_qubits
        
        psi    = self._simulate_ansatz(bound)
        Uf     = self._simulate_Uf(rhs)
        
        PG     = self._assemble_global_projector(num_qubits)
        A      = self._assemble_dense_Poisson_matrix(num_qubits)
        Asq    = A @ A
        
        HG     = A @ Uf @ (np.eye(N) - PG) @ Uf.transpose() @ A
        Hpsi   = np.real(np.vdot(psi, HG @ psi))
        Apsi   = np.real(np.vdot(psi, Asq @ psi))
        
        for i in range(num_parameters):
            theta_i     = np.copy(theta)
            theta_i[i] += pi
            bound       = ansatz.bind_parameters(theta_i)
            psi_i       = self._simulate_ansatz(bound)
            
            Hpsi_i      = np.real(0.5*(np.vdot(psi_i, HG @ psi) +
                                       np.vdot(psi, HG @ psi_i)))
            Apsi_i      = np.real(0.5*(np.vdot(psi_i, Asq @ psi) +
                                       np.vdot(psi, Asq @ psi_i)))
            grad[i]     = (Hpsi_i*Apsi-Hpsi*Apsi_i)/Apsi**2
            
        return grad
    
    
    def _simulate_ansatz(self, bound):
        
        sim = StatevectorSimulator()
        psi = sim.run(bound).result().get_statevector().data
        
        return psi
    
    
    def _simulate_Uf(self, rhs):
        
        if self._Uf is not None:
            return self._Uf
        else:
            sim      = UnitarySimulator()
            Uf       = sim.run(rhs).result().get_unitary().data
            self._Uf = Uf
            
            return Uf
        
    
    def _assemble_global_projector(self, num_qubits):
        
        if self._PG is not None:
            return self._PG
        else:
            N        = 2**num_qubits
            PG       = np.zeros((N, N))
            PG[0][0] = 1.
            self._PG = PG
            
            return PG
        
        
    def _assemble_dense_Poisson_matrix(self, num_qubits):
        
        if self._A is not None:
            return self._A
        else:
            N = 2**num_qubits
            A = np.zeros((N, N))
            for i in range(1, N-1):
                A[i][i]    = 2.
                A[i][i-1]  = -1.
                A[i][i+1]  = -1.
                
            A[0][0]   = 2.
            A[0][1]   = -1.

            A[N-1][N-2] = -1.
            A[N-1][N-1] = 2.
                
            self._A = A
            
            return A
        
        
        
class PoissonVQACost_BravoPrieto20_Local(PoissonVQACost_OpInnerp):
    """Class for numerically exact evaluation of the local cost function from [1] for the Poisson problem.
    
    References
    ----------
    [1] Carlos Bravo-Prieto, Ryan LaRose, M. Cerezo, Yigit Subasi, Lukasz Cincio and Patrick J. Coles,
        "Variational quantum linear solver", arXiv:1909.05820v2 (2020)
    """
    
    
    def __init__(self, op_driver, innerp_driver):
        
        self._Uf = None
        self._PL = None
        self._A  = None
        
        super(PoissonVQACost_BravoPrieto20_Local, self).__init__(op_driver, innerp_driver)
    
    
    def get_circuits(self):
        return [None]
    
    
    def energy_from_measurements(self, measurements=None):
        return None
    
    
    cost_from_measurements = energy_from_measurements
    
    
    def energy_exact(self, theta):
        
        ansatz     = self.op_driver.ansatz
        rhs        = self.innerp_driver.rhs
        bound      = ansatz.bind_parameters(theta)
        num_qubits = ansatz.num_qubits
        N          = 2**num_qubits
        
        psi    = self._simulate_ansatz(bound)
        Uf     = self._simulate_Uf(rhs)
        
        PL     = self._assemble_local_projector(num_qubits)
        A      = self._assemble_dense_Poisson_matrix(num_qubits)
        
        HL     = A @ Uf @ (np.eye(N) - PL) @ Uf.transpose() @ A
        nom    = np.real(np.vdot(psi, HL @ psi))
        denom  = np.real(np.vdot(psi, A @ A @ psi))
        energy = nom/denom
        
        return energy
    
    
    cost_exact = energy_exact
    
    
    def gradient_exact(self, theta):
        
        ansatz         = self.op_driver.ansatz
        rhs            = self.innerp_driver.rhs
        bound          = ansatz.bind_parameters(theta)
        num_qubits     = ansatz.num_qubits
        num_parameters = ansatz.num_parameters
        grad           = np.empty(num_parameters)
        N              = 2**num_qubits
        
        psi    = self._simulate_ansatz(bound)
        Uf     = self._simulate_Uf(rhs)
        
        PL     = self._assemble_local_projector(num_qubits)
        A      = self._assemble_dense_Poisson_matrix(num_qubits)
        Asq    = A @ A
        
        HL     = A @ Uf @ (np.eye(N) - PL) @ Uf.transpose() @ A
        Hpsi   = np.real(np.vdot(psi, HL @ psi))
        Apsi   = np.real(np.vdot(psi, Asq @ psi))
        
        for i in range(num_parameters):
            theta_i     = np.copy(theta)
            theta_i[i] += pi
            bound       = ansatz.bind_parameters(theta_i)
            psi_i       = self._simulate_ansatz(bound)
            
            Hpsi_i      = np.real(0.5*(np.vdot(psi_i, HL @ psi) +
                                       np.vdot(psi, HL @ psi_i)))
            Apsi_i      = np.real(0.5*(np.vdot(psi_i, Asq @ psi) +
                                       np.vdot(psi, Asq @ psi_i)))
            grad[i]     = (Hpsi_i*Apsi-Hpsi*Apsi_i)/Apsi**2
            
        return grad
    
    
    def _simulate_ansatz(self, bound):
        
        sim = StatevectorSimulator()
        psi = sim.run(bound).result().get_statevector().data
        
        return psi
    
    
    def _simulate_Uf(self, rhs):
        
        if self._Uf is not None:
            return self._Uf
        else:
            sim      = UnitarySimulator()
            Uf       = sim.run(rhs).result().get_unitary().data
            self._Uf = Uf
            
            return Uf
        
    
    def _assemble_local_projector(self, num_qubits):
        
        if self._PL is not None:
            return self._PL
        else:
            Locj       = np.zeros((2, 2))
            Locj[0][0] = 1.
            PL         = np.kron(np.eye(2**(num_qubits-1)), Locj)
            
            for j in range(num_qubits-1, 0, -1):
                Pj = np.eye(2)

                for i in range(num_qubits-1, 0, -1):
                    if i==j:
                        Pj = np.kron(Locj, Pj)
                    else:
                        Pj = np.kron(np.eye(2), Pj)
        
                PL += Pj
        
            PL      *= 1./num_qubits
            self._PL = PL

            return PL
        
        
    def _assemble_dense_Poisson_matrix(self, num_qubits):
        
        if self._A is not None:
            return self._A
        else:
            N = 2**num_qubits
            A = np.zeros((N, N))
            for i in range(1, N-1):
                A[i][i]    = 2.
                A[i][i-1]  = -1.
                A[i][i+1]  = -1.
                
            A[0][0]   = 2.
            A[0][1]   = -1.

            A[N-1][N-2] = -1.
            A[N-1][N-1] = 2.
                
            self._A = A
            
            return A
    
    
    
DRIVERS = {'Sato21_Innerp'       : Poisson1dVQACost_Sato21_Innerp,
           'Sato21_Overlap'      : Poisson1dVQACost_Sato21_Overlap,
           'Sato21_Nonnorm'      : Poisson1dVQACost_Sato21_Nonnorm,
           'BravoPrieto20_Global': PoissonVQACost_BravoPrieto20_Global,
           'BravoPrieto20_Local' : PoissonVQACost_BravoPrieto20_Local}



def get_poisson_cost_driver(name):
    """Return Poisson cost driver if available.

    Parameters
    ----------
    name : str
        Name of driver.

    Raises
    ------
    ValueError
        Raises error if driver name not recognized.

    Returns
    -------
    Class
        Constructor class for chosen driver.

    """
    
    if name not in DRIVERS:
        message = "Poisson cost method not implemented. Available drivers: "
        message += str(list(DRIVERS.keys()))
        raise ValueError(message)
        
    return DRIVERS[name]