"""
Wrapper classes to combine backend calls and post processing into one call (suitable for, e.g., runtime).
"""


from abc import ABC, abstractmethod


class CostEstimator(ABC):
    """Abstract class for cost estimators.
    Basic functionality is to estimate the energy and possibly gradients
    for a provided parameter vector.
    Wraps backend (sampler) calls and post processing from other classes.
    
    Parameters
    ----------
    cost_driver : object
        Cost driver for assembling circuits and estimating from measurements.
        
    sampler : obj
        Sampler for running the circuits.
    """
    
    
    def __init__(self, circuits, cost_driver, sampler):
        self._circuits    = circuits
        self._cost_driver = cost_driver
        self._sampler     = sampler
        
        self._energy_log  = []
        self._grad_log    = []
    
    
    @abstractmethod
    def energy(self, theta):
        """Estimate energy for the given ansatz paramaters.

        Parameters
        ----------
        theta : numpy array
            Vector of parameters for the provided circuits.

        Returns
        -------
        float
            Energy (or cost) value.

        """
        
        pass
    
    
    cost = energy
    
    
    @abstractmethod
    def gradient(self, theta):
        """Estimate the gradient of the Dirichlet energy
        for the given ansatz paramaters.

        Parameters
        ----------
        theta : numpy.array
            Vector of parameters for the provided circuits.

        Returns
        -------
        numpy.array
            Gradient vector.
        """
        
        pass


    @property
    def circuits(self):
        """Circuits require for estimating cost and gradients."""
        return self._circuits
    
    
    @property
    def cost_driver(self):
        """Cost driver for estimating from measurements (object)."""
        return self._cost_driver
    
    
    @property
    def sampler(self):
        """Sampler for running circuits (object)."""
        return self._sampler
    
    
    @property
    def energy_log(self):
        """Log of all energy (cost) calls (list)."""
        return self._energy_log
    
    
    @property
    def grad_log(self):
        """Log of all gradient calls (list)."""
        return self._grad_log



class MeasCostEstimator(CostEstimator):
    """Measurement based cost estimator."""

            
    def energy(self, theta):
        
        bound_circs   = self._cost_driver.get_circuits_energy(self._circuits, theta)
        sampled_probs = self._sampler.sample(bound_circs)
        energy        = self._cost_driver.energy_from_measurements(sampled_probs)
        
        self._energy_log.append({'theta': theta.copy(),
                                 'energy': energy})
        
        return energy
    
    
    cost = energy
    
    
    def gradient(self, theta):
        
        bound_circs   = self._cost_driver.get_circuits_gradient(self._circuits, theta)
        sampled_probs = self._sampler.sample(bound_circs)
        gradient      = self._cost_driver.gradient_from_measurements(sampled_probs)
        
        self._grad_log.append({'theta': theta.copy(),
                               'grad': gradient})
        
        return gradient



class MeasCostNonnormEstimator(CostEstimator):
    """Measurement based cost estimator with last measurement equal parameter."""

            
    def energy(self, theta):
        
        bound_circs   = self._cost_driver.get_circuits_energy(self._circuits, theta)
        sampled_probs = self._sampler.sample(bound_circs)
        energy        = self._cost_driver.energy_from_measurements(sampled_probs+[theta[-1]])
        
        self._energy_log.append({'theta': theta.copy(),
                                 'energy': energy})
        
        return energy
    
    
    cost = energy
    
    
    def gradient(self, theta):
        
        bound_circs   = self._cost_driver.get_circuits_gradient(self._circuits, theta)
        sampled_probs = self._sampler.sample(bound_circs)
        gradient      = self._cost_driver.gradient_from_measurements(sampled_probs+[theta[-1]])
        
        self._grad_log.append({'theta': theta.copy(),
                               'grad': gradient})
        
        return gradient



class ExactCostEstimator(CostEstimator):
    """Exact cost estimator."""
    
    
    def __init__(self, cost_driver):
        self._circuits    = None
        self._cost_driver = cost_driver
        self._sampler     = None
        
        self._energy_log  = []
        self._grad_log    = []
        

    def energy(self, theta):
        
        energy        = self._cost_driver.cost_exact(theta)
        
        self._energy_log.append({'theta': theta.copy(),
                                 'energy': energy})
        
        return energy
    
    
    cost = energy
    
    
    def gradient(self, theta):
        
        gradient      = self._cost_driver.gradient_exact(theta)
        
        self._grad_log.append({'theta': theta.copy(),
                               'grad': gradient})
        
        return gradient


    
DRIVERS = {'Measurement'       : MeasCostEstimator,
           'MeasurementNonnorm': MeasCostNonnormEstimator,
           'Exact'             : ExactCostEstimator}



def get_cost_estimator_driver(name):
    """Return cost estimator driver if available.

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
        message = "Cost estimator method not implemented. Available drivers: "
        message += str(list(DRIVERS.keys()))
        raise ValueError(message)
        
    return DRIVERS[name]