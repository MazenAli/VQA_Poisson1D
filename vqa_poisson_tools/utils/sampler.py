"""
Wrapper classes to sample circuits.
"""


from abc import ABC, abstractmethod
from qiskit.providers.aer import StatevectorSimulator
import mthree
from vqa_poisson_tools.utils.utils import counts2probs



class Sampler(ABC):
    """Abstract class for samplers. Wrappers for backends for convenient circuit sampling.
    
    Parameters
    ----------
    backend : object
        Backend device.
        
    options : dict
        Run options.
    """
    
    def __init__(self, backend, **options):
        
        self._backend = backend
        self._options = options
        
        
    @abstractmethod
    def sample(self, circuits):
        
        pass
        
    
        
class VanillaSampler(Sampler):
    """A simple circuit sampler supporting raw and mitigated sampling.
    
    Parameters
    ----------
    backend : object
        Backend for running circuits.
        
    options : dict
        Run options for backend and measurement error mitigator (optional).
    """
    
    
    def __init__(self, backend, **options):
        
        self._backend     = backend
        self._mit         = options.pop('mitigator', None)
        self._run_options = options


    def raw_sample(self, circuits):
        """Sample circuits without measurement error mitigation.
        
        Parameters
        ----------
        circuits : list of QuantumCircuit
            Circuits to sample.

        Returns
        -------
        probabilities : dict
            Probabilities of outcomes.
        """
        
        # run circuits
        probs = None
        if isinstance(self._backend, StatevectorSimulator):
            probs = self._backend.run(circuits).result().get_statevector().probabilities_dict()
        else:
            job    = self._backend.run(circuits, **self._run_options)
            counts = job.result().get_counts()
            probs  = counts2probs(counts)
        
        return probs

    
    def mitigated_sample(self, circuits):
        """Sample circuits with measurement error mitigation.
        
        Parameters
        ----------
        circuits : list of QuantumCircuit
            Circuits to sample.

        Raises
        ------
        ValueError
            Raises error if error mitigator not previously specified.

        Returns
        -------
        probabilities : dict
            Probabilities of outcomes.
        
        """
        
        if self.mit is None:
            raise ValueError("Mitigator not specified")
            
        mapping = mthree.utils.final_measurement_mapping(circuits)
        job     = self._backend.run(circuits, **self._run_options)
        counts  = job.result().get_counts()
        quasi   = self._mit.apply_correction(counts, mapping)
        
        return quasi
        
    
    def sample(self, circuits):
        """Sample circuits.
        
        Parameters
        ----------
        circuits : list of QuantumCircuit
            Circuits to sample.

        Returns
        -------
        probabilities : dict
            Probabilities of outcomes.
        """
        
        if self._mit is None:
            return self.raw_sample(circuits)
        
        return self.mitigated_sample(circuits)
        
    
    @property
    def backend(self):
        """Backend for executing circuits (object)."""
        return self._backend
    
    
    @property
    def mit(self):
        """Measurement error mitigator (object)."""
        return self._mit
    
    
    @mit.setter
    def mit(self, mit_):
        self._mit = mit_
    
    
    @property
    def run_options(self):
        """Run options for backend (dict)."""
        return self._run_options