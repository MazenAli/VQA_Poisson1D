# -*- coding: utf-8 -*-
"""
Convert Qiskit circuits to probability distributions
"""

from abc import ABC, abstractmethod

from qiskit.providers.aer import StatevectorSimulator


class Circ2Dist(ABC):
    """Abstract class for converting circuits to probability distributions."""    
    
    def __init__(self):
        
        pass
    
    
    @abstractmethod
    def get_distribution(self):
        
        pass
    

class StatevecSimDist(Circ2Dist):
    """Class for simulating distributions with Qiskit's statevector simulator.
    
    Example 1:
        distr = StatevecSimDist(qc)
        probs = distr.simulate_distribution() # returns dictionary of probabilities
        
    Example 2:
        distr = StatevecSimDist([qc1, qc2])
        probs = distr.simulate_distribution(as_dict = False) # returns list of 2 numpy arrays
    
    Parameters
    ----------
    circuits : QuantumCircuit or list[QuantumCircuit]
        Circuits preparing the quantum state (w/out measurement instructions).
    """
    
    def __init__(self, circuits):
        
        self._circuits  = circuits
        self._dist_vec  = None
        self._dist_dict = None
        
        self._num_circs = 1
        self._is_list   = False
        if type(circuits)==list:
            self._is_list   = True
            self._num_circs = len(circuits)
        
    
    def simulate_distribution(self, qargs=None, as_dict=True):
        """Simulate the probability distribution from the quantum state.

        Parameters
        ----------
        qargs   : list
            Subsystems to return probabilities for. Default is all.
        as_dict : bool, optional
            Return dictionary of probabilities or vector if False. The default is True.

        Returns
        -------
        dict, list[dict] or np.array, list[np.array]
            Simulated probability distribution.
        """
        
        state_sim = StatevectorSimulator()
        
        vectors = None
        dicts   = None
        if self._is_list:
            
            vectors = [None]*self._num_circs
            dicts   = [None]*self._num_circs
            for q in range(self._num_circs):
                statevec   = state_sim.run(self._circuits[q]).result().get_statevector()
                probs_vec  = statevec.probabilities(qargs)
                probs_dict = statevec.probabilities_dict(qargs)
                
                vectors[q] = probs_vec
                dicts[q]   = probs_dict
                
        else:
            statevec = state_sim.run(self._circuits).result().get_statevector()
            vectors  = statevec.probabilities(qargs)
            dicts    = statevec.probabilities_dict(qargs)
            
        self._dist_vec  = vectors
        self._dist_dict = dicts
        
        if as_dict:
            return dicts
        else:
            return vectors
        
    
    def get_distribution(self, as_dict = True):
        """Return the saved probability distribution.
        Returns None if simulate_distribution not previously called.

        Parameters
        ----------
        as_dict : bool, optional
            Return dictionary of probabilities or vector if False. The default is True.

        Returns
        -------
        dict, list[dict] or np.array, list[np.array]
            Simulated probability distribution.
       """
        
        if as_dict:
            return self.get_distribution_dict()
        else:
            return self.get_distribution_vec()
        
        
    def get_distribution_dict(self):
        """Return the saved probability distribution as a dictionary.
        Returns None if simulate_distribution not previously called.

        Returns
        -------
        dict, list[dict]
            Simulated probability distribution.
        """
        
        return self._dist_dict
   
    
    def get_distribution_vec(self):
        """Return the saved probability distribution as a vector.
        Returns None if simulate_distribution not previously called.

        Returns
        -------
        np.array, list[np.array]
            Simulated probability distribution.
        """
        
        return self._dist_vec