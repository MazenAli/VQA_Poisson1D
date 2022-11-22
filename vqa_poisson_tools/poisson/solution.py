"""Numerically exact solution to the Poisson Problem"""

from abc import ABC, abstractmethod

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from qiskit.providers.aer import StatevectorSimulator


class SolvePoisson(ABC):
    """Abstract class for numerically exact solutions of the Poisson problem."""
    
    def __init__(self):
        
        pass
    
    
    @abstractmethod
    def solve(self):
        
        pass
    
    
    
class SolvePoisson1D(SolvePoisson):
    """Class computing exact solution to 1D Poisson.
    
    Parameters
    ----------
    rhs_circuit : QuantumCircuit
        Circuit preparing the right-hand-side.
        
    """
    
    def __init__(self, rhs_circuit):
        
        self._rhs_circuit = rhs_circuit
        self._A           = None
        self._rhs_vec     = None
        self._solution    = None
        self._num_qubits  = rhs_circuit.num_qubits
        self._state_sim   = StatevectorSimulator()
        
        
    def solve(self, recompute=False):
        """Return (non-normalized) solution vector.

        Parameters
        ----------
        recompute : bool
            Force recomputing even if rhs, operator and/or solution already computed.
            The default is False.
            
        Returns
        -------
        numpy.array
            Complex solution vector.
        """
            
        if self._solution is None or recompute:
            
            if self._rhs_vec is None or recompute:
                self._rhs_vec = self._simulate_vector(self._rhs_circuit)
            
            if self._A is None or recompute:
                self._assemble_operator()
            
            self._solution = spsolve(self._A.tocsc(), self._rhs_vec)
            
        return self._solution
        
    
    @property
    def rhs_circuit(self):
        """Circuit preparing the right-hand-side (QuantumCircuit)."""
        return self._rhs_circuit
    
    
    @property
    def num_qubits(self):
        """Number of qubits (int)."""
        return self._num_qubits
    
    
    @property
    def A(self):
        """System (stiffness) matrix (sparse matrix)."""
        return self._A
    
    
    @property
    def state_sim(self):
        """State vector simulator for running circuits (object)."""
        return self._state_sim
    
    
    @property
    def rhs_vec(self):
        """Complex right-hand-side vector (numpy.array)."""
        return self._rhs_vec
    
    
    @property
    def solution(self):
        """Complex (non-normalized) solution vector (numpy.array)"""
        return self._solution
    
    
    def _assemble_operator(self):
        N         = 2**self._num_qubits
        diagonals = [[-1]*(N-1), [2]*N, [-1]*(N-1)]
        A         = diags(diagonals, [-1, 0, 1])
        self._A   = A
        
        return A
    
    
    def _simulate_vector(self, circuit):
        statevector = self._state_sim.run(circuit).result().get_statevector().data
        return statevector