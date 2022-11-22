from abc import ABC, abstractmethod

from math import pi
import numpy as np
from scipy.sparse import diags

import time

from qiskit import QuantumCircuit, ClassicalRegister, Aer
import qiskit.algorithms.optimizers as qiskit_opt
from qiskit.algorithms.optimizers import QNSPSA
from qiskit.providers.aer import StatevectorSimulator, UnitarySimulator
from qiskit.circuit.library import MCMT
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.opflow import PauliSumOp, AbelianGrouper

import mthree



# %% UTILS
#
"""
Frequently used utility functions
"""


def counts2probs(counts):
    """Convert a dictionary of experimental counts or a list of
    dictionaries to probabilities.

    Parameters
    ----------
    counts : Counts, list[Counts] or dict, list[dict]
        Experimental counts.

    Returns
    -------
    dict or list[dict]
        Experimental probabilities.
    """
    
    probs   = None
    is_list = (type(counts)==list)
    
    if is_list:
        num   = len(counts)
        probs = [None]*num
        for i in range(num):
            counts_  = counts[i]
            shots    = sum(counts_.values())
            probs_   = {b: c/shots for b, c in counts_.items()}
            probs[i] = probs_
    else:
        shots = sum(counts.values())
        probs = {b: c/shots for b, c in counts.items()}
        
    
    return probs



# %% SAMPLER
#

"""
Wrapper classes to sample circuits.
"""



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



# %% RHS
#
"""
Circuits for the RHS
"""


def hnx(num_qubits):
    """Prepares the circuit for |f> = H^n(X \otimes I^{n-1})|0>

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    Returns
    -------
    Uf : QuantumCircuit.

    """
    
    
    Uf = QuantumCircuit(num_qubits, name='HnX')
    Uf.x(num_qubits-1)

    for i in range(num_qubits):
        Uf.h(i)
    
    return Uf


def hn(num_qubits):
    """Prepares the circuit for |f> = H^n|0>

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    Returns
    -------
    Uf : QuantumCircuit.

    """
    
    
    Uf = QuantumCircuit(num_qubits, name='Hn')

    for i in range(num_qubits):
        Uf.h(i)
    
    return Uf


def idn(num_qubits):
    """Prepares the circuit for |f> = |0>

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    Returns
    -------
    Uf : QuantumCircuit.

    """
    
    
    Uf = QuantumCircuit(num_qubits, name='Idn')

    return Uf



def get_rhs(name):
    """Return rhs driver if available.

    Parameters
    ----------
    name : str
        Name of rhs.

    Raises
    ------
    ValueError
        Raises error if rhs name not recognized.

    Returns
    -------
    Function
        Function for chosen rhs.
    """

    RHS = {'id': idn,
           'hn': hn,
           'hnx': hnx}
    
    if name not in RHS:
        message = "Rhs not implemented. Available rhs: "
        message += str(list(RHS.keys()))
        raise ValueError(message)
        
    return RHS[name]



# %% INNER PRODUCT
#
"""
Prepare Qiskit circuits for estimating inner products.
"""


class InnerProduct(ABC):
    """Abstract class for VQA inner product drivers.
    Basic functionality is to prepare circuits for measurement of <lhs|rhs>.
    
    Parameters
    ----------
    lhs : QuantumCircuit
        Left hand side.
    rhs : QuantumCircuit
        Right hand side.
    """
    
    def __init__(self, lhs, rhs):
        
        self._lhs = lhs
        self._rhs = rhs
    
    
    @abstractmethod        
    def dot_circuit(self):
        """Prepare circuits for <lhs|rhs>."""
        
        pass
    
    
    get_circuits = dot_circuit
    
    
    @abstractmethod
    def dot_from_counts(self, counts, as_prob = False):
        """Estimate inner product from counts.

        Parameters
        ----------
        counts : Counts
            Experimental counts.
        as_prob : bool, optional
            The counts are in probability format (normalized). The default is False.

        Returns
        -------
        float
            Estimate for inner product
        """
        
        pass
    
    
    @property
    def lhs(self):
        """Left hand side (QuantumCircuit)."""
        return self._lhs
    
    
    @property
    def rhs(self):
        """Right hand side (QuantumCircuit)."""
        return self._rhs
    
    
    @lhs.setter
    def lhs(self, lhs_):
        self._lhs = lhs_
        
        
    @rhs.setter
    def rhs(self, rhs_):
        self._rhs = rhs_


    
class InnerProductHTest(InnerProduct):
    """Class estimating the inner product via the Hadamard test."""
    
    def __init__(self, lhs, rhs):
        
        self._lhs    = lhs
        self._rhs    = rhs
        
        n1 = lhs.num_qubits
        n2 = rhs.num_qubits
        
        if n1 != n2:
            msg  = "Qubit size of lhs (n="
            msg += str(n1)
            msg += ") does not match qubit size of rhs (n="
            msg += str(n2) + ")"
            
            raise ValueError(msg)
            
            
    def dot_circuit(self, w_meas = True):
        """Prepare circuits for <lhs|rhs>.

        Parameters
        ----------
        w_meas : bool, optional
            Add measurement instructions. The default is True.

        Returns
        -------
        innerp : QuantumCircuit
            Circuit for <lhs|rhs>.
        """
        
        num_qubits = self._lhs.num_qubits
        lhs_rhs    = bell_state(self._lhs, self._rhs)
        
        # rotate into X basis
        num_clbits = 0
        if w_meas:
            num_clbits = 1
            
        innerp = QuantumCircuit(num_qubits+1, num_clbits, name='inner_product')
        innerp = innerp.compose(lhs_rhs, list(range(num_qubits+1)))
        innerp.h(num_qubits)
        
        if w_meas:
            innerp.measure(num_qubits, 0)
        
        return innerp
    
    
    get_circuits = dot_circuit
    
    
    @staticmethod
    def dot_from_counts(counts, as_prob=False):
        """Estimate inner product from counts.

        Parameters
        ----------
        counts : Counts
            Experimental counts.
        as_prob : bool, optional
            The counts are in probability format (normalized). The default is False.

        Raises
        ------
        ValueError
            Raises error if outcome bitstrings have length unequal to 1.

        Returns
        -------
        expec : float
            Inner product estimate.

        """
        
        # check length
        num_clbits = len(next(iter(counts)))
        
        if num_clbits != 1:
            raise ValueError("Length of binary outcome for this experiment \
                             must be 1")
        
        # convert to probabilities
        probs = counts
        if not as_prob:
            probs = counts2probs(counts)
            
        # compute expectation
        p0    = 0.
        if '0' in probs.keys():
            p0 += probs['0']
        p1    = 1. - p0
        expec = p0 - p1
        
        return expec
    
    
    @staticmethod
    def dot_from_amplitude(p1):
        """Inner product from probability of |1> state.

        Parameters
        ----------
        p1 : float
            Probability of measuring the |1> state.

        Returns
        -------
        expec : float
            Inner product estimate.

        """
        
        expec = 1. - 2.*p1
        
        return expec
    
    
    def state_preparation(self):
        """State preparation circuit for QAE.

        Returns
        -------
        QuantumCircuit
            State preparation circuit.
        """
        
        return self.dot_circuit(w_meas=False)
    
    
    def S0(self):
        """Circuit for adding phase to the all-0 state.

        Returns
        -------
        S0_qc : QuantumCircuit
            Circuit for the all-0 state.
        """
        
        num_qubits = self._lhs.num_qubits
        S0_qc      = QuantumCircuit(num_qubits+1)
        
        for q in range(num_qubits+1):
            S0_qc.x(q)
            
        S0_qc.h(num_qubits)
        S0_qc.mcx(list(range(num_qubits)), num_qubits)
        S0_qc.h(num_qubits)
        
        for q in range(num_qubits+1):
            S0_qc.x(q)
            
        return S0_qc
    
    
    def Schi(self):
        """Oracle circuit.

        Returns
        -------
        Schi_qc : QuantumCircuit
            Oracle circuit.
        """
        
        num_qubits = self._lhs.num_qubits
        Schi_qc    = QuantumCircuit(num_qubits+1)
        
        Schi_qc.z(num_qubits)
        
        return Schi_qc
    
    
    def Grover(self):
        """Circuit for Grover operator.

        Returns
        -------
        Q_qc : QuantumCircuit
            Circuit for Grover operator.
        """
        
        S0_qc   = self.S0()
        Schi_qc = self.Schi()
        A_qc    = self.state_preparation()

        Q_qc    = Schi_qc.compose(A_qc.inverse())
        Q_qc    = Q_qc.compose(S0_qc)
        Q_qc    = Q_qc.compose(A_qc)
        
        return Q_qc
    
    
    def objective_qubits(self):
        """List of objective qubits for QAE.

        Returns
        -------
        oq : list[int]
            List of qubit indices.
        """
        
        num_qubits = self._lhs.num_qubits
        oq         = [num_qubits]
        
        return oq
    
    @staticmethod
    def is_good_state(string):
        """Checks of string corresponds to good state.

        Parameters
        ----------
        string : str
            Bitstring.

        Raises
        ------
        ValueError
            Raises error of string length unequal to 1.

        Returns
        -------
        result : bool
            True if good state.
        """
        
        length = len(string)
        result = False
        
        if length != 1:
            msg = "Invalid string length (" + str(length) + "), must be 1"
            raise ValueError(msg)
        
        ones = '1'
        ones = ones.rjust(length, '1')
        if string == ones:
            result = True
            
        return result


        
class InnerProductOverlap(InnerProduct):
    """Class estimating the squared inner product via the overlap."""
    
    def __init__(self, lhs, rhs):
        
        self._lhs = lhs
        self._rhs = rhs
        
        n1 = lhs.num_qubits
        n2 = rhs.num_qubits
        
        if n1 != n2:
            msg  = "Qubit size of lhs (n="
            msg += str(n1)
            msg += ") does not match qubit size of rhs (n="
            msg += str(n2) + ")"
            
            raise ValueError(msg)
            
            
    def dot_circuit(self, w_meas = True):
        """Prepare circuits for <lhs|rhs>.

        Parameters
        ----------
        w_meas : bool, optional
            Add measurement instructions. The default is True.

        Returns
        -------
        innerp : QuantumCircuit
            Circuit for <lhs|rhs>.
        """
        
        innerp = self._lhs.compose(self._rhs.inverse())
        
        if w_meas:
            innerp.measure_all()
        
        return innerp
    
    
    get_circuits = dot_circuit
    
    
    def dot_from_counts(self, counts, as_prob=False):
        """Estimate inner product from counts.

        Parameters
        ----------
        counts : Counts
            Experimental counts.
        as_prob : bool, optional
            The counts are in probability format (normalized). The default is False.

        Raises
        ------
        ValueError
            Raises error if outcome bitstrings have length unequal to 1.

        Returns
        -------
        expec : float
            Inner product estimate.
        """
        
        # check length
        num_clbits = len(next(iter(counts)))
        num_qubits = self._lhs.num_qubits
        
        if num_qubits != num_clbits:
            msg  = "Number of classical bits (n="
            msg += str(num_clbits)
            msg += ") does not match number of qubits (n="
            msg += str(num_qubits) + ")"
            
            raise ValueError(msg)
        
        # convert to probabilities
        probs = counts
        if not as_prob:
            probs = counts2probs(counts)
            
        # compute expectation
        measurement = '0'.zfill(num_qubits)
        p0          = 0.
        if measurement in probs.keys():
            p0 += probs[measurement]
            
        return p0
    
    @staticmethod
    def dot_from_amplitude(a):
        """Inner product from probability of |1> state.

        Parameters
        ----------
        p1 : float
            Probability of measuring the |1> state.

        Returns
        -------
        expec : float
            Inner product estimate.

        """
        
        expec = a
        
        return expec
    
    
    def state_preparation(self):
        """State preparation circuit for QAE.

        Returns
        -------
        QuantumCircuit
            State preparation circuit.
        """
        
        A = self.dot_circuit(w_meas=False)
        for q in range(A.num_qubits):
            A.x(q)
        
        return A
    
    
    def S0(self):
        """Circuit for adding phase to the all-0 state.

        Returns
        -------
        S0_qc : QuantumCircuit
            Circuit for the all-0 state.
        """
        
        num_qubits = self._lhs.num_qubits
        S0_qc      = QuantumCircuit(num_qubits)
        
        for q in range(num_qubits):
            S0_qc.x(q)
            
        S0_qc.h(num_qubits-1)
        S0_qc.mcx(list(range(num_qubits-1)), num_qubits-1)
        S0_qc.h(num_qubits-1)
        
        for q in range(num_qubits):
            S0_qc.x(q)
            
        return S0_qc
    
    
    def Schi(self):
        """Oracle circuit.

        Returns
        -------
        Schi_qc : QuantumCircuit
            Oracle circuit.
        """
        
        num_qubits = self._lhs.num_qubits
        S1_qc      = QuantumCircuit(num_qubits)
        
        S1_qc.h(num_qubits-1)
        S1_qc.mcx(list(range(num_qubits-1)), num_qubits-1)
        S1_qc.h(num_qubits-1)
        
        return S1_qc
    
    
    def Grover(self):
        """Circuit for Grover operator.

        Returns
        -------
        Q_qc : QuantumCircuit
            Circuit for Grover operator.
        """
        
        S0_qc   = self.S0()
        Schi_qc = self.Schi()
        A_qc    = self.state_preparation()
        
        Q_qc    = Schi_qc.compose(A_qc.inverse())
        Q_qc    = Q_qc.compose(S0_qc)
        Q_qc    = Q_qc.compose(A_qc)
        
        return Q_qc
    
    
    def objective_qubits(self):
        """List of objective qubits for QAE.

        Returns
        -------
        oq : list[int]
            List of qubit indices.
        """
        
        num_qubits = self._lhs.num_qubits
        oq         = list(range(num_qubits))
        
        return oq
    
    
    def is_good_state(self, string):
        """Checks of string corresponds to good state.

        Parameters
        ----------
        string : str
            Bitstring.

        Raises
        ------
        ValueError
            Raises error of string length unequal to 1.

        Returns
        -------
        result : bool
            True if good state.
        """
        
        num_qubits = self._lhs.num_qubits
        length     = len(string)
        result     = False
        
        if length != num_qubits:
            msg = "Invalid string length (" + str(length) + "), must be " + str(num_qubits)
            raise ValueError(msg)
        
        ones = '1'
        ones = ones.rjust(length, '1')
        if string == ones:
            result = True
            
        return result
        

        
class InnerProductStatevec(InnerProduct):
    """Class estimating the inner product numerically exact via state vector simulation."""
    
    def __init__(self, lhs, rhs):
        
        self._lhs     = lhs
        self._rhs     = rhs
        self._lhs_vec = None
        self._rhs_vec = None
        
        n1 = lhs.num_qubits
        n2 = rhs.num_qubits
        
        if n1 != n2:
            msg  = "Qubit size of lhs (n="
            msg += str(n1)
            msg += ") does not match qubit size of rhs (n="
            msg += str(n2) + ")"
            
            raise ValueError(msg)
            
            
    def vdot_exact(self):
        """Numerically exact inner product.

        Returns
        -------
        innerp : complex
            Inner product <lhs|vhs>.

        """
        
        self.simulate()
        innerp = np.vdot(self._lhs_vec, self._rhs_vec)

        return innerp
    
    
    def dot_circuit(self):
        """For this class no circuits required, returns None.
        
        Returns
        -------
        None
        """
        return None
    
    
    get_circuits = dot_circuit
    
    
    def dot_from_counts(self, counts=None, as_prob=False):
        """Same as real(vdot_exact) for this class."""
        return np.real(self.vdot_exact())
    
    
    def simulate(self):
        """Simulates state vectors.

        Returns
        -------
        None.
        """
        
        state_sim     = Aer.get_backend('statevector_simulator')
        self._lhs_vec = state_sim.run(self._lhs).result().get_statevector().data
        self._rhs_vec = state_sim.run(self._rhs).result().get_statevector().data
    
    
    @property
    def lhs_vec(self):
        """Numpy array for left-hand-side state/vector."""
        return self._lhs_vec
    
    
    @property
    def rhs_vec(self):
        """Numpy array for rhs-hand-side state/vector."""
        return self._rhs_vec



def get_innerp_circuit_driver(name):
    """Return inner product circuit driver if available.

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
    
    DRIVERS = {'HTest': InnerProductHTest,
               'Overlap': InnerProductOverlap,
               'Statevec': InnerProductStatevec}
    
    if name not in DRIVERS:
        message = "Inner product circuit method not implemented. Available drivers: "
        message += str(list(DRIVERS.keys()))
        raise ValueError(message)
        
    return DRIVERS[name]
        
        
def bell_state(f0, f1):
    """Prepares the circuit for the Bell-like superposition
    1/\sqrt(2)(|0>|f_0>+|1>|f_1>)

    Parameters
    ----------
    f0 : QuantumCircuit
        State attached to the 0-qubit.
    f1 : QuantumCircuit
        State attached to the 1-qubit.

    Returns
    -------
    f0_f1 : QuantumCircuit
        Circuit for preparing 1/\sqrt(2)(|0>|f_0>+|1>|f_1>).
    """
  
    
    assert f0.num_qubits == f1.num_qubits
    
    n = f0.num_qubits
    
    Cf0     = QuantumCircuit(n+1)
    Cf0gate = f0.to_gate().control(1)
    Cf0.append(Cf0gate, [n]+list(range(0, n)))
    
    Cf1     = QuantumCircuit(n+1)
    Cf1gate = f1.to_gate().control(1)
    Cf1.append(Cf1gate, [n]+list(range(0, n)))

    f0_f1 = QuantumCircuit(n+1, name='bell_state')
    f0_f1.h(n)
    f0_f1.compose(Cf0, list(range(0,n+1)), inplace=True)
    f0_f1.x(n)
    f0_f1.compose(Cf1, list(range(0,n+1)), inplace=True)

    return f0_f1



# %% OPERATOR
#
"""
Prepare Qiskit circuits for estimating operator expectation values.
"""

class PoissonVQAOperator(ABC):
    """Abstract class for VQA operator drivers.
    Basic functionality is to prepare circuits for measurement.
    
    Parameters
    ----------
    ansatz : QuantumCircuit
        Parametrized ansatz Quantum circuit.
    """
    
    
    def __init__(self, ansatz):

        self._ansatz = ansatz
    
    
    @abstractmethod
    def get_circuits(self):
        """Prepare (parametrized) circuits for measurement
            
        Returns
        -------
        object
            Circuit or collection of circuits.
        
        """
        
        
        pass
    
    
    @abstractmethod
    def energy_from_measurements(self, measurements):
        """Estimate <\psi | A | \psi> from measurements
            
        Parameters
        ----------
        measurements : list
            List of measurement probabilities.

        Returns
        -------
        float
            Energy estimate.
        """
        
        
        pass
    
    
    @property
    def ansatz(self):
        """Parametrized ansatz circuit (QuantumCircuit)."""
        return self._ansatz


    @ansatz.setter
    def ansatz(self, ansatz_):
        self._ansatz = ansatz_



class Poisson1dVQAOperator_exact(PoissonVQAOperator):
    """Class using the statevector simulator to compute numerically exact operator norm values."""
    
    def __init__(self, ansatz):

        self._ansatz   = ansatz
        self._operator = None
        
    
    def get_circuits(self):
        """This method does nothing, no circuits required for this class.
   
        Returns
        -------
        None.
        
        """
        
        return None
    
    
    def energy_from_measurements(self, measurements=None):
        """Compute <\psi | A | \psi> numerically exact, no measurements needed.
            
        Parameters
        ----------
        measurements : list
            List of measurement probabilities (input ignored).

        Returns
        -------
        float
            Energy estimate.
    
        """
        
        return self.energy()
    
    
    def energy(self):
        """Compute <\psi | A | \psi> numerically exact.
            
        Returns
        -------
        float
            Energy estimate.
    
        """
        
        A   = self._operator
        if A is None:
            A   = self._assemble_operator()
        psi = self._simulate_statevector()
        
        E_A = np.real(np.vdot(psi, A.dot(psi)))
        
        return E_A
    
    
    def _assemble_operator(self):
        N              = 2**self._ansatz.num_qubits
        diagonals      = [[-1]*(N-1), [2]*N, [-1]*(N-1)]
        A              = diags(diagonals, [-1, 0, 1])
        self._operator = A
        
        return A
    
    
    def _simulate_statevector(self):
        
        simulator   = StatevectorSimulator()
        statevector = simulator.run(self._ansatz).result().get_statevector().data
        
        return statevector
    
    
    @property
    def operator(self):
        """Stiffness matrix for the 1D Dirichlet Poisson operator."""
        return self._operator
        
            

class Poisson1dVQAOperator_Paulis(PoissonVQAOperator):
    """Class preparing circuits for estimating the operator norm
    using Pauli strings.
    Not scalable for large number of qubits.
    """
    
    
    def __init__(self, ansatz):

        self._ansatz = ansatz
        self._groups = None
        self._bases  = None
    
    
    def get_circuits(self):
        """Prepare (parametrized) circuits for measurement.
   
        Returns
        -------
        list[QuantumCircuit]
            Circuits required for estimating the energy.
            
        """
        
        # convert to Pauli sum
        PauliSum     = self._get_paulis()
        
        # group measurements
        grouper      = AbelianGrouper()
        groups       = grouper.convert(PauliSum)
        self._groups = groups
        
        # extract measurement basis
        bases        = self._get_bases()
        self._bases  = bases
        
        # add rotations
        circuits     = self._add_rotations()
        
        return circuits
    
    
    def energy_from_measurements(self, measurements):
        """Estimate <\psi | A | \psi> from measurements
            
        Parameters
        ----------
        measurements : list
            List of measurement probabilities.

        Returns
        -------
        float
            Energy estimate.
        """
        
        # check lengths
        len_meas   = len(measurements)
        len_groups = len(self._groups)
        if len_meas != len_groups:
            msg = f"Length of measurements ({len_meas}) incompatible with length of groups ({len_groups})"
            raise ValueError(msg)
        
        # postprocess results
        E_A = 0.
        for group, probs in zip(self._groups, measurements):
            E_A += estimate_pauli_group(group, probs)
        
        return np.real(E_A)
        
    
    def _get_paulis(self):
        
        num_qubits = self._ansatz.num_qubits
        N          = 2**num_qubits
        diagonals  = [[2]*N, [-1]*(N-1), [-1]*(N-1)]
        A          = diags(diagonals, [0, -1, 1])
        Op         = Operator(A.toarray())
        Sparse     = SparsePauliOp.from_operator(Op)
        Sum        = PauliSumOp.from_list(Sparse.to_list())
        
        return Sum

    
    def _get_bases(self):
        
        bases = []
        for group in self._groups:
            basis = ['I']*group.num_qubits
            for pauli_string in group.primitive.paulis:
                for i, pauli in enumerate(pauli_string):
                    p = str(pauli)
                    if p != 'I':
                        if basis[i] == 'I':
                            basis[i] = p
                        elif basis[i] != p:
                            raise ValueError('PauliSumOp contains non-commuting terms!')
                            
            bases.append(basis)
            
        return bases
    
    
    def _add_rotations(self):
        
        circuits = []
        for basis in self._bases:
            
            new_qc = self._ansatz.copy()
            for i, pauli in enumerate(basis):
                
                if pauli == 'X':
                    new_qc.h(i)
                if pauli == 'Y':
                    new_qc.s(i)
                    new_qc.h(i)
            
            new_qc.measure_all()
            circuits.append(new_qc)
            
        return circuits
        
    

class Poisson1dVQAOperator_Sato21(PoissonVQAOperator):
    """Class preparing circuits for estimating the operator norm as in [1].
    
    References
    ----------
    [1] Yuki Sato, Ruho Kondo, Satoshi Koide, Hideki Takamatsu and Nobuyuki Imoto,
        "Variational quantum algorithm based on the minimum potential energy
        for solving the Poisson equation", PRA 104, 052409 (2021).
        
    """
    
    
    def get_circuits(self):
        """Prepare (parametrized) circuits for measurement.
        
        Returns
        -------
        dict of QuantumCircuit
            Circuits required for estimating energy as in [1].
            
        """
        
        circuits = self._assemble_circuits()
        
        return circuits
    
    
    def energy_from_measurements(self, measurements):
        """Estimate <\psi | A | \psi> from measurements
            
        Parameters
        ----------
        measurements : list
            List of measurement probabilities.

        Returns
        -------
        float
            Energy estimate.
        """
        
        # postprocess results
        even   = estimate_X(measurements[0])
        odd    = estimate_I0X(measurements[1])
        
        # estimated dirichlet energy
        E_A = 2. - even - odd
        
        return E_A
        
    
    def _assemble_circuits(self):
        
        # circuit for <psi | I \otimes X | psi>
        x_psi = self._ansatz.copy()
        cr    = ClassicalRegister(1)
        x_psi.h(0)
        x_psi.add_register(cr)
        x_psi.measure(0, cr)
        
        # circuit for <psi | P* (I^{n-1} \otimes X) P | psi>
        P           = shift_operator(self._ansatz.num_qubits)
        P.h(0)
        x_shift_psi = self._ansatz.compose(P)
        x_shift_psi.measure_all()
    
        return [x_psi, x_shift_psi]

    
    
class Poisson1dVQAOperator_SatoLiu21(PoissonVQAOperator):
    """Class preparing circuits for estimating the operator norm as in [1]
    and the decomposition for the Poisson operator as in [2].
    
    References
    ----------
    [1] Yuki Sato, Ruho Kondo, Satoshi Koide, Hideki Takamatsu
        and Nobuyuki Imoto,
        "Variational quantum algorithm based on the minimum potential energy
        for solving the Poisson equation", PRA 104, 052409 (2021).
        
    [2] Hai-Ling Liu, Yu-Sen Wu, Lin-Chun Wan, Shi-Jie Pan, Su-Juan Qin,
        Fei Gao and Qiao-Yan Wen,
        "Variational quantum algorithm for the Poisson equation",
        PRA 104, 022418 (2021).
    """
    
    
    def get_circuits(self):
        """Prepare (parametrized) circuits for measurement.
        
        Returns
        -------
        circuits : dict of QuantumCircuit
            Circuits required for estimating energy as in [1, 2].
            
        """
        
        circuits = self._assemble_circuits()
            
        return circuits
    
    
    def energy_from_measurements(self, measurements):
        """Estimate <\psi | A | \psi> from measurements
            
        Parameters
        ----------
        measurements : list
            List of measurement probabilities.

        Returns
        -------
        float
            Energy estimate.
        """
        
        # postprocess results
        length   = self._ansatz.num_qubits - 1
        even     = estimate_X(measurements[0])
        odd      = estimate_sigmas(measurements[1:1+length])
        
        # estimated dirichlet energy
        E_A = 2. - even - odd
        
        return E_A
    
    
    def _assemble_circuits(self):
        
        # circuit for <psi | I \otimes X | psi>
        x_psi = self._ansatz.copy()
        cr    = ClassicalRegister(1)
        x_psi.h(0)
        x_psi.add_register(cr)
        x_psi.measure(0, cr)
        
        # one-to-all cnots
        num_qubits = self._ansatz.num_qubits
        cnots      = []
        for m in range(1, num_qubits):
            controls = [0]*(m+1)
            targets  = list(range(1, m+1))
            targets.append(num_qubits)
            cnots_   = QuantumCircuit(num_qubits+1, name='one_to_all_cnots')
            cnots_.cx(controls, targets)
            cnots_.h(0)
            cnots.append(cnots_)
        
        # circuit for <psi | P* (I^{n-1} \otimes sigm+-) P | psi>
        sigmas_real = []
        
        for i in range(len(cnots)):
            sigmas_real_ = QuantumCircuit(num_qubits+1, name='sigmas_real')
            sigmas_real_.h(num_qubits)
        
            sigmas_real_.compose(self._ansatz, range(0, num_qubits),
                                 inplace=True)
            sigmas_real_.compose(cnots[i], inplace=True)
            sigmas_real_.measure_all()
            sigmas_real.append(sigmas_real_)

        return [x_psi] + sigmas_real



class Poisson1dVQAOperator_SatoLiu21_efficient(PoissonVQAOperator):
    """Class preparing circuits for estimating the operator norm as in [1],
    the decomposition for the Poisson operator as in [2] and a modified
    implementation requiring measuring only 2 instead of 2*(n-1) circuits.
    
    References
    ----------
    [1] Yuki Sato, Ruho Kondo, Satoshi Koide, Hideki Takamatsu
        and Nobuyuki Imoto,
        "Variational quantum algorithm based on the minimum potential energy
        for solving the Poisson equation", PRA 104, 052409 (2021).
        
    [2] Hai-Ling Liu, Yu-Sen Wu, Lin-Chun Wan, Shi-Jie Pan, Su-Juan Qin,
        Fei Gao and Qiao-Yan Wen,
        "Variational quantum algorithm for the Poisson equation",
        PRA 104, 022418 (2021).
    """
    
    
    def get_circuits(self):
        """Prepare (parametrized) circuits for measurement.
        
        Returns
        -------
        circuits : dict of QuantumCircuit
            Circuits required for estimating energy as in [1, 2].
            
        """
        
        circuits = self._assemble_circuits()
            
        return circuits
    
    
    def energy_from_measurements(self, measurements):
        """Estimate <\psi | A | \psi> from measurements
            
        Parameters
        ----------
        measurements : list
            List of measurement probabilities.

        Returns
        -------
        float
            Energy estimate.
            
        """
        
        # postprocess results
        even = estimate_X(measurements[0])
        odd  = estimate_sigmas_efficient(measurements[1])
        
        # estimated dirichlet energy
        E_A = 2. - even - odd
        
        return E_A
    
    
    def _assemble_circuits(self):
        
        # circuit for <psi | I \otimes X | psi>
        x_psi = self._ansatz.copy()
        cr    = ClassicalRegister(1)
        x_psi.h(0)
        x_psi.add_register(cr)
        x_psi.measure(0, cr)
        
        # one-to-all cnots
        num_qubits = self._ansatz.num_qubits
        controls   = [0]*num_qubits
        targets    = list(range(1, num_qubits+1))
        cnots      = QuantumCircuit(num_qubits+1, name='one_to_all_cnots')
        cnots.cx(controls, targets)
            
        # one-to-all Hadamards
        chs = QuantumCircuit(num_qubits+1, name='one_to_all_hs')
        for i in range(num_qubits-2):
            for j in range(i+1):
                tmp = QuantumCircuit(num_qubits-1-i)
                
                for l in range(num_qubits-3-i):
                    tmp.x(l)
                    
                ch = MCMT('h', num_qubits-2-i, 1)
                tmp.compose(ch, inplace=True)
                
                for l in range(num_qubits-3-i):
                    tmp.x(l)
                
                controls = list(range(1, num_qubits-1-i))
                nodes    = controls + [num_qubits-1-j]
                chs.compose(tmp, nodes, inplace=True)
        
        # circuit for <psi | P* (I^{n-1} \otimes sigm+-) P | psi>
        sigmas_real = QuantumCircuit(num_qubits+1, name='sigmas_real')
        sigmas_real.h(num_qubits)
        
        sigmas_real.compose(self._ansatz, range(0, num_qubits), inplace=True)
        sigmas_real.compose(cnots, inplace=True)
        sigmas_real.h(0)
        sigmas_real.compose(chs, inplace=True)
        sigmas_real.measure_all()

        return [x_psi, sigmas_real]



def get_poisson_operator_driver(name):
    """Return Poisson circuit driver if available.

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
    
    DRIVERS = {'exact'              : Poisson1dVQAOperator_exact,
               'Paulis'             : Poisson1dVQAOperator_Paulis,
               'Sato21'             : Poisson1dVQAOperator_Sato21,
               'SatoLiu21'          : Poisson1dVQAOperator_SatoLiu21,
               'SatoLiu21_efficient': Poisson1dVQAOperator_SatoLiu21_efficient}
    
    if name not in DRIVERS:
        message = "Poisson operator method not implemented. Available drivers: "
        message += str(list(DRIVERS.keys()))
        raise ValueError(message)
        
    return DRIVERS[name]


def shift_operator(num_qubits):
    """Creates the circuit for the shift operator from [1].
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    
    Returns
    -------
    P : QuantumCircuit
        Circuit corresponding to the shift gate mapping |k> to |(k+1) mod 2^n>.
        
     References
    ----------
    [1] Yuki Sato, Ruho Kondo, Satoshi Koide, Hideki Takamatsu and Nobuyuki Imoto,
        "Variational quantum algorithm based on the minimum potential energy
        for solving the Poisson equation", PRA 104, 052409 (2021).

    """

    assert num_qubits>0
    
    P = QuantumCircuit(num_qubits, name='shift_operator')
    
    for i in range(num_qubits-1, 0, -1):
        P.mcx(list(range(0, i)), i)
        
    P.x(0)

    return P


def estimate_X(probs):
    
    # check correct string length
    num_clbits = len(next(iter(probs)))
    
    if num_clbits != 1:
        raise ValueError("Length of binary outcome for this experiment \
                         must be 1")
        
    # probabilities
    p0 = 0.
    
    if '0' in probs.keys():
        p0 += probs['0']
        
    p1 = 1. - p0
    
    # expectation
    return p0 - p1


def estimate_I0X(probs):
    
    num_bits = len(next(iter(probs)))
    
    # outcomes for I_0 \otimes X
    zero = '0'.zfill(num_bits)
    one  = '1'.zfill(num_bits)

    # probabilities
    pI_0  = 0.
    pI_1  = 0.
    pI0_0 = 0.
    pI0_1 = 0.

    for key, value in probs.items():
        if key == zero:
            pI0_0 += value
        if key == one:
            pI0_1 += value
        if key[num_bits-1] == '0':
            pI_0 += value
        if key[num_bits-1] == '1':
            pI_1 += value

    # expectation
    E_IX  = pI_0 - pI_1
    E_I0X = pI0_0 - pI0_1

    return -E_I0X + E_IX


def estimate_sigmas(probs):
    
    num_meas   = len(probs)
    num_qubits = len(next(iter(probs[0]))) - 1
    
    if num_meas != num_qubits - 1:
        raise ValueError("Number of sigma circuits incompatible \
                         with number of qubits.")
    
    total = 0.
    for i in range(num_meas):
        idx = num_qubits - 1 - i
        for key, value in probs[i].items():
            
            root   = key[idx:num_qubits]
            length = num_qubits - idx
        
            if root == '1'.ljust(length, '0'):
                
                one_count = 0
                if key[num_qubits] == '1':
                    one_count += 1
                
                total += (-1)**one_count*value
                
    return total


def estimate_sigmas_efficient(probs):
    
    num_qubits = len(next(iter(probs))) - 1
    num_meas   = num_qubits - 1
    total      = 0.
    
    for key, value in probs.items():
        for i in range(num_meas):
            idx    = num_qubits - 1 - i
            root   = key[idx:num_qubits]
            length = num_qubits - idx
        
            if root == '1'.ljust(length, '0'):
                
                one_count = 0
                for k in range(1, idx):
                    if key[k] == '1':
                        one_count += 1
                if key[num_qubits] == '1':
                    one_count += 1
                
                total += (-1)**one_count*value
                
    return total


def estimate_pauli_group(group, probs):
    """Estimates expectation for group of commuting Paulis.
    

    Parameters
    ----------
    group : PauliSumOp
        Group of commuting Paulis.
    probs : dict.
        Measurement results formatted as a dictionary of probabilities.

    Returns
    -------
    complex.
    
    """
    
    value = 0
    for (pauli, coeff) in zip(group.primitive.paulis, group.primitive.coeffs):
        val = 0
        p = str(pauli)
        for b, prob in probs.items():
            val += prob * np.prod([(-1)**(b[i] == '1' and p[i] != 'I') for i in range(len(b))])

        value += coeff * val

    return value



# %% COST
#
"""
Prepare Qiskit circuits for estimating Poisson cost functions.
"""


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
    
    DRIVERS = {'Sato21_Innerp'       : Poisson1dVQACost_Sato21_Innerp,
               'Sato21_Overlap'      : Poisson1dVQACost_Sato21_Overlap,
               'Sato21_Nonnorm'      : Poisson1dVQACost_Sato21_Nonnorm,
               'BravoPrieto20_Global': PoissonVQACost_BravoPrieto20_Global,
               'BravoPrieto20_Local' : PoissonVQACost_BravoPrieto20_Local}
    
    if name not in DRIVERS:
        message = "Poisson cost method not implemented. Available drivers: "
        message += str(list(DRIVERS.keys()))
        raise ValueError(message)
        
    return DRIVERS[name]



# %% ESTIMATOR
#
"""
Wrapper classes to combine backend calls and post processing into one call (suitable for, e.g., runtime).
"""


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
    
    DRIVERS = {'Measurement'       : MeasCostEstimator,
               'MeasurementNonnorm': MeasCostNonnormEstimator,
               'Exact'             : ExactCostEstimator}
    
    if name not in DRIVERS:
        message = "Cost estimator method not implemented. Available drivers: "
        message += str(list(DRIVERS.keys()))
        raise ValueError(message)
        
    return DRIVERS[name]



# %% SOLVER
#

def vqa_poisson1d(backend,
                  user_messenger,
                  ansatz,
                  rhs,
                  circuits,
                  sampler_options,
                  operator,
                  innerp,
                  cost,
                  exact,
                  w_grad,
                  optimizer,
                  optimizer_options,
                  init_params):
    """Qiskit runtime program for a variational quantum algorithm approximating
    the solution to the 1D Poisson problem.

    Parameters
    ----------
    backend : object
        Backend for running the VQA.
    user_messenger : UserMessenger
        Messenger for streaming intermediate results.
    ansatz : QuantumCircuit
        Parametrized quantum circuit for the ansatz (unbound parameters).
    rhs : QuantumCircuit
        Quantum circuit preparing the right hand side.
    circuits : list[QuantumCircuit]
        List of (transpiled) circuits required for estimating the cost/energy
        function and possibly gradients.
    sampler_options : dict
        Options to pass to the sampler.
    operator : str
        Method for estimating the operator part.
    innerp : str
        Method for estimating the inner product part.
    cost : str
        Type of cost function.
    exact : bool
        Flag if all estimations are performed numerically exactly (StatevectorSimulator).
    w_grad : bool
        Flag if optimizer uses gradients.
    optimizer : str
        Classical optimization method (implemented in Qiskit).
    optimizer_options : dict
        Options to pass to the optimizer.
    init_params : numpy.array
        Initial parameter vector.

    Returns
    -------
    opt_result : OptimizerResult
        Result of the optimization.
    energy_log : list[dict]
        History of all cost function evaluations.
    grad_log : list[dict]
        History of all gradient evaluations.
    hist_params : list[numpy.array]
        List of all (accepted) parameter iterates.
    """
    
    # MEM
    use_mem     = sampler_options.pop('use_mem')
    if use_mem:
        user_messenger.publish('Calibrating MEM...')
        mit     = mthree.M3Mitigation(backend)
        mapping = mthree.utils.final_measurement_mapping(circuits)
        
        start_mem  = time.time()
        mit.cals_from_system(mapping)
        mem_time   = time.time() - start_mem
        
        user_messenger.publish({'MEM cal. time (s)': mem_time})
        sampler_options['mitigator'] = mit
    
    # set up cost driver        
    sampler       = VanillaSampler(backend, **sampler_options)
    op_driver     = get_poisson_operator_driver(operator)(ansatz)
    innerp_driver = get_innerp_circuit_driver(innerp)(ansatz, rhs)
    cost_driver   = get_poisson_cost_driver(cost)(op_driver, innerp_driver)
    
    # set up cost and gradients
    estimator = None 
    costf     = None
    gradf     = None
    
    if exact:
        estimator = get_cost_estimator_driver('Exact')(cost_driver)
    elif cost=='Sato21_Nonnorm':
        estimator = get_cost_estimator_driver('MeasurementNonnorm')(circuits,
                                                                    cost_driver,
                                                                    sampler)
    else:
        estimator = get_cost_estimator_driver('Measurement')(circuits,
                                                             cost_driver,
                                                             sampler)
        
    costf = estimator.cost
    
    if w_grad:
        gradf = estimator.gradient
    
    
    # call optimizer
    hist_params   = [init_params.copy()]
    iteration     = 0
    opt_driver    = None
    start_time    = 0.
    last_it       = 0.
    cum_time      = 0.
    it_time       = 0.

    if optimizer in ('SPSA'):
        
        def callback(num_evals, params, fval, lrate, accepted):
            nonlocal iteration
            nonlocal cum_time
            nonlocal it_time
            nonlocal last_it
            
            now        = time.time()
            cum_time   = now - start_time
            it_time    = now - last_it
            last_it    = now
            
            user_messenger.publish({'iteration'    : iteration,
                                    'it time (s)'  : it_time,
                                    'cum time (s)' : cum_time})
            hist_params.append(params)
            
            iteration += 1
            return None
            
        opt_driver = getattr(qiskit_opt, optimizer)(**optimizer_options,
                                                    callback=callback)
        
    elif optimizer in ('QNSPSA'):
        
        fidelity = QNSPSA.get_fidelity(ansatz)
        def callback(num_evals, params, fval, lrate, accepted):
            nonlocal iteration
            nonlocal cum_time
            nonlocal it_time
            nonlocal last_it
            
            now        = time.time()
            cum_time   = now - start_time
            it_time    = now - last_it
            last_it    = now
            
            user_messenger.publish({'iteration'    : iteration,
                                    'it time (s)'  : it_time,
                                    'cum time (s)' : cum_time})
            hist_params.append(params)
            
            iteration += 1
            return None
        
        opt_driver = getattr(qiskit_opt, optimizer)(fidelity,
                                                    **optimizer_options,
                                                    callback=callback)
        
    elif optimizer in ('GradientDescent'):
          
        def callback(num_evals, params, fval, lrate):
              nonlocal iteration
              nonlocal cum_time
              nonlocal it_time
              nonlocal last_it
              
              now        = time.time()
              cum_time   = now - start_time
              it_time    = now - last_it
              last_it    = now
              
              user_messenger.publish({'iteration'    : iteration,
                                      'it time (s)'  : it_time,
                                      'cum time (s)' : cum_time})
              hist_params.append(params.copy())
              
              iteration += 1
              return None
              
        opt_driver = getattr(qiskit_opt, optimizer)(**optimizer_options,
                                                      callback=callback)

    
    elif optimizer in ('ADAM', 'AQGD', 'SNOBFIT', 'CRS', 'GSLS', 'COBYLA',
                       'UMDA', 'BOBYQA', 'IMFIL', 'SNOBFIT', 'CRS',
                       'DIRECT_L', 'DIRECT_L_RAND', 'ESCH', 'ISRES'):
        
        opt_driver = getattr(qiskit_opt, optimizer)(**optimizer_options)

        
    elif optimizer in ('L_BFGS_B', 'P_BFGS', 'NFT', 'CG', 'NELDER_MEAD', 'POWELL', 'SLSQP', 'TNC'):
        
        def callback(params):
            nonlocal iteration
            nonlocal cum_time
            nonlocal it_time
            nonlocal last_it
            
            now        = time.time()
            cum_time   = now - start_time
            it_time    = now - last_it
            last_it    = now
            
            user_messenger.publish({'iteration'    : iteration,
                                    'it time (s)'  : it_time,
                                    'cum time (s)' : cum_time})
            hist_params.append(params.copy())
            
            iteration += 1
            return None
        
        opt_driver = getattr(qiskit_opt, optimizer)(**optimizer_options,
                                                    callback=callback)
    else:
        
        msg = "Optimizer not recognized"
        raise ValueError(msg)
        
    bounds = None
    if optimizer in ('BOBYQA', 'IMFIL', 'SNOBFIT', 'CRS', 'DIRECT_L',
                     'DIRECT_L_RAND', 'ESCH', 'ISRES'):
        num_parameters = init_params.size
        bounds         = np.empty((num_parameters, 2))
        for k in range(num_parameters):
            bounds[k, 0] = -4.*pi
            bounds[k, 1] = 4.*pi
        
    start_time = time.time()
    last_it    = start_time
    opt_result = opt_driver.minimize(fun=costf,
                                     x0=init_params.copy(),
                                     jac=gradf,
                                     bounds=bounds)
    result     = {'fun': opt_result.fun,
                  'jac': opt_result.jac,
                  'nfev': opt_result.nfev,
                  'nit': opt_result.nit,
                  'njev': opt_result.njev,
                  'x': opt_result.x}

    return result, estimator.energy_log, estimator.grad_log, hist_params



# main function (entry point)
def main(backend, user_messenger, **kwargs):

    result = vqa_poisson1d(backend, user_messenger, **kwargs) 
    return result
