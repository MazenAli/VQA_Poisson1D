"""
Prepare Qiskit circuits for estimating inner products.
"""

from abc import ABC, abstractmethod
import numpy as np

from qiskit import QuantumCircuit, Aer

from vqa_poisson_tools.utils.utils import counts2probs


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
    
        
      
DRIVERS = {'HTest': InnerProductHTest,
           'Overlap': InnerProductOverlap,
           'Statevec': InnerProductStatevec}


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