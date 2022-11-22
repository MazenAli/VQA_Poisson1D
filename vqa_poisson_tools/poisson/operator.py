"""
Prepare Qiskit circuits for estimating operator expectation values.
"""


from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import diags

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.providers.aer import StatevectorSimulator
from qiskit.circuit.library import MCMT
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.opflow import PauliSumOp, AbelianGrouper



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


    
DRIVERS = {'exact'              : Poisson1dVQAOperator_exact,
           'Paulis'             : Poisson1dVQAOperator_Paulis,
           'Sato21'             : Poisson1dVQAOperator_Sato21,
           'SatoLiu21'          : Poisson1dVQAOperator_SatoLiu21,
           'SatoLiu21_efficient': Poisson1dVQAOperator_SatoLiu21_efficient}


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