"""
Circuits for the RHS
"""


from qiskit import QuantumCircuit


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


RHS = {'id': idn,
       'hn': hn,
       'hnx': hnx}


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
    
    if name not in RHS:
        message = "Rhs not implemented. Available rhs: "
        message += str(list(RHS.keys()))
        raise ValueError(message)
        
    return RHS[name]