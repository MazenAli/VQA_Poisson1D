"""
Custom two-local ansatz circuits
"""


from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import TwoLocal

ROTATION_GATES      = ['ry', 'rx', 'rz']
ENTANGLERS_NONPARAM = ['csx', 'cy', 'cx', 'cz']
MIXER_GATES         = ROTATION_GATES
DRIVER_GATES        = ['ryy', 'rxx', 'rzz', 'rzx']


def ansatz_linear(num_qubits, num_layers,
                  rotation_gates = ['ry'],
                  entangler      = 'cx',
                  prefix         = 'θ',
                  **qc_options):
    """Function assembling a parametrized hardware efficient
    ansatz circuit with linear entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    
    num_layers : int
        Number of layers (repetitions).
    
    rotation_gates: list of str, optional
        List of strings specifying the single-parameter rotation gates
        to apply in each layer. Default is 'ry'.
    
    entangler : str, optional
        Name of 2-qubit entangling gates to apply in each layer.
        Default is 'cx'.
        
    prefix : str, optional
        String for name of abstract parameters. Default is 'θ'.
        
    qc_options : dict, optional
        Keyword arguments for the quantum circuit constructor. Default is empty.

    Raises
    ------
    ValueError
        Raises error if illegal values for rotation gates or entangler.

    Returns
    -------
    Utheta : QuantumCircuit
        Parametrized ansatz circuit.

    """
    
    
    if not set(rotation_gates) <= set(ROTATION_GATES):
        message  = "Illegal value for rotation gates. Allowed values: "
        message += str(ROTATION_GATES)
        raise ValueError(message)
        
    if entangler not in ENTANGLERS_NONPARAM:
        message  = "Illegal value for entangler. Allowed values: "
        message += str(ENTANGLERS_NONPARAM)
        raise ValueError(message)

    Utheta = TwoLocal(num_qubits=num_qubits,
                      rotation_blocks=rotation_gates,
                      entanglement_blocks=entangler,
                      entanglement='linear',
                      reps=num_layers,
                      parameter_prefix=prefix,
                      **qc_options)
    
    return Utheta.decompose()


def ansatz_linear_periodic(num_qubits, num_layers,
                           rotation_gates = ['ry'],
                           entangler      = 'cx',
                           prefix         = 'θ',
                           **qc_options):
    """Function assembling a parametrized hardware efficient
    ansatz circuit with linear periodic entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    
    num_layers : int
        Number of layers (repetitions).
    
    rotation_gates: list of str, optional
        List of strings specifying the single-parameter rotation gates
        to apply in each layer. Default is 'ry'.
    
    entangler : str, optional
        Name of 2-qubit entangling gates to apply in each layer.
        Default is 'cx'.
        
    prefix : str, optional
        String for name of abstract parameters. Default is 'θ'.
        
    qc_options : dict, optional
        Keyword arguments for the quantum circuit constructor. Default is empty.

    Raises
    ------
    ValueError
        Raises error if illegal values for rotation gates or entangler.

    Returns
    -------
    Utheta : QuantumCircuit
        Parametrized ansatz circuit.
    """
    
    
    if not set(rotation_gates) <= set(ROTATION_GATES):
        message  = "Illegal value for rotation gates. Allowed values: "
        message += str(ROTATION_GATES)
        raise ValueError(message)
        
    if entangler not in ENTANGLERS_NONPARAM:
        message  = "Illegal value for entangler. Allowed values: "
        message += str(ENTANGLERS_NONPARAM)
        raise ValueError(message)

    Utheta = TwoLocal(num_qubits=num_qubits,
                      rotation_blocks=rotation_gates,
                      entanglement_blocks=entangler,
                      entanglement='circular',
                      reps=num_layers,
                      parameter_prefix=prefix,
                      **qc_options)
    
    return Utheta.decompose()


def ansatz_linear_alternating(num_qubits, num_layers,
                              rotation_gates = ['ry'],
                              entangler      = 'cx',
                              prefix         = 'θ',
                              **qc_options):
    """Function assembling a parametrized hardware efficient
    ansatz circuit with linear alternating entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    
    num_layers : int
        Number of layers (repetitions).
    
    rotation_gates: list of str, optional
        List of strings specifying the single-parameter rotation gates
        to apply in each layer. Default is 'ry'.
    
    entangler : str, optional
        Name of 2-qubit entangling gates to apply in each layer.
        Default is 'cx'.
        
    prefix : str, optional
        String for name of abstract parameters. Default is 'θ'.
        
    qc_options : dict, optional
        Keyword arguments for the quantum circuit constructor. Default is empty.

    Raises
    ------
    ValueError
        Raises error if illegal values for rotation gates or entangler.

    Returns
    -------
    Utheta : QuantumCircuit
        Parametrized ansatz circuit.

    """
    
    
    if not set(rotation_gates) <= set(ROTATION_GATES):
        message  = "Illegal value for rotation gates. Allowed values: "
        message += str(ROTATION_GATES)
        raise ValueError(message)
        
    if entangler not in ENTANGLERS_NONPARAM:
        message  = "Illegal value for entangler. Allowed values: "
        message += str(ENTANGLERS_NONPARAM)
        raise ValueError(message)
    
    num_parameters  = 2*num_layers*(num_qubits-1)+num_qubits
    num_parameters *= len(rotation_gates)
    
    theta   = ParameterVector(prefix, num_parameters) 
    
    # initial layer
    Utheta = QuantumCircuit(num_qubits, **qc_options)
    index  = 0
    for j in range(num_qubits):
        for rotation in rotation_gates:
            getattr(Utheta, rotation)(theta[index], j)
            index += 1
        
    # internal layers
    for l in range(num_layers):
        
        # 1st layer
        for j in range(1, num_qubits, 2):
            getattr(Utheta, entangler)(j-1, j)
            
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j-1)
                index += 1
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j)
                index += 1
        
        # 2nd (alternatring) layer
        for j in range(2, num_qubits, 2):
            getattr(Utheta, entangler)(j-1, j)
            
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j-1)
                index += 1
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j)
                index += 1
    
    return Utheta


def ansatz_linear_alternating_periodic(num_qubits, num_layers,
                                       rotation_gates = ['ry'],
                                       entangler      = 'cx',
                                       prefix         = 'θ',
                                       **qc_options):
    """Function assembling a parametrized hardware efficient
    ansatz circuit with linear alternating entanglement and periodic
    boundary conditions (first and last qubit entangled).

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    
    num_layers : int
        Number of layers (repetitions).
    
    rotation_gates: list of str, optional
        List of strings specifying the single-parameter rotation gates
        to apply in each layer. Default is 'ry'.
    
    entangler : str, optional
        Name of 2-qubit entangling gates to apply in each layer.
        Default is 'cx'.
        
    prefix : str, optional
        String for name of abstract parameters. Default is 'θ'.
        
    qc_options : dict, optional
        Keyword arguments for quantum circuit constructor. Default is empty.

    Raises
    ------
    ValueError
        Raises error if illegal values for rotation gates or entangler.

    Returns
    -------
    Utheta : QuantumCircuit
        Parametrized ansatz circuit.

    """
    
    
    if not set(rotation_gates) <= set(ROTATION_GATES):
        message  = "Illegal value for rotation gates. Allowed values: "
        message += str(ROTATION_GATES)
        raise ValueError(message)
        
    if entangler not in ENTANGLERS_NONPARAM:
        message  = "Illegal value for entangler. Allowed values: "
        message += str(ENTANGLERS_NONPARAM)
        raise ValueError(message)
    
    num_parameters  = 2*num_layers*(num_qubits+1)+num_qubits
    num_parameters *= len(rotation_gates)
    
    theta   = ParameterVector(prefix, num_parameters) 
    
    # initial layer
    Utheta = QuantumCircuit(num_qubits, **qc_options)
    index  = 0
    for j in range(num_qubits):
        for rotation in rotation_gates:
            getattr(Utheta, rotation)(theta[index], j)
            index += 1
        
    # internal layers
    for l in range(num_layers):
        
        # 1st layer
        for j in range(1, num_qubits, 2):
            getattr(Utheta, entangler)(j-1, j)
            
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j-1)
                index += 1
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j)
                index += 1
        
        # 2nd (alternatring) layer
        for j in range(2, num_qubits, 2):
            getattr(Utheta, entangler)(j-1, j)
            
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j-1)
                index += 1
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j)
                index += 1
                
        # 3rd (periodic) layer
        if num_qubits > 2:
            getattr(Utheta, entangler)(0, num_qubits-1)
                
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], 0)
                index += 1
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], num_qubits-1)
                index += 1
    
    return Utheta


def ansatz_linear_alternating_periodic_bidirectional(num_qubits, num_layers,
                                                     rotation_gates = ['ry'],
                                                     entangler      = 'cx',
                                                     prefix         = 'θ',
                                                     **qc_options):
    """Function assembling a parametrized hardware efficient
    ansatz circuit with linear alternating entanglement, periodic
    boundary conditions (first and last qubit entangled) and
    entanglers repeated in both directions.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    
    num_layers : int
        Number of layers (repetitions).
    
    rotation_gates: list of str, optional
        List of strings specifying the single-parameter rotation gates
        to apply in each layer. Default is 'ry'.
    
    entangler : str, optional
        Name of 2-qubit entangling gates to apply in each layer.
        Default is 'cx'.
        
    prefix : str, optional
        String for name of abstract parameters. Default is 'θ'.
        
    qc_options : dict, optional
        Keyword arguments for quantum circuit constructor. Default is empty.

    Raises
    ------
    ValueError
        Raises error if illegal values for rotation gates or entangler.

    Returns
    -------
    Utheta : QuantumCircuit
        Parametrized ansatz circuit.

    """
    
    
    if not set(rotation_gates) <= set(ROTATION_GATES):
        message  = "Illegal value for rotation gates. Allowed values: "
        message += str(ROTATION_GATES)
        raise ValueError(message)
        
    if entangler not in ENTANGLERS_NONPARAM:
        message  = "Illegal value for entangler. Allowed values: "
        message += str(ENTANGLERS_NONPARAM)
        raise ValueError(message)
    
    num_parameters  = 2*num_layers*(num_qubits+1)+num_qubits
    num_parameters *= len(rotation_gates)*2
    
    theta   = ParameterVector(prefix, num_parameters) 
    
    # initial layer
    Utheta = QuantumCircuit(num_qubits, **qc_options)
    index  = 0
    for j in range(num_qubits):
        for rotation in rotation_gates:
            getattr(Utheta, rotation)(theta[index], j)
            index += 1
        
    # internal layers
    for l in range(num_layers):
        
        # 1st layer
        for j in range(1, num_qubits, 2):
            getattr(Utheta, entangler)(j-1, j)
            
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j-1)
                index += 1
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j)
                index += 1
        
        # 2nd (alternatring) layer
        for j in range(2, num_qubits, 2):
            getattr(Utheta, entangler)(j-1, j)
            
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j-1)
                index += 1
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j)
                index += 1
                
        # 3rd (periodic) layer
        if num_qubits > 2:
            getattr(Utheta, entangler)(0, num_qubits-1)
                
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], 0)
                index += 1
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], num_qubits-1)
                index += 1
                
    # repeat in reversed direction
    # internal layers
    for l in range(num_layers):
        
        # 1st layer
        for j in range(1, num_qubits, 2):
            getattr(Utheta, entangler)(j, j-1)
            
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j-1)
                index += 1
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j)
                index += 1
        
        # 2nd (alternatring) layer
        for j in range(2, num_qubits, 2):
            getattr(Utheta, entangler)(j, j-1)
            
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j-1)
                index += 1
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], j)
                index += 1
                
        # 3rd (periodic) layer
        if num_qubits > 2:
            getattr(Utheta, entangler)(num_qubits-1, 0)
                
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], 0)
                index += 1
            for rotation in rotation_gates:
                getattr(Utheta, rotation)(theta[index], num_qubits-1)
                index += 1
    
    return Utheta


def ansatz_qaoa(num_qubits, num_layers, initial_state,
                mixer          = 'rx',
                driver         = 'rzz',
                prefix         = 'θ',
                **qc_options):
    """Function assembling a parametrized QAOA-inspired ansatz circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    
    num_layers : int
        Number of layers (repetitions).
        
    initial_state: QuantumCircuit
        Circuit prepended at the beginning of the ansatz
    
    mixer: str, optional
        Strings specifying the single-parameter mixer rotation gate.
        Default is 'rx'.
    
    driver : str, optional
        Name of 2-qubit single-parameter entangling rotation gate.
        Default is 'rzz'.
        
    prefix : str, optional
        String for name of abstract parameters. Default is 'θ'.
        
    qc_options : dict, optional
        Keyword arguments for the quantum circuit constructor. Default is empty.

    Raises
    ------
    ValueError
        Raises error if illegal values for mixer or driver gates.

    Returns
    -------
    Utheta : QuantumCircuit
        Parametrized ansatz circuit.

    """
    
    
    if mixer not in set(MIXER_GATES):
        message  = "Illegal value for mixer. Allowed values: "
        message += str(MIXER_GATES)
        raise ValueError(message)
        
    if driver not in DRIVER_GATES:
        message  = "Illegal value for driver. Allowed values: "
        message += str(DRIVER_GATES)
        raise ValueError(message)
    
    num_parameters  = 2*num_layers
    theta           = ParameterVector(prefix, num_parameters) 
        
    # internal layers
    Utheta = QuantumCircuit(num_qubits, **qc_options)
    for l in range(num_layers):
        
        # driver layer
        for j in range(0, num_qubits-1):
            getattr(Utheta, driver)(theta[2*l], j, j+1)
        
        # mixer layer
        for j in range(num_qubits):
            getattr(Utheta, mixer)(theta[2*l+1], j)
    
    return initial_state.compose(Utheta)


def ansatz_qaoa_periodic(num_qubits, num_layers, initial_state,
                         mixer          = 'rx',
                         driver         = 'rzz',
                         driver_per     = 'ryy',
                         prefix         = 'θ',
                         **qc_options):
    """Function assembling a parametrized QAOA-inspired ansatz circuit
    with periodic entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    
    num_layers : int
        Number of layers (repetitions).
        
    initial_state: QuantumCircuit
        Circuit prepended at the beginning of the ansatz
    
    mixer: str, optional
        Strings specifying the single-parameter mixer rotation gate.
        Default is 'rx'.
    
    driver : str, optional
        Name of 2-qubit single-parameter entangling rotation gate.
        Default is 'rzz'.
        
    driver_per : str, optional
        Name of 2-qubit single-parameter entangling rotation gate for first
        and last qubit. Default is 'ryy'.
        
    prefix : str, optional
        String for name of abstract parameters. Default is 'θ'.
        
    qc_options : dict, optional
        Keyword arguments for the quantum circuit constructor. Default is empty.

    Raises
    ------
    ValueError
        Raises error if illegal values for mixer or driver gates.

    Returns
    -------
    Utheta : QuantumCircuit
        Parametrized ansatz circuit.

    """
    
    
    if mixer not in set(MIXER_GATES):
        message  = "Illegal value for mixer. Allowed values: "
        message += str(MIXER_GATES)
        raise ValueError(message)
        
    if driver not in DRIVER_GATES:
        message  = "Illegal value for driver. Allowed values: "
        message += str(DRIVER_GATES)
        raise ValueError(message)
    
    num_parameters  = 2*num_layers
    theta           = ParameterVector(prefix, num_parameters) 
            
    # internal layers
    Utheta = QuantumCircuit(num_qubits, **qc_options)
    for l in range(num_layers):
        
        # driver layer
        for j in range(0, num_qubits-1):
            getattr(Utheta, driver)(theta[2*l], j, j+1)
            
        # periodic layer
        getattr(Utheta, driver_per)(theta[2*l], 0, num_qubits-1)
        
        # mixer layer
        for j in range(num_qubits):
            getattr(Utheta, mixer)(theta[2*l+1], j)
    
    return initial_state.compose(Utheta)


ARCHTS = {'linear': ansatz_linear,
          'linear_periodic': ansatz_linear_periodic,
          'linear_alternating': ansatz_linear_alternating,
          'linear_alternating_periodic': ansatz_linear_alternating_periodic,
          'linear_alternating_periodic_bidirectional': ansatz_linear_alternating_periodic_bidirectional,
          'qaoa': ansatz_qaoa,
          'qaoa_periodic': ansatz_qaoa_periodic}


def get_ansatz_archt(name):
    """Return ansatz architecture driver if available.

    Parameters
    ----------
    name : str
        Name of architecture.

    Raises
    ------
    ValueError
        Raises error if architecture name not recognized.

    Returns
    -------
    Function
        Function for chosen architecture.

    """
    
    if name not in ARCHTS:
        message = "Ansatz architecture not implemented. Available architectures: "
        message += str(list(ARCHTS.keys()))
        raise ValueError(message)
        
    return ARCHTS[name]