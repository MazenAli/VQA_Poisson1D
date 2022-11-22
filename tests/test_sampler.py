from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator, StatevectorSimulator

from vqa_poisson_tools.utils.sampler import VanillaSampler



def test_statevector():
    
    device = StatevectorSimulator()
    qc     = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    
    sampler = VanillaSampler(device)
    probs   = sampler.sample(qc)
    
    print("--------------------    Circuit    --------------------")
    print(qc)
    print("-------------------- Probabilities --------------------")
    print(probs)
    
    return 0



def test_aer(shots=1000):
    device = AerSimulator()
    qc     = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.measure_all()
    
    sampler = VanillaSampler(device, shots=shots)
    probs   = sampler.sample(qc)
    
    print("--------------------    Circuit    --------------------")
    print(qc)
    print("-------------------- Probabilities --------------------")
    print(probs)
    
    return 0



def test_mitigated(shots=1000):
    device = AerSimulator()
    qc     = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.measure_all()
    
    sampler = VanillaSampler(device, shots=shots)
    probs   = sampler.mitigated_sample(qc)
    
    print("--------------------    Circuit    --------------------")
    print(qc)
    print("-------------------- Probabilities --------------------")
    print(probs)
    
    return 0



def test_all(shots=1000):
    
    test_statevector()
    test_aer(shots=shots)
    test_mitigated(shots=shots)
    
    return 0