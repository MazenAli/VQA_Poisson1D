from math import pi
import numpy as np

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator, StatevectorSimulator

from qiskit.algorithms.optimizers import SPSA, NFT, L_BFGS_B, POWELL, ADAM
from vqa_poisson_tools.ansatz.two_local import ansatz_linear_alternating
from vqa_poisson_tools.poisson.rhs import hn, hnx
from vqa_poisson_tools.poisson.operator import (Poisson1dVQAOperator_exact,
                                                Poisson1dVQAOperator_Sato21,
                                                Poisson1dVQAOperator_SatoLiu21)
from vqa_poisson_tools.poisson.innerp import (InnerProductHTest,
                                              InnerProductStatevec)
from vqa_poisson_tools.poisson.cost import (Poisson1dVQACost_Sato21_Innerp,
                                            Poisson1dVQACost_Sato21_Nonnorm)
from vqa_poisson_tools.poisson.cost_estimator import (MeasCostEstimator,
                                                      MeasCostNonnormEstimator,
                                                      ExactCostEstimator)
from vqa_poisson_tools.utils.sampler import VanillaSampler
from vqa_poisson_tools.poisson.solution import SolvePoisson1D
from qiskit_transpiler_tools.transpiler import TranspilerSabreMapomaticDD


def test_cost_estimator(num_qubits=3, num_layers=1, shots=1000):
    
    # sampler
    device  = AerSimulator()
    sampler = VanillaSampler(device, shots=shots)
    
    # transpiler
    transpiler = TranspilerSabreMapomaticDD(device)
    
    # problem input
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs    = hn(num_qubits)
    
    # estimator
    op_sato21    = Poisson1dVQAOperator_Sato21(ansatz)
    innerp_htest = InnerProductHTest(ansatz, rhs)
    cost_driver  = Poisson1dVQACost_Sato21_Innerp(op_sato21, innerp_htest)
    og_circuits  = cost_driver.get_circuits()
    trans        = transpiler.transpile(og_circuits)
    estimator    = MeasCostEstimator(trans, cost_driver, sampler)
    
    # exact
    op_exact     = Poisson1dVQAOperator_exact(ansatz)
    innerp_exact = InnerProductStatevec(ansatz, rhs)
    cost_exact   = Poisson1dVQACost_Sato21_Innerp(op_exact, innerp_exact)
    exact        = ExactCostEstimator(cost_exact)
    
    # random parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # error
    est_cost = estimator.cost(theta)
    est_grad = estimator.gradient(theta)
    
    exact_cost = exact.cost(theta)
    exact_grad = exact.gradient(theta)
    
    abs_err_cost = abs(est_cost-exact_cost)
    rel_err_cost = abs_err_cost/abs(exact_cost)
    
    abs_err_grad = np.linalg.norm(est_grad-exact_grad)
    rel_err_grad = abs_err_grad/np.linalg.norm(exact_grad)
    
    print("Abs error cost : ", abs_err_cost)
    print("Rel error cost : ", rel_err_cost)
    print("Abs error grad : ", abs_err_grad)
    print("Rel error grad : ", rel_err_grad)
    
    print("################# LOGS #################")
    print("Estimator cost    : \n", estimator.energy_log)
    print("Estimator gradient: \n", estimator.grad_log)
    print("Exact cost        : \n", exact.energy_log)
    print("Exact gradient    : \n", exact.grad_log)
    
    return 0



def test_cost_nonnorm_estimator(num_qubits=3, num_layers=1, shots=1000):
    
    # sampler
    device  = AerSimulator()
    sampler = VanillaSampler(device, shots=shots)
    
    # transpiler
    transpiler = TranspilerSabreMapomaticDD(device)
    
    # problem input
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs    = hn(num_qubits)
    
    # estimator
    op_sato21    = Poisson1dVQAOperator_Sato21(ansatz)
    innerp_htest = InnerProductHTest(ansatz, rhs)
    cost_driver  = Poisson1dVQACost_Sato21_Nonnorm(op_sato21, innerp_htest)
    og_circuits  = cost_driver.get_circuits()
    trans        = transpiler.transpile(og_circuits)
    estimator    = MeasCostNonnormEstimator(trans, cost_driver, sampler)
    
    # exact
    op_exact     = Poisson1dVQAOperator_exact(ansatz)
    innerp_exact = InnerProductStatevec(ansatz, rhs)
    cost_exact   = Poisson1dVQACost_Sato21_Nonnorm(op_exact, innerp_exact)
    exact        = ExactCostEstimator(cost_exact)
    
    # random parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters+1)
    
    # error
    est_cost = estimator.cost(theta)
    est_grad = estimator.gradient(theta)
    
    exact_cost = exact.cost(theta)
    exact_grad = exact.gradient(theta)
    
    abs_err_cost = abs(est_cost-exact_cost)
    rel_err_cost = abs_err_cost/abs(exact_cost)
    
    abs_err_grad = np.linalg.norm(est_grad-exact_grad)
    rel_err_grad = abs_err_grad/np.linalg.norm(exact_grad)
    
    print("Abs error cost : ", abs_err_cost)
    print("Rel error cost : ", rel_err_cost)
    print("Abs error grad : ", abs_err_grad)
    print("Rel error grad : ", rel_err_grad)
    
    print("################# LOGS #################")
    print("Estimator cost    : \n", estimator.energy_log)
    print("Estimator gradient: \n", estimator.grad_log)
    print("Exact cost        : \n", exact.energy_log)
    print("Exact gradient    : \n", exact.grad_log)
    
    return 0



def test_sato21_spsa(num_qubits = 3,
                     num_layers = 1,
                     maxiter    = 100,
                     shots      = 1000):
    
    
    # sampler
    device  = AerSimulator()
    sampler = VanillaSampler(device, shots=shots)
    
    # transpiler
    transpiler = TranspilerSabreMapomaticDD(device)
    
    # problem input
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs    = hnx(num_qubits)
    
    # estimator
    op_sato21    = Poisson1dVQAOperator_Sato21(ansatz)
    innerp_htest = InnerProductHTest(ansatz, rhs)
    cost_driver  = Poisson1dVQACost_Sato21_Innerp(op_sato21, innerp_htest)
    og_circuits  = cost_driver.get_circuits()
    trans        = transpiler.transpile(og_circuits)
    estimator    = MeasCostNonnormEstimator(trans, cost_driver, sampler)
    
    # exact solution
    solver     = SolvePoisson1D(rhs)
    solution   = solver.solve()
    normalized = solution/np.linalg.norm(solution)
    
    # random parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # callback
    energies   = []
    fidelities = []
    statevec   = StatevectorSimulator()
    
    def quantum_fidelity(x, y):
       
        result = np.abs(np.vdot(x, y))**2
        return result
    
    def callback_spsa(num_evals, theta, energy, learning_rate, accepted):
        energies.append(energy)
        psi       = statevec.run(ansatz.bind_parameters(theta)).result().get_statevector().data
        fidelity  = quantum_fidelity(normalized, psi)
        fidelities.append(fidelity)

    
    # optimize
    spsa = SPSA(maxiter=maxiter,
                blocking=True,
                trust_region=True,
                learning_rate=1.,
                perturbation=0.1,
                callback=callback_spsa)
    spsa.minimize(estimator.cost, theta)
    
    # results
    plt.plot(energies, label="energies")
    plt.legend()
    plt.show()
    
    plt.plot(fidelities, label="fidelities")
    plt.legend()
    plt.show()

    return 0



def test_satoliu21_spsa(num_qubits = 3,
                        num_layers = 1,
                        maxiter    = 100,
                        shots      = 1000):
    
    
    # sampler
    device  = AerSimulator()
    sampler = VanillaSampler(device, shots=shots)
    
    # transpiler
    transpiler = TranspilerSabreMapomaticDD(device)
    
    # problem input
    qc     = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.h(q)
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    ansatz = qc.compose(ansatz)
    rhs    = hnx(num_qubits)
    
    # estimator
    op_sato21    = Poisson1dVQAOperator_SatoLiu21(ansatz)
    innerp_htest = InnerProductHTest(ansatz, rhs)
    cost_driver  = Poisson1dVQACost_Sato21_Innerp(op_sato21, innerp_htest)
    og_circuits  = cost_driver.get_circuits()
    trans        = transpiler.transpile(og_circuits)
    estimator    = MeasCostEstimator(trans, cost_driver, sampler)
    
    # exact solution
    solver     = SolvePoisson1D(rhs)
    solution   = solver.solve()
    normalized = solution/np.linalg.norm(solution)
    
    # random parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # callback
    energies   = []
    fidelities = []
    statevec   = StatevectorSimulator()
    
    def quantum_fidelity(x, y):
       
        result = np.abs(np.vdot(x, y))**2
        return result
    
    def callback_spsa(num_evals, theta, energy, learning_rate, accepted):
        energies.append(energy)
        psi       = statevec.run(ansatz.bind_parameters(theta)).result().get_statevector().data
        fidelity  = quantum_fidelity(normalized, psi)
        fidelities.append(fidelity)

    
    # optimize
    spsa = SPSA(maxiter=maxiter,
                blocking=True,
                trust_region=True,
                learning_rate=1.,
                perturbation=0.1,
                callback=callback_spsa)
    spsa.minimize(estimator.cost, theta)
    
    # results
    plt.plot(energies, label="energies")
    plt.legend()
    plt.show()
    
    plt.plot(fidelities, label="fidelities")
    plt.legend()
    plt.show()

    return 0



def test_satoliu21_nft(num_qubits = 3,
                       num_layers = 1,
                       maxiter    = 100,
                       interval   = 32,
                       shots      = 1000):
    
    
    # sampler
    device  = AerSimulator()
    sampler = VanillaSampler(device, shots=shots)
    
    # transpiler
    transpiler = TranspilerSabreMapomaticDD(device)
    
    # problem input
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs    = hnx(num_qubits)
    
    # estimator
    op_sato21    = Poisson1dVQAOperator_SatoLiu21(ansatz)
    innerp_htest = InnerProductHTest(ansatz, rhs)
    cost_driver  = Poisson1dVQACost_Sato21_Nonnorm(op_sato21, innerp_htest)
    og_circuits  = cost_driver.get_circuits()
    trans        = transpiler.transpile(og_circuits)
    estimator    = MeasCostNonnormEstimator(trans, cost_driver, sampler)
    
    # exact solution
    solver     = SolvePoisson1D(rhs)
    solution   = solver.solve()
    normalized = solution/np.linalg.norm(solution)
    
    # random parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # callback
    fidelities = []
    statevec   = StatevectorSimulator()
    
    def quantum_fidelity(x, y):
       
        result = np.abs(np.vdot(x, y))**2
        return result
    
    def callback_nft(theta):
        psi       = statevec.run(ansatz.bind_parameters(theta)).result().get_statevector().data
        fidelity  = quantum_fidelity(normalized, psi)
        fidelities.append(fidelity)

    
    # optimize
    nft = NFT(maxiter=maxiter,
              reset_interval=interval,
              callback=callback_nft)
    nft.minimize(estimator.cost, theta)
    
    # results
    
    plt.plot(fidelities, label="fidelities")
    plt.legend()
    plt.show()

    return 0



def test_satoliu21_bfgs(num_qubits = 3,
                        num_layers = 1,
                        maxiter    = 30,
                        shots      = 1000):
    
    
    # sampler
    device  = AerSimulator()
    sampler = VanillaSampler(device, shots=shots)
    
    # transpiler
    transpiler = TranspilerSabreMapomaticDD(device)
    
    # problem input
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs    = hnx(num_qubits)
    
    # estimator
    op_sato21    = Poisson1dVQAOperator_SatoLiu21(ansatz)
    innerp_htest = InnerProductHTest(ansatz, rhs)
    cost_driver  = Poisson1dVQACost_Sato21_Innerp(op_sato21, innerp_htest)
    og_circuits  = cost_driver.get_circuits()
    trans        = transpiler.transpile(og_circuits)
    estimator    = MeasCostEstimator(trans, cost_driver, sampler)
    
    # exact solution
    solver     = SolvePoisson1D(rhs)
    solution   = solver.solve()
    normalized = solution/np.linalg.norm(solution)
    
    # random parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # callback
    fidelities = []
    statevec   = StatevectorSimulator()
    
    def quantum_fidelity(x, y):
       
        result = np.abs(np.vdot(x, y))**2
        return result
    
    psi       = statevec.run(ansatz.bind_parameters(theta)).result().get_statevector().data
    fidelity  = quantum_fidelity(normalized, psi)
    fidelities.append(fidelity)
    def callback_bfgs(theta):
        psi       = statevec.run(ansatz.bind_parameters(theta)).result().get_statevector().data
        fidelity  = quantum_fidelity(normalized, psi)
        fidelities.append(fidelity)
    
    # optimize
    bfgs = L_BFGS_B(maxiter=maxiter, iprint=1, callback=callback_bfgs)
    bfgs.minimize(estimator.cost, theta, estimator.gradient)
    
    # results
    
    plt.plot(fidelities, label="fidelities")
    plt.legend()
    plt.show()

    return 0



def test_satoliu21_powell(num_qubits = 3,
                          num_layers = 1,
                          maxfev     = 200,
                          shots      = 1000):
    
    
    # sampler
    device  = AerSimulator()
    sampler = VanillaSampler(device, shots=shots)
    
    # transpiler
    transpiler = TranspilerSabreMapomaticDD(device)
    
    # problem input
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs    = hnx(num_qubits)
    
    # estimator
    op_sato21    = Poisson1dVQAOperator_SatoLiu21(ansatz)
    innerp_htest = InnerProductHTest(ansatz, rhs)
    cost_driver  = Poisson1dVQACost_Sato21_Innerp(op_sato21, innerp_htest)
    og_circuits  = cost_driver.get_circuits()
    trans        = transpiler.transpile(og_circuits)
    estimator    = MeasCostEstimator(trans, cost_driver, sampler)
    
    # exact solution
    solver     = SolvePoisson1D(rhs)
    solution   = solver.solve()
    normalized = solution/np.linalg.norm(solution)
    
    # random parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # callback
    fidelities = []
    statevec   = StatevectorSimulator()
    
    def quantum_fidelity(x, y):
       
        result = np.abs(np.vdot(x, y))**2
        return result
    
    psi       = statevec.run(ansatz.bind_parameters(theta)).result().get_statevector().data
    fidelity  = quantum_fidelity(normalized, psi)
    fidelities.append(fidelity)
    def callback_powell(theta):
        psi       = statevec.run(ansatz.bind_parameters(theta)).result().get_statevector().data
        fidelity  = quantum_fidelity(normalized, psi)
        fidelities.append(fidelity)
    
    # optimize
    bfgs = POWELL(maxfev=maxfev, disp=True, callback=callback_powell)
    bfgs.minimize(estimator.cost, theta, estimator.gradient)
    
    # results
    
    plt.plot(fidelities, label="fidelities")
    plt.legend()
    plt.show()

    return 0



def test_satoliu21_adam(num_qubits = 3,
                        num_layers = 1,
                        maxiter    = 50,
                        shots      = 1000):
    
    
    # sampler
    device  = AerSimulator()
    sampler = VanillaSampler(device, shots=shots)
    
    # transpiler
    transpiler = TranspilerSabreMapomaticDD(device)
    
    # problem input
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs    = hnx(num_qubits)
    
    # estimator
    op_sato21    = Poisson1dVQAOperator_SatoLiu21(ansatz)
    innerp_htest = InnerProductHTest(ansatz, rhs)
    cost_driver  = Poisson1dVQACost_Sato21_Innerp(op_sato21, innerp_htest)
    og_circuits  = cost_driver.get_circuits()
    trans        = transpiler.transpile(og_circuits)
    estimator    = MeasCostEstimator(trans, cost_driver, sampler)
    
    # exact solution
    solver     = SolvePoisson1D(rhs)
    solution   = solver.solve()
    normalized = solution/np.linalg.norm(solution)
    
    # random parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # callback
    fidelities = []
    statevec   = StatevectorSimulator()
    
    def quantum_fidelity(x, y):
       
        result = np.abs(np.vdot(x, y))**2
        return result
    
    psi       = statevec.run(ansatz.bind_parameters(theta)).result().get_statevector().data
    fidelity  = quantum_fidelity(normalized, psi)
    fidelities.append(fidelity)
    
    # optimize
    adam = ADAM(maxiter=maxiter,
                amsgrad=True)
    adam.minimize(estimator.cost, theta, estimator.gradient)
    
    for entry in estimator.grad_log:
        theta     = entry['theta']
        psi       = statevec.run(ansatz.bind_parameters(theta)).result().get_statevector().data
        fidelity  = quantum_fidelity(normalized, psi)
        fidelities.append(fidelity)

    # results
    
    plt.plot(fidelities, label="fidelities")
    plt.legend()
    plt.show()
    return 0