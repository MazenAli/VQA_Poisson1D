from math import pi
import numpy as np
import matplotlib.pyplot as plt

from qiskit.providers.aer import AerSimulator, StatevectorSimulator
from qiskit.algorithms.optimizers import L_BFGS_B, ADAM

from vqa_poisson_tools.ansatz.two_local import ansatz_linear_alternating
from vqa_poisson_tools.poisson.rhs import hn, hnx
from vqa_poisson_tools.poisson.operator import (Poisson1dVQAOperator_exact,
                                                Poisson1dVQAOperator_Paulis,
                                                Poisson1dVQAOperator_Sato21)
from vqa_poisson_tools.poisson.innerp import (InnerProductHTest,
                                              InnerProductOverlap,
                                              InnerProductStatevec)
from vqa_poisson_tools.poisson.cost import (Poisson1dVQACost_Sato21_Innerp,
                                            Poisson1dVQACost_Sato21_Overlap,
                                            Poisson1dVQACost_Sato21_Nonnorm,
                                            PoissonVQACost_BravoPrieto20_Global,
                                            PoissonVQACost_BravoPrieto20_Local)
from vqa_poisson_tools.poisson.solution import SolvePoisson1D
from vqa_poisson_tools.utils.utils import counts2probs
from qiskit_transpiler_tools.transpiler import TranspilerSabreMapomaticDD



def prepare_circuits_sato21(num_qubits = 3, num_layers = 1):
    # ansatz and rhs
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs    = hn(num_qubits)
    
    # operator drivers
    op_exact         = Poisson1dVQAOperator_exact(ansatz)
    op_paulis        = Poisson1dVQAOperator_Paulis(ansatz)
    op_sato21        = Poisson1dVQAOperator_Sato21(ansatz)
    
    # innerp drivers
    innerp_exact   = InnerProductStatevec(ansatz, rhs)
    innerp_htest   = InnerProductHTest(ansatz, rhs)
    innerp_overlap = InnerProductOverlap(ansatz, rhs)
    
    # cost drivers
    cost_sato21_exact_exact    = Poisson1dVQACost_Sato21_Innerp(op_exact, innerp_exact)

    cost_sato21_paulis_htest   = Poisson1dVQACost_Sato21_Innerp(op_paulis, innerp_htest)
    cost_sato21_paulis_overlap = Poisson1dVQACost_Sato21_Overlap(op_paulis, innerp_overlap)
    
    cost_sato21_sato21_htest   = Poisson1dVQACost_Sato21_Innerp(op_sato21, innerp_htest)
    cost_sato21_sato21_overlap = Poisson1dVQACost_Sato21_Overlap(op_sato21, innerp_overlap)
    
    cost_drivers = {'Sato21_Exact_Exact': cost_sato21_exact_exact,
                    'Sato21_Paulis_HTest': cost_sato21_paulis_htest,
                    'Sato21_Paulis_Overlap': cost_sato21_paulis_overlap,
                    'Sato21_Sato21_HTest': cost_sato21_sato21_htest,
                    'Sato21_Sato21_Overlap': cost_sato21_sato21_overlap}
    
    return cost_drivers



def prepare_circuits_sato21_grad(num_qubits = 3, num_layers = 1):
    # ansatz and rhs
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs    = hn(num_qubits)
    
    # operator drivers
    op_exact         = Poisson1dVQAOperator_exact(ansatz)
    op_paulis        = Poisson1dVQAOperator_Paulis(ansatz)
    op_sato21        = Poisson1dVQAOperator_Sato21(ansatz)
    
    # innerp drivers
    innerp_exact   = InnerProductStatevec(ansatz, rhs)
    innerp_htest   = InnerProductHTest(ansatz, rhs)
    
    # cost drivers
    cost_sato21_exact_exact    = Poisson1dVQACost_Sato21_Innerp(op_exact, innerp_exact)
    cost_sato21_paulis_htest   = Poisson1dVQACost_Sato21_Innerp(op_paulis, innerp_htest)
    cost_sato21_sato21_htest   = Poisson1dVQACost_Sato21_Innerp(op_sato21, innerp_htest)
    
    cost_drivers = {'Sato21_Exact_Exact': cost_sato21_exact_exact,
                    'Sato21_Paulis_HTest': cost_sato21_paulis_htest,
                    'Sato21_Sato21_HTest': cost_sato21_sato21_htest}
    
    return cost_drivers



def prepare_circuits_sato21_nonnorm(num_qubits = 3, num_layers = 1):
    # ansatz and rhs
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs    = hn(num_qubits)

    # operator drivers
    op_exact         = Poisson1dVQAOperator_exact(ansatz)
    op_paulis        = Poisson1dVQAOperator_Paulis(ansatz)
    op_sato21        = Poisson1dVQAOperator_Sato21(ansatz)
    
    # innerp drivers
    innerp_exact   = InnerProductStatevec(ansatz, rhs)
    innerp_htest   = InnerProductHTest(ansatz, rhs)
    
    # cost drivers
    cost_sato21_exact_exact    = Poisson1dVQACost_Sato21_Nonnorm(op_exact, innerp_exact)
    cost_sato21_paulis_htest   = Poisson1dVQACost_Sato21_Nonnorm(op_paulis, innerp_htest)
    cost_sato21_sato21_htest   = Poisson1dVQACost_Sato21_Nonnorm(op_sato21, innerp_htest)
    
    cost_drivers = {'Sato21_Nonnorm_Exact_Exact': cost_sato21_exact_exact,
                    'Sato21_Nonnorm_Paulis_HTest': cost_sato21_paulis_htest,
                    'Sato21_Nonnorm_Sato21_HTest': cost_sato21_sato21_htest}
    
    return cost_drivers


def prepare_circuits_sato21_nonnorm_grad(num_qubits = 3, num_layers = 1):
    # ansatz and rhs
    ansatz = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs    = hn(num_qubits)
    
    # operator drivers
    op_exact         = Poisson1dVQAOperator_exact(ansatz)
    op_paulis        = Poisson1dVQAOperator_Paulis(ansatz)
    op_sato21        = Poisson1dVQAOperator_Sato21(ansatz)
    
    # innerp drivers
    innerp_exact   = InnerProductStatevec(ansatz, rhs)
    innerp_htest   = InnerProductHTest(ansatz, rhs)
    
    # cost drivers
    cost_sato21_exact_exact    = Poisson1dVQACost_Sato21_Nonnorm(op_exact, innerp_exact)
    cost_sato21_paulis_htest   = Poisson1dVQACost_Sato21_Nonnorm(op_paulis, innerp_htest)
    cost_sato21_sato21_htest   = Poisson1dVQACost_Sato21_Nonnorm(op_sato21, innerp_htest)
    
    cost_drivers = {'Sato21_Nonnorm_Exact_Exact': cost_sato21_exact_exact,
                    'Sato21_Nonnorm_Paulis_HTest': cost_sato21_paulis_htest,
                    'Sato21_Nonnorm_Sato21_HTest': cost_sato21_sato21_htest}
    
    return cost_drivers

                    
def test_sato21_print_circuits(num_qubits = 3, num_layers = 1):
    
    cost_drivers = prepare_circuits_sato21(num_qubits, num_layers)
    
    # get_circuits
    print("------------------- get_circuits -------------------")
    print("------------------- ------------ -------------------")
    print("\n")
    
    for drivers in cost_drivers.items():
        msg   = f'----------- {drivers[0]} -----------'
        print(msg)
        circs = drivers[1].get_circuits()
    
        if isinstance(circs, list):
            for obj in circs:
                if isinstance(obj, list):
                    for qc in obj:
                        print(qc)
                        print("################")
                        print("################")
                else:
                    print(obj)
                    print("################")
                    print("################")
        print("\n")
    
    return 0



def test_sato21_print_errors(num_qubits = 3,
                             num_layers = 1,
                             shots      = 1000):
    
    # prepare
    cost_drivers = prepare_circuits_sato21(num_qubits, num_layers)
    exact_key    = 'Sato21_Exact_Exact'
    exact_driver = cost_drivers[exact_key]

    # parameter
    num_parameters = exact_driver.op_driver.ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)

    # exact
    exact_cost   = exact_driver.energy_exact(theta)
    
    # get measurements
    measurements = {}
    device       = AerSimulator()
    transpiler   = TranspilerSabreMapomaticDD(device)
    for drivers in cost_drivers.items():
        og_circuits = drivers[1].get_circuits()
        circuits    = drivers[1].get_circuits_energy(og_circuits, theta)
        out         = []
        for qc in circuits:
            counts = None
            if qc is not None:
                trans  = transpiler.transpile(qc)
                job    = device.run(trans, shots=shots)
                counts = job.result().get_counts()
                counts = counts2probs(counts)
            out.append(counts)
        
        measurements[drivers[0]] = out
                
    # compute cost and errors
    print("------------------- cost errors    -------------------")
    print("------------------- ------------ -------------------")
    print("\n")
    
    for drivers in cost_drivers.items():
        msg   = f'----------- {drivers[0]} -----------'
        print(msg)
        estimate = None
        if drivers[0] == exact_key:
            estimate = exact_cost
        else:
            estimate = drivers[1].cost_from_measurements(measurements[drivers[0]])
        abs_err  = abs(estimate-exact_cost)
        rel_err  = abs_err/abs(exact_cost)
        print("###")
        print("Abs error = ", abs_err)
        print("Rel error = ", rel_err)
        print("###")

    return 0



def test_sato21_nonnorm_print_errors(num_qubits = 3,
                                     num_layers = 1,
                                     shots      = 1000):
    
    # prepare
    cost_drivers = prepare_circuits_sato21_nonnorm(num_qubits, num_layers)
    exact_key    = 'Sato21_Nonnorm_Exact_Exact'
    exact_driver = cost_drivers[exact_key]
    
    # parameter
    num_parameters = exact_driver.op_driver.ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters+1)
    norm_sqrt      = theta[-1]

    # exact
    exact_cost     = exact_driver.energy_exact(theta)
    
    # get measurements
    measurements = {}
    device       = AerSimulator()
    transpiler   = TranspilerSabreMapomaticDD(device)
    for drivers in cost_drivers.items():
        og_circuits = drivers[1].get_circuits()
        circuits    = drivers[1].get_circuits_energy(og_circuits, theta)
        out         = []
        for qc in circuits:
            counts = None
            if qc is not None:
                trans  = transpiler.transpile(qc)
                job    = device.run(trans, shots=shots)
                counts = job.result().get_counts()
                counts = counts2probs(counts)
            out.append(counts)
        
        measurements[drivers[0]] = out
                
    # compute cost and errors
    print("------------------- cost errors -------------------")
    print("------------------- ------------ -------------------")
    print("\n")
    
    for drivers in cost_drivers.items():
        msg   = f'----------- {drivers[0]} -----------'
        print(msg)
        estimate = None
        if drivers[0] == exact_key:
            estimate = exact_cost
        else:
            estimate = drivers[1].cost_from_measurements(measurements[drivers[0]]+[norm_sqrt])
        abs_err  = abs(estimate-exact_cost)
        rel_err  = abs_err/abs(exact_cost)
        print("###")
        print("Abs error = ", abs_err)
        print("Rel error = ", rel_err)
        print("###")

    return 0



def test_sato21_gradient_exact_plot_errors(num_qubits = 3, num_layers = 1):
    
    # ansatz and rhs
    ansatz         = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs            = hn(num_qubits)
    
    # parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # drivers
    op_exact     = Poisson1dVQAOperator_exact(ansatz)
    innerp_exact = InnerProductStatevec(ansatz, rhs)
    cost_exact   = Poisson1dVQACost_Sato21_Innerp(op_exact, innerp_exact)
    
    # get gradient
    grad_trial = cost_exact.gradient_exact(theta)
    
    # approximate gradient
    steps  = np.linspace(1., 0.00001, 5)
    costf  = cost_exact.energy_exact
    errors = np.zeros(steps.size)
    for i in range(steps.size):
        grad_test = np.empty(num_parameters)
        for j in range(num_parameters):
            theta_j       = np.copy(theta)
            theta_j[j]   += steps[i]
            grad_test[j]  = costf(theta_j) - costf(theta)
            grad_test[j] /= steps[i]
        
        errors[i] = np.linalg.norm(grad_trial-grad_test)
        
    plt.plot(steps, errors, label = "grad error")
    plt.legend()
    plt.show()
    print("Error vector:\n", errors)

    return 0



def test_sato21_nonnorm_gradient_exact_plot_errors(num_qubits = 3, num_layers = 1):
    
    # ansatz and rhs
    ansatz         = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs            = hn(num_qubits)
    
    # parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters+1)
    
    # drivers
    op_exact     = Poisson1dVQAOperator_exact(ansatz)
    innerp_exact = InnerProductStatevec(ansatz, rhs)
    cost_exact   = Poisson1dVQACost_Sato21_Nonnorm(op_exact, innerp_exact)
    
    # get gradient
    grad_trial = cost_exact.gradient_exact(theta)
    
    # approximate gradient
    steps  = np.linspace(1., 0.00001, 5)
    costf  = cost_exact.energy_exact
    errors = np.zeros(steps.size)
    for i in range(steps.size):
        grad_test = np.empty(num_parameters+1)
        for j in range(num_parameters+1):
            theta_j       = np.copy(theta)
            theta_j[j]   += steps[i]
            grad_test[j]  = costf(theta_j) - costf(theta)
            grad_test[j] /= steps[i]
        
        errors[i] = np.linalg.norm(grad_trial-grad_test)
        
    plt.plot(steps, errors, label = "grad error")
    plt.legend()
    plt.show()
    print("Error vector:\n", errors)

    return 0



def test_bravoprieto20_global_gradient_exact_plot_errors(num_qubits = 3, num_layers = 1):
    
    # ansatz and rhs
    ansatz         = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs            = hnx(num_qubits)
    
    # parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # drivers
    op_exact     = Poisson1dVQAOperator_exact(ansatz)
    innerp_exact = InnerProductStatevec(ansatz, rhs)
    cost_exact   = PoissonVQACost_BravoPrieto20_Global(op_exact, innerp_exact)
    
    # get gradient
    grad_trial = cost_exact.gradient_exact(theta)
    
    # approximate gradient
    steps  = np.linspace(1., 0.00001, 5)
    costf  = cost_exact.energy_exact
    errors = np.zeros(steps.size)
    for i in range(steps.size):
        grad_test = np.empty(num_parameters)
        for j in range(num_parameters):
            theta_j       = np.copy(theta)
            theta_j[j]   += steps[i]
            grad_test[j]  = costf(theta_j) - costf(theta)
            grad_test[j] /= steps[i]
        
        errors[i] = np.linalg.norm(grad_trial-grad_test)
        
    plt.plot(steps, errors, label = "grad error")
    plt.legend()
    plt.show()
    print("Error vector:\n", errors)

    return 0



def quantum_fidelity(x, y):
   
    result = np.abs(np.vdot(x, y))**2
    return result



def test_bravoprieto20_global_gradient_bfgs(num_qubits = 3,
                                            num_layers = 1,
                                            maxiter    = 50):
    
    # ansatz and rhs
    ansatz         = ansatz_linear_alternating(num_qubits,
                                               num_layers,
                                               entangler='cz',
                                               name = 'psi')
    rhs            = hnx(num_qubits)
    
    # exact solution
    solver     = SolvePoisson1D(rhs)
    solution   = solver.solve()
    normalized = solution/np.linalg.norm(solution)
    
    # parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # drivers
    op_exact     = Poisson1dVQAOperator_exact(ansatz)
    innerp_exact = InnerProductStatevec(ansatz, rhs)
    cost_exact   = PoissonVQACost_BravoPrieto20_Global(op_exact, innerp_exact)
    costf        = cost_exact.energy_exact
    gradf        = cost_exact.gradient_exact
    
    # callback
    energies   = []
    fidelities = []
    simulator  = StatevectorSimulator()
    def verbose_bfgs(theta):
        bound    = ansatz.bind_parameters(theta)
        psi      = simulator.run(bound).result().get_statevector().data
        energy   = costf(theta)
        fidelity = quantum_fidelity(normalized, psi)
        energies.append(energy)
        fidelities.append(fidelity)
    
    # optimize
    bfgs          = L_BFGS_B(maxiter=maxiter, iprint=1, callback=verbose_bfgs)
    bfgs.minimize(costf, theta, gradf)
    
    # approximate gradient
    plt.plot(energies, label="energies")
    plt.legend()
    plt.show()
    
    plt.plot(fidelities, label="fidelities")
    plt.legend()
    plt.show()

    return 0



def test_bravoprieto20_local_gradient_bfgs(num_qubits = 3,
                                           num_layers = 1,
                                           maxiter    = 50):
    
    # ansatz and rhs
    ansatz         = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs            = hnx(num_qubits)
    
    # exact solution
    solver     = SolvePoisson1D(rhs)
    solution   = solver.solve()
    normalized = solution/np.linalg.norm(solution)
    
    # parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # drivers
    op_exact     = Poisson1dVQAOperator_exact(ansatz)
    innerp_exact = InnerProductStatevec(ansatz, rhs)
    cost_exact   = PoissonVQACost_BravoPrieto20_Local(op_exact, innerp_exact)
    costf        = cost_exact.energy_exact
    gradf        = cost_exact.gradient_exact
    
    # callback
    energies   = []
    fidelities = []
    simulator  = StatevectorSimulator()
    def verbose_bfgs(theta):
        bound    = ansatz.bind_parameters(theta)
        psi      = simulator.run(bound).result().get_statevector().data
        energy   = costf(theta)
        fidelity = quantum_fidelity(normalized, psi)
        energies.append(energy)
        fidelities.append(fidelity)
    
    # optimize
    bfgs          = L_BFGS_B(maxiter=maxiter, iprint=1, callback=verbose_bfgs)
    bfgs.optimize(num_parameters,
                  costf,
                  gradf,
                  initial_point=theta)
    
    # approximate gradient
    plt.plot(energies, label="energies")
    plt.legend()
    plt.show()
    
    plt.plot(fidelities, label="fidelities")
    plt.legend()
    plt.show()

    return 0



def test_bravoprieto20_global_gradient_adam(num_qubits = 3,
                                            num_layers = 1,
                                            maxiter    = 50):
    
    # ansatz and rhs
    ansatz         = ansatz_linear_alternating(num_qubits,
                                               num_layers,
                                               name = 'psi')
    rhs            = hnx(num_qubits)
    
    # parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # drivers
    op_exact     = Poisson1dVQAOperator_exact(ansatz)
    innerp_exact = InnerProductStatevec(ansatz, rhs)
    cost_exact   = PoissonVQACost_BravoPrieto20_Global(op_exact, innerp_exact)
    costf        = cost_exact.energy_exact
    gradf        = cost_exact.gradient_exact
    
    # optimize
    adam          = ADAM(maxiter=maxiter, amsgrad=True)
    adam.minimize(costf, theta, gradf)

    return 0



def test_sato21_gradient_bfgs(num_qubits = 3,
                              num_layers = 1,
                              maxiter    = 50):
    
    # ansatz and rhs
    ansatz         = ansatz_linear_alternating(num_qubits, num_layers, name = 'psi')
    rhs            = hnx(num_qubits)
    
    # exact solution
    solver     = SolvePoisson1D(rhs)
    solution   = solver.solve()
    normalized = solution/np.linalg.norm(solution)
    
    # parameter
    num_parameters = ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # drivers
    op_exact     = Poisson1dVQAOperator_exact(ansatz)
    innerp_exact = InnerProductStatevec(ansatz, rhs)
    cost_exact   = Poisson1dVQACost_Sato21_Innerp(op_exact, innerp_exact)
    costf        = cost_exact.energy_exact
    gradf        = cost_exact.gradient_exact
    
    # callback
    energies   = []
    fidelities = []
    simulator  = StatevectorSimulator()
    def verbose_bfgs(theta):
        bound    = ansatz.bind_parameters(theta)
        psi      = simulator.run(bound).result().get_statevector().data
        energy   = costf(theta)
        fidelity = quantum_fidelity(normalized, psi)
        energies.append(energy)
        fidelities.append(fidelity)
    
    # optimize
    bfgs          = L_BFGS_B(maxiter=maxiter, iprint=1, callback=verbose_bfgs)
    bfgs.minimize(costf, theta, gradf)
    
    # approximate gradient
    plt.plot(energies, label="energies")
    plt.legend()
    plt.show()
    
    plt.plot(fidelities, label="fidelities")
    plt.legend()
    plt.show()

    return 0



def test_sato21_gradients_print_errors(num_qubits = 3,
                                       num_layers = 1,
                                       shots      = 1000):
    # prepare
    cost_drivers = prepare_circuits_sato21_grad(num_qubits, num_layers)
    exact_key    = 'Sato21_Exact_Exact'
    exact_driver = cost_drivers[exact_key]
    
    # parameter
    num_parameters = exact_driver.op_driver.ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    # exact gradient
    grad_exact     = exact_driver.gradient_exact(theta)
    
    # get measurements
    measurements = {}
    device       = AerSimulator()
    transpiler   = TranspilerSabreMapomaticDD(device)
    for drivers in cost_drivers.items():
        circuits = drivers[1].get_circuits_gradient(drivers[1].get_circuits(), theta)
        out      = []
        for qc in circuits:
            counts = None
            if qc is not None:
                trans  = transpiler.transpile(qc)
                job    = device.run(trans, shots=shots)
                counts = job.result().get_counts()
                counts = counts2probs(counts)
            out.append(counts)
        
        measurements[drivers[0]] = out
                
    # compute cost and errors
    print("------------------- gradient errors -------------------")
    print("------------------- ------------ -------------------")
    print("\n")
    
    for drivers in cost_drivers.items():
        msg   = f'----------- {drivers[0]} -----------'
        print(msg)
        grad_est = None
        if drivers[0] == exact_key:
            grad_est = grad_exact
        else:
            grad_est = drivers[1].gradient_from_measurements(measurements[drivers[0]])
        abs_err  = np.linalg.norm(grad_exact-grad_est)
        rel_err  = abs_err/np.linalg.norm(grad_exact)
        print("###")
        print("Abs error = ", abs_err)
        print("Rel error = ", rel_err)
        print("###")

    return 0



def test_sato21_nonnorm_gradients_print_errors(num_qubits = 3,
                                               num_layers = 1,
                                               shots      = 1000):
    # prepare
    cost_drivers = prepare_circuits_sato21_nonnorm_grad(num_qubits, num_layers)
    exact_key    = 'Sato21_Nonnorm_Exact_Exact'
    exact_driver = cost_drivers[exact_key]
    
    # parameter
    num_parameters = exact_driver.op_driver.ansatz.num_parameters
    rng            = np.random.default_rng(seed=0)
    theta          = rng.uniform(low=0., high=4.*pi,  size=num_parameters+1)
    norm_sqrt      = theta[-1]
    
    # exact gradient
    grad_exact     = exact_driver.gradient_exact(theta)
    
    # get measurements
    measurements = {}
    device       = AerSimulator()
    transpiler   = TranspilerSabreMapomaticDD(device)
    for drivers in cost_drivers.items():
        circuits = drivers[1].get_circuits_gradient(drivers[1].get_circuits(), theta[0:num_parameters])
        out      = []
        for qc in circuits:
            counts = None
            if qc is not None:
                trans  = transpiler.transpile(qc)
                job    = device.run(trans, shots=shots)
                counts = job.result().get_counts()
                counts = counts2probs(counts)
            out.append(counts)
        
        measurements[drivers[0]] = out
                
    # compute cost and errors
    print("------------------- gradient errors -------------------")
    print("------------------- ------------ -------------------")
    print("\n")
    
    for drivers in cost_drivers.items():
        msg   = f'----------- {drivers[0]} -----------'
        print(msg)
        grad_est = None
        if drivers[0] == exact_key:
            grad_est = grad_exact
        else:
            grad_est = drivers[1].gradient_from_measurements(measurements[drivers[0]]+[norm_sqrt])
        abs_err  = np.linalg.norm(grad_exact-grad_est)
        rel_err  = abs_err/np.linalg.norm(grad_exact)
        print("###")
        print("Abs error = ", abs_err)
        print("Rel error = ", rel_err)
        print("###")

    return 0



def test_all(num_qubits = 3, num_layers = 1, shots = 1000, maxiter=50):
    
    test_sato21_print_circuits(num_qubits, num_layers)
    test_sato21_print_errors(num_qubits, num_layers, shots)
    test_sato21_nonnorm_print_errors(num_qubits, num_layers, shots)
    test_sato21_gradient_exact_plot_errors(num_qubits, num_layers)
    test_sato21_nonnorm_gradient_exact_plot_errors(num_qubits, num_layers)
    test_bravoprieto20_global_gradient_exact_plot_errors(num_qubits, num_layers)
    test_bravoprieto20_global_gradient_bfgs(num_qubits, num_layers, maxiter)
    test_bravoprieto20_local_gradient_bfgs(num_qubits, num_layers, maxiter)
    test_sato21_gradients_print_errors(num_qubits, num_layers, shots)
    test_sato21_nonnorm_gradients_print_errors(num_qubits, num_layers, shots)
    
    return 0