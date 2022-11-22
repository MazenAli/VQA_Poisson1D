"""
Script for processing results of experiments on IBM backends
to test the accuracy of operator expectation values.
"""

# %% dependencies
#
import os
import sys
import argparse
import pickle
import json
import pandas as pd

import numpy as np

from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator, StatevectorSimulator
from qiskit.providers import JobStatus

from vqa_poisson_tools.poisson.cost import get_poisson_cost_driver
from vqa_poisson_tools.poisson.innerp import InnerProductStatevec
from vqa_poisson_tools.poisson.operator import Poisson1dVQAOperator_exact
from vqa_poisson_tools.poisson.solution import SolvePoisson1D


# %% allowed architechture inputs
#
case_hea      = ['linear', 'linear_periodic',
                 'linear_alternating', 'linear_alternating_periodic',
                 'linear_alternating_periodic_bidirectional']
case_qaoa     = ['qaoa']
case_qaoa_per = ['qaoa_periodic']
init_states   = ['empty', 'hadamards']


# %% argument parser
#
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--job_id',
                        required=True,
                        metavar='id',
                        help='<Required> job id')
    parser.add_argument('--force_process',
                       action='store_true',
                       help='Set flag to force post processing')
    parser.add_argument('--hub',
                        default='ibm-q',
                        metavar='hub',
                        help='Name of hub for provider')
    parser.add_argument('--group',
                        default='open',
                        metavar='group',
                        help='Name of group for provider')
    parser.add_argument('--project',
                        default='main',
                        metavar='project',
                        help='Name of project for provider')
    parser.add_argument('--input_dir',
                        default='./inputs',
                        metavar='dir',
                        help='Directory where inputs were stored')
    parser.add_argument('--output_dir',
                        default='./outputs',
                        metavar='dir',
                        help='Directory where outputs were stored')
    parser.add_argument('--post_processed_dir',
                        default='./post_processed',
                        metavar='dir',
                        help='Directory for saving post processed results')
    parser.add_argument('--ehningen',
                        action='store_true',
                        help='Set flag if accessing Ehningen because apparently IBM is shit at writing software...')
    parser.add_argument('--token',
                        default=None,
                        help='Account token, only required for Ehningen')
    parser.add_argument('--url',
                        default=None,
                        help='Authentication URL, only required for Ehningen')
    
    return parser.parse_args()


# %% load backend for experiment
#
def load_backend(args):
    hub      = args.hub
    group    = args.group
    project  = args.project
    backend  = args.backend
    ehningen = args.ehningen
    token    = args.token
    url      = args.url
    
    provider = None
    device   = None
    if backend == 'simulator':
        device = AerSimulator()
    
    elif ehningen:
        IBMQ.enable_account(token, url)
        provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        device   = provider.get_backend(backend)
    
    else:
    
        IBMQ.load_account()
        
        provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        device   = provider.get_backend(backend)
        
    return provider, device


# %% prepare cost drivers
#
def prepare_cost_driver(cost_method, op_driver, innerp_driver):
    
    # prepare list of drivers
    
    driver      = get_poisson_cost_driver(cost_method)
    cost_driver = driver(op_driver, innerp_driver)
        
    return cost_driver


# %% load experiments
#
def load_exps(args):
    
    # input
    
    exp_id    = args.job_id
    store_dir = args.post_processed_dir
    target    = os.path.join(store_dir, exp_id + '.csv')
    exists    = os.path.exists(target)
    force     = args.force_process
    
    # retrieve file names
    
    input_dir          = args.input_dir
    output_dir         = args.output_dir
  
    args_name          = exp_id + '.json'
    args_file          = os.path.join(input_dir, 'args', args_name)
    
    ansatz_name        = exp_id + '.pkl'
    ansatz_file        = os.path.join(input_dir, 'ansaetze', ansatz_name)
    
    rhs_name           = exp_id + '.pkl'
    rhs_file           = os.path.join(input_dir, 'rhs', rhs_name)
    
    trans_unbound_name = exp_id + '.pkl'
    trans_unbound_file = os.path.join(input_dir, 'trans_unbound', trans_unbound_name)
    
    init_params_name   = exp_id + '.pkl'
    init_params_file   = os.path.join(input_dir, 'init_params', init_params_name)
    
    results_name       = exp_id + '.pkl'
    results_file       = os.path.join(output_dir, results_name)
    
    # load data
    
    args_input = None
    with open(args_file) as file:
        args_input = json.load(file)
        
    ansatz_unbound = None
    with open(ansatz_file, 'rb') as file:
        ansatz_unbound = pickle.load(file)
        
    rhs_circ = None
    with open(rhs_file, 'rb') as file:
        rhs_circ = pickle.load(file)
        
    trans_unbound = None
    with open(trans_unbound_file, 'rb') as file:
        trans_unbound = pickle.load(file)
        
    init_params = None
    with open(init_params_file, 'rb') as file:
        init_params = pickle.load(file)
        
    results = None
    with open(results_file, 'rb') as file:
        results = pickle.load(file)
    
    if type(results) == JobStatus:
        msg  = "This job is not done: " + results.value
        sys.exit(msg)
        
    custom_tag        = args_input['custom_tag']
    backend           = args_input['backend']
    archt             = args_input['archt']
    num_qubits        = args_input['num_qubits']
    num_layers        = args_input['num_layers']

    rhs_type        = args_input['rhs']
    operator_method = args_input['operator']
    innerp_method   = args_input['innerp']
    cost_method     = args_input['cost']
    exact           = args_input['exact']
    optimizer       = args_input['optimizer']
    seed_init       = args_input['seed_init']
    
    shots       = args_input['shots']
    sabre       = args_input['transpile_sabre']
    mapomatic   = args_input['transpile_mapomatic']
    dd          = args_input['transpile_dd']
    mem         = args_input['mem']
    
    # check if exists
    
    if exists and not force:
        msg="Experiment ID already saved, skipping..."
        sys.exit(msg)

    
    # drivers
    num_parameters = ansatz_unbound.num_parameters
    op_driver      = Poisson1dVQAOperator_exact(ansatz_unbound)
    innerp_driver  = InnerProductStatevec(ansatz_unbound, rhs_circ)
    exact_cost     = prepare_cost_driver(cost_method, op_driver, innerp_driver)
    
    return (exp_id,
            custom_tag,
            backend,
            archt,
            ansatz_unbound,
            rhs_circ,
            num_qubits,
            num_layers,
            num_parameters,
            rhs_type,
            operator_method,
            innerp_method,
            cost_method,
            exact_cost,
            exact,
            optimizer,
            seed_init,
            init_params,
            shots,
            sabre,
            mapomatic,
            dd,
            mem,
            trans_unbound,
            results)


# %% compute exact solution
#
def exact_solution(rhs_circ):
    
    solver     = SolvePoisson1D(rhs_circ)
    solution   = solver.solve()
    normalized = solution/np.linalg.norm(solution)
    
    return normalized
        

# %% return cost series
#
def all_cost_series(results):
    
    cost = []
    
    for entry in results[1]:
        cost.append(entry['energy'])
        
    return cost


# %% return cost series
#
def accepted_cost_series(results, thetas):
    
    cost = []
    
    for theta in thetas:
        for entry in results[1]:
            if np.allclose(theta, entry['theta']):
                cost.append(entry['energy'])
                break
        
    return cost


# %% return gradient series
#
def all_grad_series(results):
    
    grad = []
    
    for entry in results[2]:
        grad.append(entry['grad'])
        
    return grad


# %% return exact cost series
#
def all_exact_cost_series(results, exact_cost):
    
    exact = []
    
    for entry in results[1]:
        theta = entry['theta']
        exact.append(exact_cost.cost_exact(theta))
        
    return exact


# %% return exact cost series
#
def accepted_exact_cost_series(thetas, exact_cost):
    
    exact = []
    
    for theta in thetas:
        exact.append(exact_cost.cost_exact(theta))
        
    return exact


# %% return exact gradient series
#
def all_exact_grad_series(results, exact_cost):
    
    exact = []
    
    for entry in results[2]:
        theta = entry['theta']
        exact.append(exact_cost.gradient_exact(theta))
        
    return exact


# %% cosine similarity
#
def compute_cos_sim(est_grad, exact_grad):
    
    num_its = len(exact_grad)
    errors  = [None]*num_its

    for m in range(num_its):
        exact     = exact_grad[m]
        est       = est_grad[m]
        errors[m] = np.inner(est, exact)/(np.linalg.norm(est)*np.linalg.norm(exact))
        
    return errors


# %% quantum fidelity for normalized vectors
#
def quantum_fidelity(x, y):
   
    result = np.abs(np.vdot(x, y))**2
    return result


# %% return solution fidelity series
#
def solution_fidels(ansatz_unbound, thetas, exact_vec):
    
    simulator      = StatevectorSimulator()
    fidels         = []
    num_parameters = ansatz_unbound.num_parameters
    
    for theta in thetas:
        bound = ansatz_unbound.bind_parameters(theta[0:num_parameters])
        vec   = simulator.run(bound).result().get_statevector().data
        fidels.append(quantum_fidelity(exact_vec, vec))
        
    return fidels


# %% extract metadata
#
def to_depths(trans_unbound_):
    
    is_list       = (type(trans_unbound_) is list)
    trans_unbound = trans_unbound_
    if not is_list:
        trans_unbound = [trans_unbound_]
        
    depths = []
    for qc in trans_unbound:
        depths.append(qc.depth())
            
    return depths


def to_cnots(trans_unbound_):
    
    is_list       = (type(trans_unbound_) is list)
    trans_unbound = trans_unbound_
    if not is_list:
        trans_unbound = [trans_unbound_]
        
    cnots = []
    for qc in trans_unbound:
        val = 0
        if 'cx' in qc.count_ops():
            val = qc.count_ops()['cx']
        cnots.append(val)
            
    return cnots


# %% save data
#
def save_data(args,
              exp_id,
              custom_tag,
              backend,
              archt,
              rhs_type,
              num_qubits,
              num_layers,
              num_parameters,
              seed_init,
              shots,
              depths,
              cnots,
              operator_method,
              innerp_method,
              cost_method,
              exact,
              optimizer,
              cost_series,
              exact_cost_series,
              cos_sim_series,
              acc_cost_series,
              acc_exact_cost_series,
              fidels,
              fidels_cost,
              fidels_grad,
              sabre,
              mapomatic,
              dd,
              mem):
    
    # store directory
    store_dir = args.post_processed_dir
    target    = os.path.join(store_dir, exp_id + '.csv')
    
    # input line
    input_dir = args.input_dir
    args_name = exp_id + '.json'
    args_file = os.path.join(input_dir, 'args', args_name)
    
    args_input = None
    with open(args_file) as file:
        args_input = json.load(file)
    
    # create data
    rotations    = args_input['rotations']
    entangler    = args_input['entangler']
    mixer        = args_input['mixer']
    driver       = args_input['driver']
    driver_per   = args_input['driver_per']
    initial_qaoa = args_input['initial_qaoa']
    
    data = {'custom_tag': pd.Series(custom_tag, dtype=str),
            'backend': pd.Series(backend, dtype=str),
            'cost_approx': pd.Series(cost_series, dtype=float),
            'cost_exact': pd.Series(exact_cost_series, dtype=float),
            'cosine_sim': pd.Series(cos_sim_series, dtype=float),
            'acc_cost_approx': pd.Series(acc_cost_series, dtype=float),
            'acc_cost_exact': pd.Series(acc_exact_cost_series, dtype=float),
            'fidelities': pd.Series(fidels, dtype=float),
            'fidelities_cost': pd.Series(fidels_cost, dtype=float),
            'fidelities_grad': pd.Series(fidels_grad, dtype=float),
            'archt': pd.Series(archt, dtype=str),
            'rhs': pd.Series(rhs_type, dtype=str),
            'qubits': pd.Series(num_qubits, dtype=int),
            'layers': pd.Series(num_layers, dtype=int),
            'parameters': pd.Series(num_parameters, dtype=int),
            'seed_init': pd.Series(seed_init, dtype=int),
            'exact': pd.Series(exact, dtype=bool),
            'shots': pd.Series(shots, dtype=int),
            'depth': pd.Series(depths, dtype=int),
            'cnots': pd.Series(cnots, dtype=int),
            'operator': pd.Series(operator_method, dtype=str),
            'innerp': pd.Series(innerp_method, dtype=str),
            'cost': pd.Series(cost_method, dtype=str),
            'optimizer': pd.Series(optimizer, dtype=str),
            'rotations': pd.Series(rotations, dtype=str),
            'entangler': pd.Series(entangler, dtype=str),
            'mixer': pd.Series(mixer, dtype=str),
            'driver': pd.Series(driver, dtype=str),
            'driver_per': pd.Series(driver_per, dtype=str),
            'initial_qaoa': pd.Series(initial_qaoa, dtype=str),
            'sabre': pd.Series(sabre, dtype=bool),
            'mapomatic': pd.Series(mapomatic, dtype=bool),
            'mem': pd.Series(mem, dtype=bool),
            'DD': pd.Series(dd, dtype=bool)}
    
    # store
    mode = 'w'
    df   = pd.DataFrame.from_dict(data)
    df.to_csv(target, mode=mode, header=True, index=True,
              index_label='Iteration', encoding='utf-8')
    
    return 0


# %% main
#
def main():
    args = parse_arguments()
    
    (exp_id,
     custom_tag,
     backend,
     archt,
     ansatz_unbound,
     rhs_circ,
     num_qubits,
     num_layers,
     num_parameters,
     rhs_type,
     operator_method,
     innerp_method,
     cost_method,
     exact_cost,
     exact,
     optimizer,
     seed_init,
     init_params,
     shots,
     sabre,
     mapomatic,
     dd,
     mem,
     trans_unbound,
     results) = load_exps(args)
  
    exact_vec         = exact_solution(rhs_circ)
    cost_series       = all_cost_series(results)
    grad_series       = all_grad_series(results)
    exact_cost_series = all_exact_cost_series(results, exact_cost)
    exact_grad_series = all_exact_grad_series(results, exact_cost)
    cos_sim_series    = compute_cos_sim(grad_series, exact_grad_series)
    thetas_           = results[3]
    opt               = results[0]['x']
    if opt is not None:
        thetas_ += [opt]
    
    thetas_cost = []
    for entry in results[1]:
        theta = entry['theta']
        thetas_cost.append(theta)
        
    thetas_grad = []
    for entry in results[2]:
        theta = entry['theta']
        thetas_grad.append(theta)
    
    fidels      = solution_fidels(ansatz_unbound, thetas_, exact_vec)
    fidels_cost = solution_fidels(ansatz_unbound, thetas_cost, exact_vec)
    fidels_grad = solution_fidels(ansatz_unbound, thetas_grad, exact_vec)
    
    acc_cost_series       = accepted_cost_series(results, thetas_)
    acc_exact_cost_series = accepted_exact_cost_series(thetas_, exact_cost)
    
    depths = None
    cnots  = None
    if not exact:
        depths = to_depths(trans_unbound)
        cnots  = to_cnots(trans_unbound)
    
    save_data(args,
              exp_id,
              custom_tag,
              backend,
              archt,
              rhs_type,
              num_qubits,
              num_layers,
              num_parameters,
              seed_init,
              shots,
              depths,
              cnots,
              operator_method,
              innerp_method,
              cost_method,
              exact,
              optimizer,
              cost_series,
              exact_cost_series,
              cos_sim_series,
              acc_cost_series,
              acc_exact_cost_series,
              fidels,
              fidels_cost,
              fidels_grad,
              sabre,
              mapomatic,
              dd,
              mem)
    

# %% run script
#
if __name__ == '__main__':
    main()