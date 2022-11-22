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
from numpy import nan

from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator
from qiskit.providers import JobStatus
import mthree

from vqa_poisson_tools.poisson.cost import get_poisson_cost_driver
from vqa_poisson_tools.poisson.innerp import (get_innerp_circuit_driver,
                                              InnerProductStatevec)
from vqa_poisson_tools.poisson.operator import (get_poisson_operator_driver,
                                                Poisson1dVQAOperator_exact)
from vqa_poisson_tools.utils.utils import counts2probs


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
    
    parser.add_argument('--exp_id',
                        required=True,
                        metavar='id',
                        help='<Required> job id')
    parser.add_argument('--error_type',
                        default='cosine_sim',
                        metavar='type',
                        help='Type of error to calculate (absolute, relative or cosine_sim)')
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
    parser.add_argument('--mem',
                        action='store_true',
                        help='Set flag to apply measurement error mitigation')
    parser.add_argument('--mem_from_file',
                        action='store_true',
                        help='Set flag to take mem calibration data from file')
    parser.add_argument('--mem_dir',
                        default='./mem_calibration',
                        help='Directory for mem calibration files (to load or save)')
    parser.add_argument('--verbose_mem',
                        action='store_true',
                        help='Set flag to print mem overhead')
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
def load_backend(backend, args):
    hub      = args.hub
    group    = args.group
    project  = args.project
    ehningen = args.ehningen
    token    = args.token
    url      = args.url
    
    device = None
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
        
    return device


# %% prepare operator drivers
#
def prepare_operator_driver(operator_method, ansatz_unbound):
    
    # prepare list of drivers
    
    driver          = get_poisson_operator_driver(operator_method)
    operator_driver = driver(ansatz_unbound)
           
    return operator_driver


# %% prepare inner product drivers
#
def prepare_innerp_driver(innerp_method, ansatz_unbound, rhs_circ):
       
    # prepare list of drivers
    
    driver        = get_innerp_circuit_driver(innerp_method)
    innerp_driver = driver(ansatz_unbound, rhs_circ)
           
    return innerp_driver


# %% prepare cost drivers
#
def prepare_cost_driver(cost_method, op_driver, innerp_driver):
    
    # prepare list of drivers
    
    driver      = get_poisson_cost_driver(cost_method)
    cost_driver = driver(op_driver, innerp_driver)
            
    return cost_driver


# %% bind parameters to circuits for experiment
#
def bind_circuits(cost_driver, trans_unbound_, num_parameters, samples):
    
    # to list
    trans_unbound = trans_unbound_
    if type(trans_unbound_) is not list:
        trans_unbound = [trans_unbound_]
        
    # compute total number of circuits to run
    num_samples    = len(samples)

    # bind parameters
    trans_bound = []
    for m in range(num_samples):
        trans_bound += cost_driver.get_circuits_gradient(trans_unbound, samples[m][0:num_parameters])
    
    return trans_bound


# %% load experiments
#
def load_exps(args):
    
    # input
    
    exp_id    = args.exp_id
    store_dir = args.post_processed_dir
    target    = os.path.join(store_dir, 'exp_stats.csv')
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
    
    cost_name          = exp_id + '.pkl'
    cost_file          = os.path.join(input_dir, 'cost', cost_name)
    
    trans_unbound_name = exp_id + '.pkl'
    trans_unbound_file = os.path.join(input_dir, 'trans_unbound', trans_unbound_name)
    
    samples_name       = exp_id + '.pkl'
    samples_file       = os.path.join(input_dir, 'samples', samples_name)
    
    exp2jobs_name      = exp_id + '.txt'
    exp2jobs_file      = os.path.join(input_dir, 'experiment2jobs', exp2jobs_name)
    
    job_ids = []
    with open(exp2jobs_file) as file:
        while (job_id := file.readline().rstrip()):
            job_ids.append(job_id)
    
    results_files = []
    for job_id in job_ids:
        results_name = job_id + '.pkl'
        results_file = os.path.join(output_dir, results_name)
        results_files.append(results_file)
    
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
        
    cost_circuits = None
    with open(cost_file, 'rb') as file:
        cost_circuits = pickle.load(file)
        
    trans_unbound = None
    with open(trans_unbound_file, 'rb') as file:
        trans_unbound = pickle.load(file)
        
    samples = None
    with open(samples_file, 'rb') as file:
        samples = pickle.load(file)
        
    results = []
    for results_file in results_files:
        with open(results_file, 'rb') as file:
            results.append(pickle.load(file))
    
    for result in results:
        if type(result) == JobStatus:
            msg  = "This job is not done: " + result.value
            sys.exit(msg)
            
    # merge together
    results_merged = results[0]
    for j in range(1, len(results)):
        results_merged.results += results[j].results
        
    backend     = load_backend(args_input['backend'], args)
    archt       = args_input['archt']
    num_qubits  = args_input['num_qubits']
    num_layers  = args_input['num_layers']

    rhs_type        = args_input['rhs']
    operator_method = args_input['operator']
    innerp_method   = args_input['innerp']
    cost_method     = args_input['cost']
    
    shots       = args_input['shots']
    sabre       = args_input['transpile_sabre']
    mapomatic   = args_input['transpile_mapomatic']
    dd          = args_input['transpile_dd']
    mem         = args.mem
    error_type  = args.error_type
    
    # check if exists
    
    if exists:
        df         = pd.read_csv(target)
        exp_exists = (exp_id in df['exp_id'].values)
        subf       = df.loc[df['exp_id']==exp_id]
        mem_same   = (mem in subf['mem'].values)
        err_same   = (error_type in subf['error_type'].values)
            
        if exp_exists and mem_same and err_same and not force:
            msg="Experiment ID already saved, same MEM flag,\
 same error type, skipping..."
            sys.exit(msg)

    
    # bind parameters
    num_parameters = ansatz_unbound.num_parameters
    op_driver     = prepare_operator_driver(operator_method, ansatz_unbound)
    innerp_driver = prepare_innerp_driver(innerp_method, ansatz_unbound, rhs_circ)
    cost_driver   = prepare_cost_driver(cost_method, op_driver, innerp_driver)
    trans_bound   = bind_circuits(cost_driver, trans_unbound, num_parameters, samples)
    
    return (exp_id,
            job_ids,
            error_type,
            backend,
            archt,
            num_qubits,
            num_layers,
            rhs_type,
            operator_method,
            innerp_method,
            cost_method,
            shots,
            sabre,
            mapomatic,
            dd,
            mem,
            ansatz_unbound,
            rhs_circ,
            cost_circuits,
            trans_unbound,
            samples,
            trans_bound,
            results_merged)


# %% M3 error mitigation
#
def apply_mem(trans_bound, counts, backend, cals_file,
              from_file=False,
              verbose=False):
    
    overhead = False
    if verbose:
        overhead = True
        print("Applying measurement error mitigation")
        print("This may take some time...")
    
    mit     = mthree.M3Mitigation(backend)
    mapping = mthree.utils.final_measurement_mapping(trans_bound)
    
    if from_file:
        mit.cals_from_file(cals_file)
    else:
        mit.cals_from_system(mapping)
        mit.cals_to_file(cals_file)
        
    quasi = mit.apply_correction(counts, mapping,
                                 return_mitigation_overhead=overhead)
        
    return quasi


# %% compute (simulated) ideal gradient
#
def compute_exact_gradient(ansatz_unbound,
                           rhs_circ,
                           cost_method,
                           samples):
    
    num_samples = len(samples)
    exact_grad  = [None]*num_samples
    
    for m in range(num_samples):
        operator      = Poisson1dVQAOperator_exact(ansatz_unbound)
        innerp        = InnerProductStatevec(ansatz_unbound, rhs_circ)
        cost          = get_poisson_cost_driver(cost_method)(operator, innerp)
        exact_grad[m] = cost.gradient_exact(samples[m])
        
    return exact_grad


# %% compute estimated gradient
#
def compute_est_gradient(cost_driver, sampled_probs, samples):
    
    num_samples = len(samples)
    est_grad    = [None]*num_samples
    len_probs   = len(sampled_probs)
    num_circs   = int(len_probs/num_samples)
    
    for m in range(num_samples):
        idx_start   = m*num_circs
        idx_end     = (m+1)*num_circs
        theta       = samples[m][-1]        
        estimate    = cost_driver.gradient_from_measurements(sampled_probs[idx_start:idx_end]+[theta])
        est_grad[m] = estimate
        
    return est_grad


# %% compute errors
#
def compute_relative_errors(est_grad, exact_grad):
    
    num_samples = len(exact_grad)
    errors      = [None]*num_samples

    for m in range(num_samples):
        exact = exact_grad[m]
        est   = est_grad[m]
        errors[m] = np.linalg.norm(est-exact)/np.linalg.norm(exact)
        
    return errors


def compute_absolute_errors(est_grad, exact_grad):
    
    num_samples = len(exact_grad)
    errors      = [None]*num_samples

    for m in range(num_samples):
        exact     = exact_grad[m]
        est       = est_grad[m]
        errors[m] = np.linalg.norm(est-exact)
        
    return errors


def compute_cos_sim(est_grad, exact_grad):
    
    num_samples = len(exact_grad)
    errors      = [None]*num_samples

    for m in range(num_samples):
        exact     = exact_grad[m]
        est       = est_grad[m]
        errors[m] = np.inner(est, exact)/(np.linalg.norm(est)*np.linalg.norm(exact))
        
    return errors


# %% compute statistics
#
def statistics(errors):
    
    errors      = np.array(errors)
    errors      = errors[np.logical_not(np.isnan(errors))]
    
    mean        = np.mean(errors)
    std         = np.std(errors)
    mean_std    = np.array([mean, std])
    percentiles = [0., 1., 5., 10., 25., 50., 75., 90., 95., 99., 100.]
    stats       = np.percentile(errors, percentiles)
    stats       = np.concatenate((stats, mean_std))

    return stats


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
              job_ids,
              error_type,
              backend,
              archt,
              rhs_type,
              num_qubits,
              num_layers,
              num_parameters,
              num_samples,
              shots,
              depths,
              cnots,
              operator_method,
              innerp_method,
              cost_method,
              sabre,
              mapomatic,
              dd,
              mem,
              stats):
    
    # store directory
    store_dir = args.post_processed_dir
    target    = os.path.join(store_dir, 'exp_stats.csv')
    
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
    
    data = {'exp_id':[exp_id],
            'job_id': [job_ids],
            'error_type': [error_type],
            'backend': [backend],
            'archt': [archt],
            'rhs': [rhs_type],
            'qubits': [num_qubits],
            'layers': [num_layers],
            'parameters': [num_parameters],
            'samples': [num_samples],
            'shots': [shots],
            'depth': [depths],
            'cnots': [cnots],
            'operator': [operator_method],
            'innerp': [innerp_method],
            'cost': [cost_method],
            'rotations': [rotations],
            'entangler': [entangler],
            'mixer': [mixer],
            'driver': [driver],
            'driver_per': [driver_per],
            'initial_qaoa': [initial_qaoa],
            'sabre': [sabre],
            'mapomatic': [mapomatic],
            'DD': [dd],
            'mem': [mem],
            'mean': stats[11],
            'std': stats[12],
            'p=0': stats[0],
            'p=1': stats[1],
            'p=5': stats[2],
            'p=10': stats[3],
            'p=25': stats[4],
            'p=50': stats[5],
            'p=75': stats[6],
            'p=90': stats[7],
            'p=95': stats[8],
            'p=99': stats[9],
            'p=100': stats[10]}
    
    # store
    exists = os.path.exists(target)
    mode   = 'w'
    if exists:
        mode = 'a'
    
    df = pd.DataFrame.from_dict(data)
    df.to_csv(target, mode=mode, header=not exists, index=False, encoding='utf-8')
    
    
    return


# %% main
#
def main():
    args = parse_arguments()
    
    (exp_id,
    job_ids,
    error_type,
    backend,
    archt,
    num_qubits,
    num_layers,
    rhs_type,
    operator_method,
    innerp_method,
    cost_method,
    shots,
    sabre,
    mapomatic,
    dd,
    mem,
    ansatz_unbound,
    rhs_circ,
    cost_circuits,
    trans_unbound,
    samples,
    trans_bound,
    results) = load_exps(args)

    sampled_probs  = None
    verbose_mem    = args.verbose_mem
    num_parameters = ansatz_unbound.num_parameters
    num_samples    = len(samples)
  
    errors = None
    if mem:
        name          = backend.name() + ".json"
        cals_file     = os.path.join(args.mem_dir, name)
        sampled_probs = apply_mem(trans_bound,
                                  results.get_counts(),
                                  backend,
                                  cals_file,
                                  args.mem_from_file,
                                  verbose=verbose_mem)
    else:
        sampled_probs = counts2probs(results.get_counts())
    
    exact_grad  = compute_exact_gradient(ansatz_unbound,
                                         rhs_circ,
                                         cost_method,
                                         samples)
    operator    = get_poisson_operator_driver(operator_method)(ansatz_unbound)
    innerp      = get_innerp_circuit_driver(innerp_method)(ansatz_unbound,
                                                           rhs_circ)
    cost_driver = get_poisson_cost_driver(cost_method)(operator,
                                                       innerp)
    est_grad    = compute_est_gradient(cost_driver,
                                       sampled_probs,
                                       samples)
   
    compute_errors = None
    if error_type =='absolute':
        compute_errors = compute_absolute_errors
    elif error_type == 'relative':
        compute_errors = compute_relative_errors
    elif error_type == 'cosine_sim':
        compute_errors = compute_cos_sim
    else:
        msg = f"Error type ({error_type}) unknown, see help -h"
        raise ValueError(msg)
        
    errors         = compute_errors(est_grad, exact_grad)
    
    stats          = statistics(errors)
    depths         = to_depths(trans_unbound)
    cnots          = to_cnots(trans_unbound)
    
    save_data(args,
              exp_id,
              job_ids,
              error_type,
              backend,
              archt,
              rhs_type,
              num_qubits,
              num_layers,
              num_parameters,
              num_samples,
              shots,
              depths,
              cnots,
              operator_method,
              innerp_method,
              cost_method,
              sabre,
              mapomatic,
              dd,
              mem,
              stats)
    

# %% run script
#
if __name__ == '__main__':
    main()