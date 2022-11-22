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
from qiskit.providers.aer import AerSimulator
from qiskit.providers import JobStatus
import mthree

from vqa_poisson_tools.poisson.operator import Poisson1dVQAOperator_exact
from vqa_poisson_tools.poisson.operator import get_poisson_operator_driver
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
    parser.add_argument('--relative',
                        action='store_true',
                        help='Set flag if relative error is to be computed')
    parser.add_argument('--constant',
                        action='store_true',
                        help='Set flag to set estimated expectation to 2')
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


# %% bind parameters to circuits for experiment
#
def bind_circuits(trans_unbound_, samples):
    
    # to list
    trans_unbound = trans_unbound_
    if type(trans_unbound_) is not list:
        trans_unbound = [trans_unbound_]

    # compute total number of circuits to run
    num_samples    = len(samples)
    num_unbound    = len(trans_unbound)
    num_circuits   = num_samples*num_unbound

    # bind parameters
    trans_bound = [None]*num_circuits
    idx         = 0
    for m in range(num_samples):
        for q in range(num_unbound):
            trans_bound[idx] = trans_unbound[q].bind_parameters(samples[m])
            idx             += 1
    
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
        
    backend     = load_backend(args_input['backend'], args)
    archt       = args_input['archt']
    num_qubits  = args_input['num_qubits']
    num_layers  = args_input['num_layers']

    operator_method = args_input['operator']
    
    shots       = args_input['shots']
    num_samples = args_input['num_samples']
    sabre       = args_input['transpile_sabre']
    mapomatic   = args_input['transpile_mapomatic']
    dd          = args_input['transpile_dd']
    mem         = args.mem
    relative    = args.relative
    constant    = args.constant
    
    # check if exists
    
    if exists:
        df         = pd.read_csv(target)
        exp_exists = (exp_id in df['exp_id'].values)
        subf       = df.loc[df['exp_id']==exp_id]
        mem_same   = (mem in subf['mem'].values)
        err_same   = (relative in subf['relative'].values)
        
        const_same = (not constant or (constant and 'constant' in subf['operator'].values))
        
        if exp_exists and mem_same and err_same and const_same and not force:
            msg="Experiment ID already saved, same MEM flag,\
 same relative flag, same constant flag, skipping..."
            sys.exit(msg)

    
    # bind parameters
    trans_bound = bind_circuits(trans_unbound, samples)
    
    return (exp_id,
            job_ids,
            backend,
            archt,
            num_qubits,
            num_layers,
            operator_method,
            shots,
            num_samples,
            sabre,
            mapomatic,
            dd,
            mem,
            ansatz_unbound,
            trans_unbound,
            samples,
            trans_bound,
            results)


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


# %% compute (simulated) ideal operator expectations
#
def compute_exact_operator_expecs(ansatz_unbound, samples):
    
    num_samples = len(samples)
    exact_ops   = [None]*num_samples
    
    for m in range(num_samples):
        ansatz_bound = ansatz_unbound.bind_parameters(samples[m])
        operator     = Poisson1dVQAOperator_exact(ansatz_bound)
        exact_ops[m] = operator.energy()
        
        
    return exact_ops


# %% compute estimated operator expectations
#
def compute_est_operator_expecs(operator_driver,
                                num_samples,
                                sampled_probs):
    
    est_ops   = [None]*num_samples
    len_probs = len(sampled_probs)
    num_circs = len(operator_driver.get_circuits())
    num_total = num_samples*num_circs
    if len_probs != num_total:
        msg = f"Length of counts ({len_probs}) and total circuits ({num_total}) do not match"
        raise ValueError(msg)
    
    for m in range(num_samples):
        idx_start  = m*num_circs
        idx_end    = (m+1)*num_circs
        estimate   = operator_driver.energy_from_measurements(sampled_probs[idx_start:idx_end])
        est_ops[m] = estimate
        
        
    return est_ops


# %% set operator expectation to 2
#
def compute_const_operator_expecs(num_samples):
    
    est_ops = [None]*num_samples
    for m in range(num_samples):
        est_ops[m] = 2.
        
    return est_ops


# %% compute errors
#
def compute_relative_errors(est_ops, exact_ops):
    
    num_samples = len(exact_ops)
    errors      = [None]*num_samples

    for m in range(num_samples):
        exact = exact_ops[m]
        est   = est_ops[m]
        errors[m] = abs(est-exact)/abs(exact)
        
    return errors


def compute_absolute_errors(est_ops, exact_ops):
    
    num_samples = len(exact_ops)
    errors      = [None]*num_samples

    for m in range(num_samples):
        exact     = exact_ops[m]
        est       = est_ops[m]
        errors[m] = abs(est-exact)
        
    return errors


# %% compute statistics
#
def statistics(errors):
    
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
              backend,
              archt,
              num_qubits,
              num_layers,
              num_parameters,
              shots,
              depths,
              cnots,
              operator_method,
              num_samples,
              sabre,
              mapomatic,
              dd,
              relative,
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
            'backend': [backend],
            'archt': [archt],
            'qubits': [num_qubits],
            'layers': [num_layers],
            'parameters': [num_parameters],
            'depth': [depths],
            'cnots': [cnots],
            'operator': [operator_method],
            'samples': [num_samples],
            'rotations': [rotations],
            'entangler': [entangler],
            'mixer': [mixer],
            'driver': [driver],
            'driver_per': [driver_per],
            'initial_qaoa': [initial_qaoa],
            'sabre': [sabre],
            'mapomatic': [mapomatic],
            'DD': [dd],
            'relative': [relative],
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
            'p=100': stats[10],
            'shots': [shots]}
    
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
    backend,
    archt,
    num_qubits,
    num_layers,
    operator_method,
    shots,
    num_samples,
    sabre,
    mapomatic,
    dd,
    mem,
    ansatz_unbound,
    trans_unbound,
    samples,
    trans_bound,
    results) = load_exps(args)
    
    mem           = args.mem
    relative      = args.relative
    constant      = args.constant
    sampled_probs = None
    verbose_mem   = args.verbose_mem
  
    if constant:
        mem             = False
        operator_method = 'constant'

    if mem:
        name          = backend.name() + ".json"
        cals_file     = os.path.join(args.mem_dir, name)
        sampled_probs = apply_mem(trans_bound,
                                  results[0].get_counts(),
                                  backend,
                                  cals_file,
                                  args.mem_from_file,
                                  verbose=verbose_mem)
    else:
        sampled_probs = counts2probs(results[0].get_counts())
    
    exact_ops       = compute_exact_operator_expecs(ansatz_unbound, samples)
    est_ops         = None
    operator_driver = None
    if constant:
        est_ops = compute_const_operator_expecs(num_samples)
    else:
        operator_driver = get_poisson_operator_driver(operator_method)
        operator_driver = operator_driver(ansatz_unbound)
        est_ops         = compute_est_operator_expecs(operator_driver,
                                                      num_samples,
                                                      sampled_probs)
   
    compute_errors = compute_absolute_errors
    if relative:
        compute_errors = compute_relative_errors
        
    errors         = compute_errors(est_ops, exact_ops)
    stats          = statistics(errors)
    num_parameters = ansatz_unbound.num_parameters
    depths         = to_depths(trans_unbound)
    cnots          = to_cnots(trans_unbound)
    
    save_data(args,
              exp_id,
              job_ids,
              backend.name(),
              archt,
              num_qubits,
              num_layers,
              num_parameters,
              shots,
              depths,
              cnots,
              operator_method,
              num_samples,
              sabre,
              mapomatic,
              dd,
              relative,
              mem,
              stats)
    

# %% run script
#
if __name__ == '__main__':
    main()