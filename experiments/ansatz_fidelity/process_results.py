#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for processing results of experiments on IBM backends
to test the fidelity of ansatz circuits.
"""

# %% dependencies
#
import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd

from qiskit import Aer, IBMQ
from qiskit.providers import JobStatus
from qiskit.quantum_info import hellinger_fidelity
import mthree

from vqa_poisson_tools.utils.circuit2distribution import StatevecSimDist
from vqa_poisson_tools.utils.utils import counts2probs


# %% argument parser
#
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--job_id',
                        required=True,
                        metavar='id',
                        help='<Required> job id')
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
        device = Aer.get_backend('aer_simulator')
    
    elif ehningen:
        IBMQ.enable_account(token, url)
        provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        device   = provider.get_backend(backend)
    
    else:
    
        IBMQ.load_account()
        
        provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        device   = provider.get_backend(backend)
        
    return device


# %% M3 error mitigation
#
def apply_mem(trans_bound, counts, backend, cals_file, from_file=False,
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
    
    num_exps = len(quasi)
    probs    = [None]*num_exps
    
    for e in range(num_exps):
        probs[e] = quasi[e].nearest_probability_distribution()
        
    return probs


# %% compute (simulated) ideal probabilities
#
def compute_ideal_probs(ansatz_unbound, samples):
    
    num_circs   = len(ansatz_unbound)
    num_samples = len(samples[0])
    num_bound   = num_circs*num_samples
    ideal_probs = [None]*num_bound
    idx         = 0
    
    for q in range(num_circs):
        for m in range(num_samples):
            ansatz_bound     = ansatz_unbound[q].bind_parameters(samples[q][m])
            distr            = StatevecSimDist(ansatz_bound)
            ideal_probs[idx] = distr.simulate_distribution()
            idx             += 1
            
    return ideal_probs


# %% compute fidelities
#
def compute_fidelities(sampled_probs, ideal_probs):
    
    num_circs = len(sampled_probs)
    fidels    = [None]*num_circs

    for q in range(num_circs):
        fidels[q] = hellinger_fidelity(sampled_probs[q], ideal_probs[q])

    return fidels


# %% aggregate fidelities
#
def aggregate_fidelities(ansatz_unbound, num_samples, fidels):
    
    num_circs  = len(ansatz_unbound)
    aggregated = {}
    for q in range(num_circs):
        idx            = q*num_samples
        num_qubits     = ansatz_unbound[q].num_qubits
        num_parameters = ansatz_unbound[q].num_parameters
        
        fidelvec       = np.empty(num_samples)
        for m in range(num_samples):
            fidelvec[m] = fidels[idx+m]
        
        key             = (num_qubits, num_parameters)
        aggregated[key] = fidelvec

    return aggregated
    

# %% compute statistics
#
def statistics(aggregated):
    
    percentiles = [0., 1., 5., 10., 25., 50., 75., 90., 95., 99., 100.]
    stats       = {}
    
    for key, val in aggregated.items():
        stats[key] = np.percentile(val, percentiles)

    return stats


# %% bind parameters to circuits for experiment (required for error mitigation)
#
def bind_circuits(trans_unbound, samples, num_qubits, num_layers, num_samples):

    # compute total number of circuits to run
    num_circuits   = len(num_qubits)*len(num_layers)*num_samples

    # bind parameters
    
    trans_bound = [None]*num_circuits
    idx         = 0
    for q in range(len(trans_unbound)):
        for m in range(num_samples):
            trans_bound[idx] = trans_unbound[q].bind_parameters(samples[q][m])
            idx += 1
    
    return trans_bound


# %% load experiments
#
def load_exps(args):
    
    # check if exists
    
    job_id    = args.job_id
    store_dir = args.post_processed_dir
    target    = os.path.join(store_dir, 'exp_stats.csv')
    exists    = os.path.exists(target)
    mem       = args.mem
    
    if exists:
        df         = pd.read_csv(target)
        job_exists = (job_id in df['job_id'].values)
        subf       = df.loc[df['job_id']==job_id]
        mem_same   = (mem in subf['mem'].values)
        
        if job_exists and mem_same:
            msg="Job ID already saved, same MEM flag, skipping..."
            sys.exit(msg)
    
    # retrieve file names
    
    input_dir          = args.input_dir
    output_dir         = args.output_dir
  
    args_name          = job_id + '.json'
    args_file          = os.path.join(input_dir, 'args', args_name)
    
    ansatz_name        = job_id + '.pkl'
    ansatz_file        = os.path.join(input_dir, 'ansaetze', ansatz_name)
    
    trans_unbound_name = job_id + '.pkl'
    trans_unbound_file = os.path.join(input_dir, 'trans_unbound', trans_unbound_name)
    
    samples_name       = job_id + '.pkl'
    samples_file       = os.path.join(input_dir, 'samples', samples_name)
    
    counts_name        = job_id + '.pkl'
    counts_file        = os.path.join(output_dir, counts_name)
    
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
        
    counts = None
    with open(counts_file, 'rb') as file:
        counts = pickle.load(file)
        
    if type(counts) == JobStatus:
        msg  = "This job is not done: " + counts.value
        sys.exit(msg)
        
    backend     = load_backend(args_input['backend'], args)
    archt       = args_input['archt']
    num_qubits  = args_input['num_qubits']
    num_layers  = args_input['num_layers']
    shots       = args_input['shots']
    num_samples = args_input['num_samples']
    sabre       = args_input['transpile_sabre']
    mapomatic   = args_input['transpile_mapomatic']
    dd          = args_input['transpile_dd']
    mem         = args.mem
    
    # bind parameters
    trans_bound = bind_circuits(trans_unbound, samples, num_qubits, num_layers, num_samples)
    
    return (job_id,
            backend,
            archt,
            num_qubits,
            num_layers,
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
            counts)


# %% save data
#
def save_data(job_id,
              backend,
              archt,
              layers,
              depths,
              cnots,
              num_samples,
              shots,
              sabre,
              mapomatic,
              dd,
              mem,
              stats,
              args):
    
    # store directory
    store_dir = args.post_processed_dir
    target    = os.path.join(store_dir, 'exp_stats.csv')
    
    # input line
    input_dir = args.input_dir
    args_name = job_id + '.json'
    args_file = os.path.join(input_dir, 'args', args_name)
    
    args_input = None
    with open(args_file) as file:
        args_input = json.load(file)
    
    # create data
    num_circs = len(stats)
    
    list_jobs      = [job_id]*num_circs
    list_backends  = [backend]*num_circs
    list_archts    = [archt]*num_circs
    list_samples   = [num_samples]*num_circs
    list_shots     = [shots]*num_circs
    list_sabre     = [sabre]*num_circs
    list_mapomatic = [mapomatic]*num_circs
    list_dd        = [dd]*num_circs
    list_mem       = [mem]*num_circs
    
    list_rotations    = [args_input['rotations']]*num_circs
    list_entangler    = [args_input['entangler']]*num_circs
    list_mixer        = [args_input['mixer']]*num_circs
    list_driver       = [args_input['driver']]*num_circs
    list_driver_per   = [args_input['driver_per']]*num_circs
    list_initial_qaoa = [args_input['initial_qaoa']]*num_circs
    
    list_qubits = [None]*num_circs
    list_layers = [None]*num_circs
    list_params = [None]*num_circs
    list_depths = [None]*num_circs
    list_cnots  = [None]*num_circs
    list_0s     = [None]*num_circs
    list_1s     = [None]*num_circs
    list_5s     = [None]*num_circs
    list_10s    = [None]*num_circs
    list_25s    = [None]*num_circs
    list_50s    = [None]*num_circs
    list_75s    = [None]*num_circs
    list_90s    = [None]*num_circs
    list_95s    = [None]*num_circs
    list_99s    = [None]*num_circs
    list_100s   = [None]*num_circs 
    
    for i, (key, val) in enumerate(stats.items()):
        num_qubits     = key[0]
        num_parameters = key[1]
        num_layers     = layers[(num_qubits, num_parameters)]
        depth          = depths[(num_qubits, num_parameters)]
        num_cnots      = cnots[(num_qubits, num_parameters)]
        
        list_qubits[i] = num_qubits
        list_layers[i] = num_layers
        list_params[i] = num_parameters
        list_depths[i] = depth
        list_cnots[i]  = num_cnots
        list_0s[i]     = val[0]
        list_1s[i]     = val[1]
        list_5s[i]     = val[2]
        list_10s[i]    = val[3]
        list_25s[i]    = val[4]
        list_50s[i]    = val[5]
        list_75s[i]    = val[6]
        list_90s[i]    = val[7]
        list_95s[i]    = val[8]
        list_99s[i]    = val[9]
        list_100s[i]   = val[10]
    
    data = {'job_id': list_jobs,
            'backend': list_backends,
            'archt': list_archts,
            'qubits': list_qubits,
            'layers': list_layers,
            'parameters': list_params,
            'depth': list_depths,
            'cnots': list_cnots,
            'samples': list_samples,
            'shots': list_shots,
            'rotations': list_rotations,
            'entangler': list_entangler,
            'mixer': list_mixer,
            'driver': list_driver,
            'driver_per': list_driver_per,
            'initial_qaoa': list_initial_qaoa,
            'sabre': list_sabre,
            'mapomatic': list_mapomatic,
            'DD': list_dd,
            'mem': list_mem,
            'p=0': list_0s,
            'p=1': list_1s,
            'p=5': list_5s,
            'p=10': list_10s,
            'p=25': list_25s,
            'p=50': list_50s,
            'p=75': list_75s,
            'p=90': list_90s,
            'p=95': list_95s,
            'p=99': list_99s,
            'p=100': list_100s}
    
    # store
    exists = os.path.exists(target)
    mode   = 'w'
    if exists:
        mode = 'a'
    
    df = pd.DataFrame.from_dict(data)
    df.to_csv(target, mode=mode, header=not exists, index=False, encoding='utf-8')
    
    
    return


# %% match metadata
#
def to_num_layers(trans_unbound, num_qubits, num_layers):
    
    len_ns     = len(num_qubits)
    len_ls     = len(num_layers)
    
    mapping    = {}
    for n in range(len_ns):
        for l in range(len_ls):
            idx            = len_ls*n + l
            num_parameters = trans_unbound[idx].num_parameters
            key            = (num_qubits[n], num_parameters)
            value          = num_layers[l]
            mapping[key]   = value
            
    return mapping


def to_depth(trans_unbound, num_qubits, num_layers):
    
    len_ns     = len(num_qubits)
    len_ls     = len(num_layers)
    
    mapping    = {}
    for n in range(len_ns):
        for l in range(len_ls):
            idx            = len_ls*n + l
            num_parameters = trans_unbound[idx].num_parameters
            key            = (num_qubits[n], num_parameters)
            value          = trans_unbound[idx].depth()
            mapping[key]   = value
            
    return mapping


def to_num_cnots(trans_unbound, num_qubits, num_layers):
    
    len_ns     = len(num_qubits)
    len_ls     = len(num_layers)
    
    mapping    = {}
    for n in range(len_ns):
        for l in range(len_ls):
            idx            = len_ls*n + l
            num_parameters = trans_unbound[idx].num_parameters
            key            = (num_qubits[n], num_parameters)
            value          = 0
            
            if 'cx' in trans_unbound[idx].count_ops():
                value = trans_unbound[idx].count_ops()['cx']

            mapping[key]   = value
            
    return mapping


# %% main
#
def main():
    args = parse_arguments()
    
    (job_id,
    backend,
    archt,
    num_qubits,
    num_layers,
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
    counts) = load_exps(args)
    
    mem           = args.mem
    sampled_probs = None
    verbose_mem   = args.verbose_mem
    
    if mem:
        name          = backend.name() + ".json"
        cals_file     = os.path.join(args.mem_dir, name)
        sampled_probs = apply_mem(trans_bound, counts, backend,
                                  cals_file,
                                  args.mem_from_file,
                                  verbose=verbose_mem)
    else:
        sampled_probs = counts2probs(counts)
    
    ideal_probs = compute_ideal_probs(ansatz_unbound, samples)
    fidels      = compute_fidelities(sampled_probs, ideal_probs)
    aggregated  = aggregate_fidelities(ansatz_unbound, num_samples, fidels)
    stats       = statistics(aggregated)
     
    layers      = to_num_layers(trans_unbound, num_qubits, num_layers)
    depth       = to_depth(trans_unbound, num_qubits, num_layers)
    cnots       = to_num_cnots(trans_unbound, num_qubits, num_layers)
    
    save_data(job_id,
              backend.name(),
              archt,
              layers,
              depth,
              cnots,
              num_samples,
              shots,
              sabre,
              mapomatic,
              dd,
              mem,
              stats,
              args)
    

# %% run script
#
if __name__ == '__main__':
    main()