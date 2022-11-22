#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for running experiments on IBM backends
to test the fidelity of ansatz circuits.
"""

# %% dependencies
#
import os
import subprocess
import argparse
import pickle
import json
import pandas as pd

import datetime

from math import pi
import numpy as np

from qiskit import IBMQ, QuantumCircuit, Aer
from qiskit.providers import JobStatus

from vqa_poisson_tools.ansatz.two_local import get_ansatz_archt
from vqa_poisson_tools.poisson.rhs import hn
from qiskit_transpiler_tools.transpiler import TranspilerSabreMapomaticDD


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
    
    parser.add_argument('--run_mode',
                        required=True,
                        metavar='mode',
                        help='<Required> run mode: submit_job or save_output')
    parser.add_argument('--num_qubits',
                        type=int,
                        nargs='+',
                        default=[1],
                        metavar='n',
                        help='List of circuit sizes')
    parser.add_argument('--num_layers',
                        type=int,
                        nargs='+',
                        default=[1],
                        metavar='p',
                        help='List of ansatz layers (repetitions) for each circuit size')
    parser.add_argument('--num_samples',
                        type=int,
                        default=1,
                        metavar='m',
                        help='Number of parameter samples for each circuit')
    parser.add_argument('--seed_samples',
                        type=int,
                        default=None,
                        metavar='seed',
                        help='RNG seed for parameter samples')
    parser.add_argument('--seed_transpiler',
                        action='store_true',
                        help='Set flag to seed random (SABRE) transpilation')
    parser.add_argument('--transpile_sabre',
                        action='store_true',
                        help='Set flag to include SABRE for transpilation')
    parser.add_argument('--transpile_mapomatic',
                        action='store_true',
                        help='Set flag to include Mapomatic for transpilation')
    parser.add_argument('--transpile_dd',
                        action='store_true',
                        help='Set flag to include dynamic decoupling transpilation')
    parser.add_argument('--num_transpilations',
                        type=int,
                        default=1,
                        metavar='num_trans',
                        help='Number of times to repeat transpilation for SABRE')
    parser.add_argument('--shots',
                        type=int,
                        default=1000,
                        metavar='S',
                        help='Number of shots for each circuit')
    parser.add_argument('--archt',
                        default='linear',
                        help='Type of ansatz architechture')
    parser.add_argument('--rotations',
                        nargs='+',
                        default=['ry'],
                        metavar='RXYZ',
                        help='List of rotation gates for the HE ansatz')
    parser.add_argument('--entangler',
                        default='cx',
                        metavar='CU',
                        help='Entangler for the HE ansatz')
    parser.add_argument('--mixer',
                        default='rx',
                        metavar='RXYZ',
                        help='Mixer for the QAOA ansatz')
    parser.add_argument('--driver',
                        default='rzz',
                        metavar='RUU',
                        help='Driver for the QAOA ansatz')
    parser.add_argument('--driver_per',
                        default='ryy',
                        metavar='RUU',
                        help='Periodic driver component for the QAOA ansatz')
    parser.add_argument('--initial_qaoa',
                        default='hadamards',
                        metavar='initial',
                        help='Initial state for the QAOA ansatz: empty or hadamards')
    parser.add_argument('--verbose',
                        nargs='+',
                        default=['all'],
                        help='List of verbose flags')
    parser.add_argument('--backend',
                        required=True,
                        metavar='device',
                        help='<Required> name of IBM backend')
    parser.add_argument('--job_id',
                        metavar='id',
                        help='Job ID for retrieving output')
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
                        help='Directory for saving inputs')
    parser.add_argument('--output_dir',
                        default='./outputs',
                        metavar='dir',
                        help='Directory for saving outputs')
    parser.add_argument('--log_dir',
                        default='./logs',
                        metavar='dir',
                        help='Directory for saving logs')
    parser.add_argument('--bash_dir',
                        default='./bash',
                        metavar='dir',
                        help='Directory for saving bash scripts input')
    parser.add_argument('--save_value',
                        type=int,
                        default=0,
                        metavar='value',
                        help='Value to print next to backend id list,\
                            helpful for the save_output run mode,\
                            value=1 means output will be retrieved for this backend and job_id')
    parser.add_argument('--git_repo_dir',
                        required=True,
                        metavar='dir',
                        help='<Required> directory for root git repository')
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


# %% submit circuits
#
def submit_job(device, trans_bound, args):
    shots         = args.shots
    experiment_id = 'ansatz_fidelity'
    simulator     = device.configuration().simulator
    
    max_exp = 0
    if not simulator:
        max_exp = device.configuration().max_experiments
    
    num_circs     = len(trans_bound)    
    if num_circs > max_exp and not simulator:
        msg  = "Number of circuits (" + str(num_circs) + ")" \
               + " too large for backend (max=" + str(max_exp) + ")"
        raise ValueError(msg)
            
    
    job = device.run(trans_bound, experiment_id=experiment_id, shots=shots)
    
    return job
    

# %% save experiment input data
#
def save_input(job, ansatz_unbound, trans_unbound, samples, args):
    job_id         = job.job_id()
    file_args      = job_id + '.json'
    file_ansatz    = job_id + '.pkl'
    file_trans     = job_id + '.pkl'
    file_samples   = job_id + '.pkl'
    
    target_args    = os.path.join(args.input_dir, 'args', file_args)
    target_ansatz  = os.path.join(args.input_dir, 'ansaetze', file_ansatz)
    target_trans   = os.path.join(args.input_dir, 'trans_unbound', file_trans)
    target_samples = os.path.join(args.input_dir, 'samples', file_samples)
    
    # git info
    repo_path = args.git_repo_dir
    git_branch = subprocess.check_output(
        ["git", "--git-dir", repo_path, "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('UTF-8')
    git_commit_short_hash = subprocess.check_output(
        ["git", "--git-dir", repo_path, "describe", "--always"]).strip().decode('UTF-8')

    with open(target_args, 'w') as file:
        args_dict = {k: args.__dict__[k] for k in sorted(args.__dict__.keys())}
        args_dict['git_branch']            = git_branch
        args_dict['git_commit_short_hash'] = git_commit_short_hash
        json.dump(args_dict, file)

    with open(target_ansatz, 'wb') as file:
        pickle.dump(ansatz_unbound, file)

    with open(target_trans, 'wb') as file:
        pickle.dump(trans_unbound, file)
        
    with open(target_samples, 'wb') as file:
        pickle.dump(samples, file)
        
        
# %% save log data
#
def save_log(job, args):
    target = os.path.join(args.log_dir, 'log.csv')
    
    creation_date = None
    if args.backend == 'simulator':
        time = datetime.datetime.now().strftime("%H:%M:%S")
        date = datetime.date.today().strftime("%d/%m/%Y")
        creation_date = date + ', ' + time
    else:
        creation_date = job.creation_date()
        time          = creation_date.strftime("%H:%M:%S")
        date          = creation_date.strftime("%d/%m/%Y")
        creation_date = date + ', ' + time
        
    # git info
    repo_path         = args.git_repo_dir
    git_diff          = subprocess.check_output(
                        ["git", "--git-dir", repo_path, "diff"]).decode('UTF-8')
    job_id            = job.job_id()
    file_git_diff     = job_id + '.patch'
    git_diff_filepath = os.path.join(args.log_dir, 'git_diff', file_git_diff)
    
    if len(git_diff) > 0:
        with open(git_diff_filepath, 'w') as f:
            f.writelines(git_diff)

    data = {'job_id': [job_id],
            'job_time': [creation_date],
            'backend': [args.backend],
            'shots': [args.shots],
            'archt': [args.archt],
            'qubits': [args.num_qubits],
            'layers': [args.num_layers],
            'samples': [args.num_samples],
            'rotations': [args.rotations],
            'entangler': [args.entangler],
            'mixer': [args.mixer],
            'driver': [args.driver],
            'driver_per': [args.driver_per],
            'initial_qaoa': [args.initial_qaoa],
            'sabre': [args.transpile_sabre],
            'mapomatic': [args.transpile_mapomatic],
            'DD': [args.transpile_dd]}
    
    exists = os.path.exists(target)
    mode   = 'w'
    if exists:
        mode = 'a'
    
    df = pd.DataFrame.from_dict(data)
    df.to_csv(target, mode=mode, header=not exists, index=False, encoding='utf-8')
    
    # for easier running of bash scripts
    target         = os.path.join(args.bash_dir, 'save_out_backend_job_list.txt')
    bash_save_data = args.backend  + " " + job_id + " " + str(args.save_value) + "\n"
    with open(target, "a") as file:
        file.write(bash_save_data)


# %% save output
#
def save_output(job, args):
    job_id = job.job_id()
    status = job.status()
    
    results = None
    if status != JobStatus.DONE:
        print("Job not done")
        print(status.value)
        print("Saving job status instead...")
        results = status
    else:
        results = job.result().get_counts()
    
    file   = job_id + '.pkl'
    target = os.path.join(args.output_dir, file)

    with open(target, 'wb') as file:
        pickle.dump(results, file)

    
# %% prepare circuits for experiment
#
def prepare_circuits(args):

    # input parameters
    
    num_qubits          = args.num_qubits
    num_layers          = args.num_layers
    archt               = args.archt
    rotations           = args.rotations
    entangler           = args.entangler
    mixer               = args.mixer
    driver              = args.driver
    driver_per          = args.driver_per 
    initial_qaoa        = args.initial_qaoa
    verbose             = args.verbose

    # verbose settings
    
    verbose_all       = False
    verbose_ansatz    = False
    
    if 'all' in verbose:
        verbose_all=True
    if 'ansatz' in verbose:
        verbose_ansatz=True
    
    # initial state for QAOA
    
    num_widths    = len(num_qubits)
    initial_state = [None]*num_widths
    if initial_qaoa == init_states[0]:
        initial_state = [QuantumCircuit(num_qubits[n]) for n in range(num_widths)]
    elif initial_qaoa == init_states[1]:
        initial_state = [hn(num_qubits[n]) for n in range(num_widths)]
    else:
        message  = "Type of initial state not implemented for this experiment. Available: "
        message += str(init_states)
        raise ValueError(message)

    # build ansatz circuit
    
    builder = get_ansatz_archt(archt)
    
    # different inputs
    
    num_depths     = len(num_layers)
    ansatz_unbound = [None]*num_widths*num_depths
    idx            = 0
    for n in range(num_widths):
        for l in range(num_depths):
            if archt in case_hea:
                
                ansatz_unbound[idx] = builder(num_qubits[n], num_layers[l],
                                             rotation_gates=rotations,
                                             entangler=entangler,
                                             prefix='θ',
                                             name='ansatz')
                        
            elif archt in case_qaoa:
                
                ansatz_unbound[idx] = builder(num_qubits[n], num_layers[l],
                                             initial_state=initial_state[n],
                                             mixer=mixer,
                                             driver=driver,
                                             prefix='θ',
                                             name='ansatz')
                
            elif archt in case_qaoa_per:
                
                ansatz_unbound[idx] = builder(num_qubits[n], num_layers[l],
                                             initial_state=initial_state[n],
                                             mixer=mixer,
                                             driver=driver,
                                             driver_per=driver_per,
                                             prefix='θ',
                                             name='ansatz')
                
            else:
                message  = "Ansatz architechture not considered in this experiment. Available: "
                cases    = case_hea + case_qaoa + case_qaoa_per
                message += str(cases)
                raise ValueError(message)
            
            idx += 1

    if verbose_all or verbose_ansatz:
        print("----------- Ansatz architechture example -----------")
        print(ansatz_unbound[0])
        print("\n")
        print("Total number of unbound circuits : ", num_widths*num_depths)
        print("----------- -------------------- ------- -----------")
    
    return ansatz_unbound


# %% add measurement instructions
#
def add_measurements(ansatz_unbound):
    
    num_circs     = len(ansatz_unbound)
    ansatz_w_meas = [None]*num_circs
    for q in range(num_circs):
        ansatz_w_meas[q] = ansatz_unbound[q].measure_all(inplace=False)
        
    return ansatz_w_meas


# %% transpile circuits for backend
#
def transpile(device, ansatz_unbound, args):
    # input parameters
    
    seed_transpile_flag = args.seed_transpiler
    transpile_sabre     = args.transpile_sabre
    transpile_mapomatic = args.transpile_mapomatic
    transpile_dd        = args.transpile_dd
    num_transpilations  = args.num_transpilations
    verbose             = args.verbose
    
    # verbose settings
    
    verbose_all       = False
    verbose_transpile = False
    
    if 'all' in verbose:
        verbose_all=True
    if 'transpile' in verbose:
        verbose_transpile=True
    
    # transpiler options
    
    seed_transpiler = [None]*num_transpilations
    if seed_transpile_flag:
        seed_transpiler = list(range(num_transpilations))

    transpile_opts  = {}
    simulator = (args.backend == 'simulator')
    if transpile_sabre and not simulator:
        transpile_opts['optimization_level'] = 3
        transpile_opts['num_transpilations'] = num_transpilations
        transpile_opts['seed_transpiler']    = seed_transpiler
    else:
        transpile_opts['optimization_level'] = 0
        transpile_opts['num_transpilations'] = 1
        transpile_opts['seed_transpiler']    = seed_transpiler
        
    if transpile_mapomatic and not simulator:
        transpile_opts['apply_mapomatic'] = True
    else:
        transpile_opts['apply_mapomatic'] = False
        
    if transpile_dd and not simulator:
        transpile_opts['apply_dd'] = True
    else:
        transpile_opts['apply_dd'] = False
        
    # transpile
    
    transpiler    = TranspilerSabreMapomaticDD(device, **transpile_opts)
    
    if verbose_all or verbose_transpile:
        print("Transpiling...")
        
    trans_unbound = transpiler.transpile(ansatz_unbound)
    
    if verbose_all or verbose_transpile:
        print("Transpilation done.")
    
    if verbose_all or verbose_transpile:
        print("----------- Post transpilation -----------")
        for i in range(len(trans_unbound)):
            print("Circuit number             : ", i)
            print("Number of (compute) qubits : ", ansatz_unbound[i].num_qubits)
            print("Number of parameters       : ", ansatz_unbound[i].num_parameters)
            print("Target backend             : ", device.name())
            print("Depth                      : ", trans_unbound[i].depth())
            
            num_cnots = 0
            if 'cx' in trans_unbound[i].count_ops():
                num_cnots = trans_unbound[i].count_ops()['cx']
            print("Number of CNOTs            : ", num_cnots)
            print("---------------")
        print("----------- -------------------- -----------")
    
    return trans_unbound


# %% prepare samples for experiment
#
def generate_samples(trans_unbound, args):

    # input parameters

    num_samples         = args.num_samples
    seed_samples        = args.seed_samples
   
    # sample parameters
    
    rng     = np.random.default_rng(seed=seed_samples)
    samples = []
    for q in range(len(trans_unbound)):
        num_parameters = trans_unbound[q].num_parameters
        samples_q      = []
        for m in range(num_samples):
            samples_q.append(rng.uniform(low=0., high=4.*pi,  size=num_parameters))
        samples.append(samples_q)
    
    return samples


# %% bind parameters to circuits for experiment
#
def bind_circuits(trans_unbound, samples, args):

    # input parameters
    
    num_qubits          = args.num_qubits
    num_layers          = args.num_layers
    num_samples         = args.num_samples

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


# %% main
#
def main():
    args = parse_arguments()
    mode = args.run_mode
    sim  = (args.backend=='simulator')

    if mode=='submit_job':
        device         = load_backend(args)
        ansatz_unbound = prepare_circuits(args)
        ansatz_w_meas  = add_measurements(ansatz_unbound)
        trans_unbound  = transpile(device, ansatz_w_meas, args)
        samples        = generate_samples(trans_unbound, args)
        trans_bound    = bind_circuits(trans_unbound, samples, args)
        
        job            = submit_job(device, trans_bound, args)
        
        save_input(job, ansatz_unbound, trans_unbound, samples, args)
        save_log(job, args)
        
        if sim:
            save_output(job, args)
        
    elif mode=='save_output':
        if sim:
            print("Backend is a simulator, job should be already saved")
            print("Exiting...")
            return

        job_id = args.job_id
        device = load_backend(args)
        job    = device.retrieve_job(job_id)
        save_output(job, args)
    
    else:
        msg = "Unknown run mode (see help -h)."
        raise ValueError(msg)
        

# %% run script
#
if __name__ == '__main__':
    main()