#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for running experiments on IBM backends
to test the accuracy of inner products.
"""

# %% dependencies
#
import os
import uuid
import warnings
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
from qiskit.utils import QuantumInstance
from qiskit.algorithms import (EstimationProblem,
                               MaximumLikelihoodAmplitudeEstimation,
                               IterativeAmplitudeEstimation,
                               FasterAmplitudeEstimation)

from vqa_poisson_tools.ansatz.two_local import get_ansatz_archt
from vqa_poisson_tools.poisson.rhs import get_rhs, hn
from vqa_poisson_tools.poisson.innerp import get_innerp_circuit_driver
from qiskit_transpiler_tools.transpiler import TranspilerSabreMapomaticDD


# %% allowed architechture inputs
#
case_hea      = ['linear', 'linear_periodic',
                 'linear_alternating', 'linear_alternating_periodic',
                 'linear_alternating_periodic_bidirectional']
case_qaoa     = ['qaoa']
case_qaoa_per = ['qaoa_periodic']
init_states   = ['empty', 'hadamards']


# %% allowed QAE methods for this experiment
#
QAES = ['mlae', 'iae', 'fae']


# %% argument parser
#
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_mode',
                        required=True,
                        metavar='mode',
                        help='<Required> run mode: submit_experiment or save_output')
    parser.add_argument('--num_qubits',
                        type=int,
                        default=1,
                        metavar='n',
                        help='Number of qubits')
    parser.add_argument('--num_layers',
                        type=int,
                        default=0,
                        metavar='p',
                        help='Number ansatz layers (repetitions)')
    parser.add_argument('--archt',
                        default='linear',
                        metavar='type',
                        help='Type of ansatz architechture')
    parser.add_argument('--rhs',
                        default='hn',
                        metavar='type',
                        help='Type of right-hand-side')
    parser.add_argument('--innerp',
                        default='HTest',
                        metavar='type',
                        help='Type of inner product circuit')
    parser.add_argument('--qae',
                        action='store_true',
                        help='Set flag to apply QAE')
    parser.add_argument('--qae_method',
                        default='iae',
                        metavar='method',
                        help='Specify QAE method')
    parser.add_argument('--max_power',
                        type=int,
                        default=1,
                        metavar='k',
                        help='Specify max power for QAE circuits')
    parser.add_argument('--iae_epsilon_target',
                        type=float,
                        default=0.1,
                        metavar='eps',
                        help='Target epsilon for iterative QAE')
    parser.add_argument('--iae_alpha',
                        type=float,
                        default=0.05,
                        metavar='alpha',
                        help='Alpha for iterative QAE')
    parser.add_argument('--fae_delta',
                        type=float,
                        default=0.1,
                        metavar='delta',
                        help='Delta for fast QAE')
    parser.add_argument('--fae_rescale',
                        action='store_true',
                        help='Set flag for FAE rescale')
    parser.add_argument('--num_samples',
                        type=int,
                        default=1,
                        metavar='m',
                        help='Number of parameter samples')
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
                        help='Job ID for saving output')
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


# %% prepare ansatz circuits for experiment
#
def prepare_ansatz_circuit(args):

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
    
    initial_state = None
    if initial_qaoa == init_states[0]:
        initial_state = QuantumCircuit(num_qubits)
    elif initial_qaoa == init_states[1]:
        initial_state = hn(num_qubits)
    else:
        message  = "Type of initial state not implemented for this experiment. Available: "
        message += str(init_states)
        raise ValueError(message)

    # build ansatz circuit
    
    builder = get_ansatz_archt(archt)
    
    # different inputs
    
    ansatz_unbound = None
    if archt in case_hea:
        
        ansatz_unbound = builder(num_qubits, num_layers,
                                 rotation_gates=rotations,
                                 entangler=entangler,
                                 prefix='θ',
                                 name='ansatz')
                
    elif archt in case_qaoa:
        
        ansatz_unbound = builder(num_qubits, num_layers,
                                 initial_state=initial_state,
                                 mixer=mixer,
                                 driver=driver,
                                 prefix='θ',
                                 name='ansatz')
        
    elif archt in case_qaoa_per:
        
        ansatz_unbound = builder(num_qubits, num_layers,
                                 initial_state=initial_state,
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

    if verbose_all or verbose_ansatz:
        print("----------- Ansatz architechture -------------------")
        print(ansatz_unbound)
        print("\n")
        print("Total number of unbound circuits : ", 1)
        print("----------- -------------------- ------- -----------")
    
    return ansatz_unbound


# %% prepare rhs circuits
#
def prepare_rhs_circuit(args):
    
    # input parameters
    
    num_qubits = args.num_qubits
    rhs_type   = args.rhs
    verbose    = args.verbose

    # verbose settings
    
    verbose_all = False
    verbose_rhs = False
    
    if 'all' in verbose:
        verbose_all = True
    if 'rhs' in verbose:
        verbose_rhs = True
    
    # build circuits
    
    rhs_circ = None
    driver   = get_rhs(rhs_type)
    rhs_circ = driver(num_qubits)
            
    if verbose_all or verbose_rhs:
        print("----------- Rhs ------------------------------------")
        print(rhs_circ)
        print("\n")
        print("Total number of rhs circuits : ", 1)
        print("----------- -------------------- ------- -----------")
    
    return rhs_circ


# %% prepare inner product drivers
#
def prepare_innerp_driver(args, ansatz_unbound, rhs_circ):
    
    # input parameters
    
    innerp_method = args.innerp
    verbose       = args.verbose
    
    # verbose settings
    
    verbose_all            = False
    verbose_innerp_driver = False
    
    if 'all' in verbose:
        verbose_all = True
    if 'innerp_driver' in verbose:
        verbose_innerp_driver = True
    
    # prepare list of drivers
    
    driver        = get_innerp_circuit_driver(innerp_method)
    innerp_driver = driver(ansatz_unbound, rhs_circ)

    if verbose_all or verbose_innerp_driver:
        print("----------- Innerp driver --------------------------")
        print("Total number of drivers : ", 1)
        print("----------- -------------------- ------- -----------")
            
    return innerp_driver


# %% prepare inner product circuits
#
def prepare_innerp_circuit(args, innerp_driver):
    
    # input parameters
    
    verbose = args.verbose
    
    # verbose settings
    
    verbose_all            = False
    verbose_innerp_circuit = False
    
    if 'all' in verbose:
        verbose_all = True
    if 'innerp_circuit' in verbose:
        verbose_innerp_circuit = True

    # dot circuit
    
    innerp_circ = innerp_driver.dot_circuit(w_meas=True)
        
    # print

    if verbose_all or verbose_innerp_circuit:
        print("----------- Inner product circuit ------------------")
        print(innerp_circ)
        print("\n")
        print("Total number of innerp circuits : ", 1)
        print("----------- -------------------- ------- -----------")
    
    return innerp_circ


# %% prepare inner product QAE circuits
#
def prepare_innerp_qaes(args,
                        device,
                        innerp_driver):
 
    # input parameters
    
    qae_method          = args.qae_method
    max_power           = args.max_power
    iae_epsilon_target  = args.iae_epsilon_target
    iae_alpha           = args.iae_alpha
    fae_delta           = args.fae_delta
    fae_rescale         = args.fae_rescale
    shots               = args.shots
    verbose             = args.verbose
    
    # verbose settings
    
    verbose_all         = False
    verbose_innerp_qaes = False
    
    if 'all' in verbose:
        verbose_all = True
    if 'innerp_qaes' in verbose:
        verbose_innerp_qaes = True
        
    # check method
    
    if qae_method not in QAES:
        msg  = "QAE method not included in this experiment. Available: "
        msg += str(QAES)
        
        raise ValueError(msg)
        
    # QAE circuits
    
    qi        = QuantumInstance(device, shots=shots)
    new_shots = [shots]
    
    A             = innerp_driver.state_preparation()
    Q             = innerp_driver.Grover()
    oq            = innerp_driver.objective_qubits()
    is_good_state = innerp_driver.is_good_state
    problem       = EstimationProblem(state_preparation=A,
                                      grover_operator=Q,
                                      objective_qubits=oq,
                                      is_good_state=is_good_state)
        
    qae_circs = []
    if qae_method == 'mlae':
        
        mlae      = MaximumLikelihoodAmplitudeEstimation(evaluation_schedule=max_power-1,
                                                         quantum_instance=qi)
        qae_circs = mlae.construct_circuits(problem, measurement=True)

    else:
        estimator = None
        ks        = None
        if qae_method == 'iae':
            estimator = IterativeAmplitudeEstimation(epsilon_target=iae_epsilon_target,
                                                     alpha=iae_alpha,
                                                     quantum_instance=qi)
            ks = list(range(max_power))
            
        elif qae_method == 'fae':
            estimator = FasterAmplitudeEstimation(delta=fae_delta,
                                                  maxiter=max_power-1,
                                                  rescale=fae_rescale,
                                                  quantum_instance=qi)
            if fae_rescale:
                problem = problem.rescale(0.25)
            new_shots = estimator._shots
            ks = list(range(2**(max_power-1)))
        
        for k in ks:        
            QkA = estimator.construct_circuit(problem, k, measurement=True)
            qae_circs.append(QkA)
        
    num_circs = len(qae_circs)
    if verbose_all or verbose_innerp_qaes:
        print("-------- Inner product QAE circuit example ---------")
        print(qae_circs[0])
        print("\n")
        print("Total number of innerp QAE circuits: ", num_circs)
        print("----------- -------------------- ------- -----------")
        
    return qae_circs, new_shots
    

# %% transpile circuits for backend
#
def transpile(args, device, circuits):
    
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
        
    trans_unbound = transpiler.transpile(circuits)
    
    if verbose_all or verbose_transpile:
        print("Transpilation done.")
    
    circuits_      = circuits    
    trans_unbound_ = trans_unbound
    if type(circuits) is not list:
        circuits_      = [circuits]
        trans_unbound_ = [trans_unbound]
    
    if verbose_all or verbose_transpile:
        print("----------- Post transpilation -----------")
        for i in range(len(trans_unbound_)):
            print("Circuit number             : ", i)
            print("Number of (compute) qubits : ", circuits_[i].num_qubits)
            print("Number of parameters       : ", circuits_[i].num_parameters)
            print("Target backend             : ", device.name())
            print("Depth                      : ", trans_unbound_[i].depth())
            
            num_cnots = 0
            if 'cx' in trans_unbound_[i].count_ops():
                num_cnots = trans_unbound_[i].count_ops()['cx']
            print("Number of CNOTs            : ", num_cnots)
            print("---------------")
        print("----------- -------------------- -----------")
    
    return trans_unbound


# %% prepare samples for experiment
#
def generate_samples(args, num_parameters):

    # input parameters

    num_samples         = args.num_samples
    seed_samples        = args.seed_samples
   
    # sample parameters
    
    rng     = np.random.default_rng(seed=seed_samples)
    samples = []
    for m in range(num_samples):
        samples.append(rng.uniform(low=0., high=4.*pi,  size=num_parameters))
    
    return samples


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
            idx += 1
    
    return trans_bound


# %% submit circuits
#
def submit_jobs(device, trans_bound, new_shots):
    
    # input parameters
    
    experiment_id = 'inner_product'
    simulator     = device.configuration().simulator
    
    # max number of experiments
    
    max_exp = 0
    if not simulator:
        max_exp = device.configuration().max_experiments
    
    num_circs = len(trans_bound)    
    if num_circs > max_exp and not simulator:
        msg  = "Number of circuits (" + str(num_circs) + ")" \
               + " too large for backend (max=" + str(max_exp) + ")"
        raise ValueError(msg)
    
    # submit jobs        
    jobs = []
    for s in new_shots:
        jobs.append(device.run(trans_bound, experiment_id=experiment_id, shots=s))
    
    return jobs
    

# %% save experiment input data
#
def save_input(args,
               jobs,
               ansatz_unbound,
               rhs_circ,
               innerp_circs,
               trans_unbound,
               samples,
               new_shots): # add this to the cmd line arguments
    
    # input parameters
    job_ids  = [job.job_id() for job in jobs]
    
    # generate experiment id
    exp_id         = str(uuid.uuid4())
    
    # target files
    file_args      = exp_id + '.json'
    file_job_ids   = exp_id + '.txt'
    file_ansatz    = exp_id + '.pkl'
    file_rhs       = exp_id + '.pkl'
    file_innerp    = exp_id + '.pkl'
    file_trans     = exp_id + '.pkl'
    file_samples   = exp_id + '.pkl'
    
    target_args    = os.path.join(args.input_dir, 'args', file_args)
    target_job_ids = os.path.join(args.input_dir, 'experiment2jobs', file_job_ids)
    target_ansatz  = os.path.join(args.input_dir, 'ansaetze', file_ansatz)
    target_rhs     = os.path.join(args.input_dir, 'rhs', file_rhs)
    target_innerp  = os.path.join(args.input_dir, 'innerp', file_innerp)
    target_trans   = os.path.join(args.input_dir, 'trans_unbound', file_trans)
    target_samples = os.path.join(args.input_dir, 'samples', file_samples)
    
    # git info
    repo_path = args.git_repo_dir
    git_branch = subprocess.check_output(
        ["git", "--git-dir", repo_path, "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('UTF-8')
    git_commit_short_hash = subprocess.check_output(
        ["git", "--git-dir", repo_path, "describe", "--always"]).strip().decode('UTF-8')
    
    # input arguments
    with open(target_args, 'w') as file:
        args_dict = {k: args.__dict__[k] for k in sorted(args.__dict__.keys())}
        args_dict['git_branch']            = git_branch
        args_dict['git_commit_short_hash'] = git_commit_short_hash
        args_dict['new_shots']             = new_shots # add arguments
        json.dump(args_dict, file)
        
    # experiment id to job id
    with open(target_job_ids, "w") as file:
        for job_id in job_ids:
            file.write("%s\n" % job_id)

    # ansatz
    with open(target_ansatz, 'wb') as file:
        pickle.dump(ansatz_unbound, file)
        
    # rhs
    with open(target_rhs, 'wb') as file:
        pickle.dump(rhs_circ, file)
        
    # innerp
    with open(target_innerp, 'wb') as file:
        pickle.dump(innerp_circs, file)

    # transpiled
    with open(target_trans, 'wb') as file:
        pickle.dump(trans_unbound, file)
        
    # sampled parameters
    with open(target_samples, 'wb') as file:
        pickle.dump(samples, file)

    return exp_id

    
# %% save log data
#
def save_log(args, exp_id, jobs, new_shots):
    target = os.path.join(args.log_dir, 'log.csv')
    
    creation_date = None
    if args.backend == 'simulator':
        time = datetime.datetime.now().strftime("%H:%M:%S")
        date = datetime.date.today().strftime("%d/%m/%Y")
        creation_date = date + ', ' + time
    else:
        creation_date = jobs[0].creation_date()
        time          = creation_date.strftime("%H:%M:%S")
        date          = creation_date.strftime("%d/%m/%Y")
        creation_date = date + ', ' + time
        
    # git info
    repo_path         = args.git_repo_dir
    git_diff          = subprocess.check_output(
                        ["git", "--git-dir", repo_path, "diff"]).decode('UTF-8')
    job_ids           = [job.job_id() for job in jobs]
    file_git_diff     = exp_id + '.patch'
    git_diff_filepath = os.path.join(args.log_dir, 'git_diff', file_git_diff)
    
    if len(git_diff) > 0:
        with open(git_diff_filepath, 'w') as f:
            f.writelines(git_diff)

    data = {'exp_id': [exp_id],
            'job_id': [job_ids],
            'job_time': [creation_date],
            'backend': [args.backend],
            'shots': [new_shots],
            'archt': [args.archt],
            'qubits': [args.num_qubits],
            'layers': [args.num_layers],
            'rhs': [args.rhs],
            'innerp': [args.innerp],
            'qae': [args.qae],
            'qae_method': [args.qae_method],
            'max_power': [args.max_power],
            'iae_epsilon_target': [args.iae_epsilon_target],
            'iae_alpha': [args.iae_alpha],
            'fae_delta': [args.fae_delta],
            'fae_rescale': [args.fae_rescale],
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
    
    # for easier job saving
    target         = os.path.join(args.bash_dir, 'save_out_backend_job_list.txt')
    bash_save_data = ""
    for job_id in job_ids:
        bash_save_data += args.backend  + " " + job_id + " " + str(args.save_value) + "\n"

    with open(target, "a") as file:
        file.write(bash_save_data)
        
    # for easier job post processing
    target         = os.path.join(args.bash_dir, 'process_out_backend_exp_list.txt')
    bash_save_data = args.backend  + " " + exp_id + " " + str(args.save_value) + "\n"

    with open(target, "a") as file:
        file.write(bash_save_data)


# %% save output
#
def save_output(job, args):
    job_id = job.job_id()
    status = job.status()
    sim    = (args.backend=='simulator')
    
    results = None
    if status != JobStatus.DONE and not sim:
        print("Job not done")
        print(status.value)
        print("Saving job status instead...")
        results = status
    else:
        results = job.result()
    
    file   = job_id + '.pkl'
    target = os.path.join(args.output_dir, file)

    with open(target, 'wb') as file:
        pickle.dump(results, file)


# %% main
#
def main():
    args       = parse_arguments()
    mode       = args.run_mode
    qae        = args.qae
    qae_method = args.qae_method
    sim        = (args.backend=='simulator')
    shots      = args.shots

    if mode=='submit_experiment':
        device         = load_backend(args)
        ansatz_unbound = prepare_ansatz_circuit(args)
        rhs_circ       = prepare_rhs_circuit(args)
        innerp_driver  = prepare_innerp_driver(args, ansatz_unbound, rhs_circ)
        
        innerp_circs   = None
        new_shots      = [shots]
        if qae:
            if qae_method == 'iae':
                msg  = "For IAE, the chosen max_power has to be high enough "
                msg += "for post processing, otherwise post processing will " 
                msg += "raise an error."
                warnings.warn(msg)
            
            innerp_circs, new_shots = prepare_innerp_qaes(args,
                                                          device,
                                                          innerp_driver)
        else:
            innerp_circs = prepare_innerp_circuit(args, innerp_driver)
            
        trans_unbound  = transpile(args, device, innerp_circs)
        num_parameters = ansatz_unbound.num_parameters
        samples        = generate_samples(args, num_parameters)
        trans_bound    = bind_circuits(trans_unbound, samples)
        jobs           = submit_jobs(device, trans_bound, new_shots)
        
        exp_id         = save_input(args,
                                    jobs,
                                    ansatz_unbound,
                                    rhs_circ,
                                    innerp_circs,
                                    trans_unbound,
                                    samples,
                                    new_shots)
        
        save_log(args, exp_id, jobs, new_shots)
        
        if sim:
            for job in jobs:
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