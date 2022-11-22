"""
Script for running experiments on IBM backends
to test the accuracy of cost function evaluations.
"""

# %% dependencies
#
import os
import uuid
import subprocess
import argparse
import pickle
import json
import pandas as pd
import datetime

from math import pi
import numpy as np

from qiskit import IBMQ, QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.providers import JobStatus
from qiskit.providers.ibmq.runtime import UserMessenger

import mthree

from vqa_poisson_tools.ansatz.two_local import get_ansatz_archt
from vqa_poisson_tools.poisson.rhs import hn, get_rhs
from vqa_poisson_tools.poisson.operator import get_poisson_operator_driver
from vqa_poisson_tools.poisson.cost import get_poisson_cost_driver
from vqa_poisson_tools.poisson.innerp import get_innerp_circuit_driver
from qiskit_transpiler_tools.transpiler import TranspilerSabreMapomaticDD

from vqa_poisson_tools.runtime.scripts import vqa_poisson1d


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
                        help='<Required> run mode: submit_experiment, save_output or mem_calibration')
    parser.add_argument('--custom_tag',
                        metavar='tag',
                        help='User specified custom tag. Can be used to filter out experiments later.')
    parser.add_argument('--program_id',
                        metavar='mode',
                        help='ID of runtime program')
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
    parser.add_argument('--operator',
                        default='Paulis',
                        metavar='method',
                        help='Method to decompose operator')
    parser.add_argument('--innerp',
                        default='HTest',
                        metavar='method',
                        help='Type of inner product circuit')
    parser.add_argument('--cost',
                        default='Sato21_Nonnorm',
                        metavar='method',
                        help='Type of inner product circuit')
    parser.add_argument('--exact',
                        action='store_true',
                        help='Set flag to use numerically exact cost and gradients (statevector)')
    parser.add_argument('--w_grad',
                        action='store_true',
                        help='Set flag if optimizer uses gradients')
    parser.add_argument('--optimizer',
                        metavar='opt',
                        help='Name of optimizer')
    parser.add_argument('--hyper_file',
                        required=True,
                        metavar='file',
                        help='<Required> path to file containing optimizer hyperparameters')
    parser.add_argument('--seed_init',
                        type=int,
                        default=None,
                        metavar='seed',
                        help='RNG seed for initial value')
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
    parser.add_argument('--mem',
                        action='store_true',
                        help='Set flag to apply measurement error mitigation')
    parser.add_argument('--mem_dir',
                        default='./mem_calibration',
                        help='Directory for mem calibration files (to load or save)')
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
    parser.add_argument('--interim_dir',
                        default='./interim',
                        metavar='dir',
                        help='Directory for saving interim results')
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
                            helpful for the save_output run mode')
    parser.add_argument('--ehningen',
                        action='store_true',
                        help='Set flag if accessing Ehningen because apparently IBM is shit at writing software...')
    parser.add_argument('--token',
                        default=None,
                        help='Account token, only required for Ehningen')
    parser.add_argument('--url',
                        default=None,
                        help='Authentication URL, only required for Ehningen')
    parser.add_argument('--git_repo_dir',
                        required=True,
                        metavar='dir',
                        help='<Required> directory for root git repository')
    
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
        print("Total number of ansatz circuits : ", 1)
        print("----------- -------------------- ------- -----------")
    
    return ansatz_unbound


# %% prepare operator drivers
#
def prepare_operator_driver(args, ansatz_unbound):
    
    # input parameters
    
    operator_method = args.operator
    verbose         = args.verbose
    
    # verbose settings
    
    verbose_all             = False
    verbose_operator_driver = False
    
    if 'all' in verbose:
        verbose_all = True
    if 'operator_driver' in verbose:
        verbose_operator_driver = True
    
    # prepare list of drivers
    
    driver          = get_poisson_operator_driver(operator_method)
    operator_driver = driver(ansatz_unbound)

    if verbose_all or verbose_operator_driver:
        print("----------- Operator driver --------------------------")
        print("Total number of drivers : ", 1)
        print("----------- -------------------- ------- -----------")
            
    return operator_driver


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


# %% prepare cost drivers
#
def prepare_cost_driver(args, op_driver, innerp_driver):
    
    # input parameters
    
    cost_method = args.cost
    verbose     = args.verbose
    
    # verbose settings
    
    verbose_all             = False
    verbose_cost_driver = False
    
    if 'all' in verbose:
        verbose_all = True
    if 'cost_driver' in verbose:
        verbose_cost_driver = True
    
    # prepare list of drivers
    
    driver      = get_poisson_cost_driver(cost_method)
    cost_driver = driver(op_driver, innerp_driver)

    if verbose_all or verbose_cost_driver:
        print("----------- Cost driver --------------------------")
        print("Total number of drivers : ", 1)
        print("----------- -------------------- ------- -----------")
            
    return cost_driver


# %% prepare cost circuits
#
def prepare_cost_circuits(args, cost_driver):
    
    # input parameters
    
    verbose = args.verbose
    
    # verbose settings
    
    verbose_all          = False
    verbose_cost_circuit = False
    
    if 'all' in verbose:
        verbose_all = True
    if 'cost_circuit' in verbose:
        verbose_cost_circuit = True

    # operator circuit
    
    cost_circuits = cost_driver.get_circuits()
        
    # print

    if verbose_all or verbose_cost_circuit:
        print("----------- Cost circuit example ------------------")
        print(cost_circuits[0])
        print("\n")
        print("Total number of cost circuits : ", len(cost_circuits))
        print("----------- -------------------- ------- -----------")
    
    return cost_circuits


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


# %% prepare initial parameters
#
def generate_init(args, num_parameters):

    # input parameters
    seed_init = args.seed_init
   
    # initial parameters
    rng          = np.random.default_rng(seed=seed_init)
    init_params  = rng.uniform(low=0., high=4.*pi,  size=num_parameters)
    
    return init_params


# %% load optimizer hyperparameters
#
def load_hyper(args):
    
    hyper_params = None
    target       = args.hyper_file
    with open(target) as file:
        hyper_params = json.load(file)
    
    return hyper_params


# %% run MEM calibration
#
def run_mem_cal(args, trans_unbound, backend):
    
    print("Running M3 measurement error calibration")
    print("This may take some time...")
    
    name       = backend.name() + ".json"
    cals_file  = os.path.join(args.mem_dir, name)
    
    mit     = mthree.M3Mitigation(backend)
    mapping = mthree.utils.final_measurement_mapping(trans_unbound)
    mit.cals_from_system(mapping)
    mit.cals_to_file(cals_file)
        
    return 0


# %% set M3 MEM mitigator
#
def set_mitigator(backend, cals_file):
    
    mit     = mthree.M3Mitigation(backend)
    mit.cals_from_file(cals_file)
        
    return mit


# %% collect inputs
#
def runtime_inputs(args,
                   ansatz,
                   rhs_circ,
                   trans_unbound,
                   hyper,
                   init_params,
                   mit=None):
    
    sampler_options = {'shots': args.shots,
                       'use_mem': args.mem}
    
    inputs          = {'ansatz': ansatz,
                       'rhs': rhs_circ,
                       'circuits': trans_unbound,
                       'sampler_options': sampler_options,
                       'operator': args.operator,
                       'innerp': args.innerp,
                       'cost': args.cost,
                       'exact': args.exact,
                       'w_grad': args.w_grad,
                       'optimizer': args.optimizer,
                       'optimizer_options': hyper,
                       'init_params': init_params}

    return inputs


# %% submit runtime job
#
def submit_job(provider,
               device,
               simulator,
               program_id,
               inputs,
               user_messenger=UserMessenger(),
               interim_callback=None,
               call_sim=None):
    
    # simulator
    if simulator:
        result         = call_sim(device, user_messenger, **inputs)
        
        return result
    
    # real backend
    options = {'backend_name': device.name()}
    job     = provider.runtime.run(program_id=program_id,
                                   options=options,
                                   inputs=inputs)
                                   #callback=interim_callback) 

    return job
    

# %% save experiment input data
#
def save_input(args,
               job_id,
               ansatz_unbound,
               rhs_circ,
               cost_circuits,
               trans_unbound,
               init_params,
               hyper):
    exp_id = job_id
    
    # target files
    file_args      = exp_id + '.json'
    file_ansatz    = exp_id + '.pkl'
    file_rhs       = exp_id + '.pkl'
    file_cost      = exp_id + '.pkl'
    file_trans     = exp_id + '.pkl'
    file_init      = exp_id + '.pkl'
    file_hyper     = exp_id + '.txt'
    
    target_args     = os.path.join(args.input_dir, 'args', file_args)
    target_ansatz   = os.path.join(args.input_dir, 'ansaetze', file_ansatz)
    target_rhs      = os.path.join(args.input_dir, 'rhs', file_rhs)
    target_cost     = os.path.join(args.input_dir, 'cost', file_cost)
    target_trans    = os.path.join(args.input_dir, 'trans_unbound', file_trans)
    target_init     = os.path.join(args.input_dir, 'init_params', file_init)
    target_hyper    = os.path.join(args.input_dir, 'hyper', file_hyper)
    
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
        json.dump(args_dict, file)

    # ansatz
    with open(target_ansatz, 'wb') as file:
        pickle.dump(ansatz_unbound, file)
        
    # rhs
    with open(target_rhs, 'wb') as file:
        pickle.dump(rhs_circ, file)
        
    # cost
    with open(target_cost, 'wb') as file:
        pickle.dump(cost_circuits, file)

    # transpiled
    with open(target_trans, 'wb') as file:
        pickle.dump(trans_unbound, file)
        
    # initial parameters
    with open(target_init, 'wb') as file:
        pickle.dump(init_params, file)
        
    # hyper parameters
    with open(target_hyper, 'w') as file:
        json.dump(hyper, file)

    return 0

    
# %% save log data
#
def save_log(args, exp_id, job=None):
    target = os.path.join(args.log_dir, 'log.csv')
    
    creation_date = None
    if args.backend == 'simulator':
        time = datetime.datetime.now().strftime("%H:%M:%S")
        date = datetime.date.today().strftime("%d/%m/%Y")
        creation_date = date + ', ' + time
    else:
        creation_date = job.creation_date
        time          = creation_date.strftime("%H:%M:%S")
        date          = creation_date.strftime("%d/%m/%Y")
        creation_date = date + ', ' + time

    data = {'exp_id': [exp_id],
            'custom_tag': [args.custom_tag],
            'job_time': [creation_date],
            'programm_id': [args.program_id],
            'backend': [args.backend],
            'exact': [args.exact],
            'shots': [args.shots],
            'archt': [args.archt],
            'rhs': [args.rhs],
            'qubits': [args.num_qubits],
            'layers': [args.num_layers],
            'optimizer': [args.optimizer],
            'cost': [args.cost],
            'operator': [args.operator],
            'innerp': [args.innerp],
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
    target          = os.path.join(args.bash_dir, 'save_out_backend_job_list.txt')
    bash_save_data  = ""
    bash_save_data += args.backend  + " " + exp_id + " " + str(args.save_value) + "\n"

    with open(target, "a") as file:
        file.write(bash_save_data)
        
    # for easier job post processing
    target         = os.path.join(args.bash_dir, 'process_out_backend_exp_list.txt')
    bash_save_data = args.backend  + " " + exp_id + " " + str(args.save_value) + "\n"

    with open(target, "a") as file:
        file.write(bash_save_data)


# %% save output
#
def save_output(args, job=None, job_id=None, results=None):
    
    # for simulators
    sim = (args.backend=='simulator')
    if sim:
        file   = job_id + '.pkl'
        target = os.path.join(args.output_dir, file)
        with open(target, 'wb') as file:
            pickle.dump(results, file)
            
        return 0
        
    # real backends
    job_id   = job.job_id()
    status   = job.status()
    results_ = None
    if status != JobStatus.DONE:
        print("Job not done")
        print(status.value)
        print("Saving job status instead...")
        results_ = status
    else:
        results_ = job.result()

    file   = job_id + '.pkl'
    target = os.path.join(args.output_dir, file)
    with open(target, 'wb') as file:
        pickle.dump(results_, file)


# %% main
#
def main():
    args        = parse_arguments()
    mode        = args.run_mode
    program_id  = args.program_id
    sim         = (args.backend=='simulator')
    interim_dir = args.interim_dir
    exact       = args.exact
    
    job_id      = None
    if sim:
        job_id = str(uuid.uuid4())
    
    def interim_callback(job_id, interim_result):
        name      = str(job_id) + '.txt'
        file_name = os.path.join(interim_dir, name)
        
        with open(file_name, "a") as file:
            file.write(str(interim_result) + '\n')
            
            
    class MyUM(UserMessenger):
        
        def __init__(self, job_id):
            
            self._job_id = job_id
            
        
        def publish(self, message, encoder=None, final=False):
            
            interim_callback(self._job_id, json.dumps(message))
            
    um = MyUM(job_id)
            
    
    if mode=='submit_experiment':
        provider, device = load_backend(args)
        ansatz_unbound   = prepare_ansatz_circuit(args)
        op_driver        = prepare_operator_driver(args, ansatz_unbound)
        rhs_circ         = prepare_rhs_circuit(args)
        innerp_driver    = prepare_innerp_driver(args, ansatz_unbound, rhs_circ)
        cost_driver      = prepare_cost_driver(args, op_driver, innerp_driver)
        cost_circuits    = prepare_cost_circuits(args, cost_driver)
            
        trans_unbound  = [None]
        if not exact:
            trans_unbound  = transpile(args, device, cost_circuits)
        num_parameters = ansatz_unbound.num_parameters
        
        offset         = 0
        if args.cost == 'Sato21_Nonnorm':
            offset = 1
        
        init_params    = generate_init(args, num_parameters+offset)
        hyper_params   = load_hyper(args)
        inputs         = runtime_inputs(args,
                                        ansatz_unbound,
                                        rhs_circ,
                                        trans_unbound,
                                        hyper_params,
                                        init_params)
        job            = submit_job(provider,
                                    device,
                                    sim,
                                    program_id,
                                    inputs,
                                    um,
                                    interim_callback,
                                    call_sim=vqa_poisson1d.main)
        
        results = None 
        if sim:
            results = job
        else:
            job_id = job.job_id()
        
        save_input(args,
                   job_id,
                   ansatz_unbound,
                   rhs_circ,
                   cost_circuits,
                   trans_unbound,
                   init_params,
                   hyper_params)
        
        save_log(args, job_id, job)
        
        if sim:
            save_output(args, job, job_id, results)
    
    elif mode=='mem_calibration':
        msg = "Currently mitigators not serializable, i.e., MEM calibration cannot be run."
        raise ValueError(msg)
        
    elif mode=='save_output':
        if sim:
            print("Backend is a simulator, job should be already saved")
            print("Exiting...")
            return

        job_id = args.job_id
        
        file   = job_id + '.pkl'
        target = os.path.join(args.output_dir, file)
        exists = os.path.exists(target)
        
        if exists:
            results = None 
            with open(target, 'rb') as file:
                results = pickle.load(file)
        
            if isinstance(results, JobStatus):
                msg  = f"This job was not done ({results.value}). Retrying save..."
                print(msg)
            else:
                print("Job already saved")
                print("Exiting...")
                return
        
        provider, device = load_backend(args)
        job              = provider.runtime.job(job_id)
        save_output(args, job)
    
    else:
        msg = "Unknown run mode (see help -h)."
        raise ValueError(msg)
        

# %% run script
#
if __name__ == '__main__':
    main()