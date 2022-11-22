"""
Script for processing results of experiments on IBM backends
to test the fidelity of ansatz circuits.
"""

# %% dependencies
#
import os
import sys
import warnings
import argparse
import json
import pickle
import numpy as np
import pandas as pd

from qiskit import Aer, IBMQ, QuantumCircuit
from qiskit.algorithms import (EstimationProblem,
                               MaximumLikelihoodAmplitudeEstimation,
                               IterativeAmplitudeEstimation,
                               FasterAmplitudeEstimation)
from qiskit.quantum_info import Statevector
from qiskit.utils import QuantumInstance
from qiskit.providers import JobStatus
import mthree

from vqa_poisson_tools.poisson.innerp import (InnerProductStatevec,
                                              get_innerp_circuit_driver)
from vqa_poisson_tools.utils.utils import counts2probs
from qiskit_transpiler_tools.transpiler import TranspilerSabreMapomaticDD


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
    parser.add_argument('--squared',
                        action='store_true',
                        help='Set flag if inner products are to be squared for error comparison')
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
    
    rhs_name           = exp_id + '.pkl'
    rhs_file           = os.path.join(input_dir, 'rhs', rhs_name)
    
    innerp_name        = exp_id + '.pkl'
    innerp_file        = os.path.join(input_dir, 'innerp', innerp_name)
    
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
        
    rhs = None
    with open(rhs_file, 'rb') as file:
        rhs = pickle.load(file)
        
    innerp_unbound = None
    with open(innerp_file, 'rb') as file:
        innerp_unbound = pickle.load(file)
        
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
    
    rhs_type           = args_input['rhs']
    innerp_type        = args_input['innerp']
    qae                = args_input['qae']
    qae_method         = args_input['qae_method']
    max_power          = args_input['max_power']
    iae_epsilon_target = args.iae_epsilon_target
    iae_alpha          = args.iae_alpha
    fae_delta          = args.fae_delta
    fae_rescale        = args_input['fae_rescale']
    
    shots       = args_input['shots']
    new_shots   = args_input['new_shots']
    num_samples = args_input['num_samples']
    sabre       = args_input['transpile_sabre']
    mapomatic   = args_input['transpile_mapomatic']
    dd          = args_input['transpile_dd']
    mem         = args.mem
    squared     = args.squared
    relative    = args.relative
    
    # check if exists
    squared_ = squared
    if innerp_type == 'Overlap':
        squared_ = False
    
    if exists:
        df         = pd.read_csv(target)
        exp_exists = (exp_id in df['exp_id'].values)
        subf       = df.loc[df['exp_id']==exp_id]
        mem_same   = (mem in subf['mem'].values)
        sq_same    = (squared_ in subf['squared'].values)
        err_same   = (relative in subf['relative'].values)
        
        if exp_exists and mem_same and sq_same and err_same and not force:
            msg="Experiment ID already saved, same MEM flag, same squared flag,\
 same relative flag, skipping..."
            sys.exit(msg)

    
    # bind parameters
    trans_bound = bind_circuits(trans_unbound, samples)
    
    return (exp_id,
            job_ids,
            backend,
            archt,
            num_qubits,
            num_layers,
            rhs_type,
            innerp_type,
            qae,
            qae_method,
            max_power,
            iae_epsilon_target,
            iae_alpha,
            fae_delta,
            fae_rescale,
            shots,
            new_shots,
            num_samples,
            sabre,
            mapomatic,
            dd,
            mem,
            ansatz_unbound,
            rhs,
            innerp_unbound,
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


# %% compute (simulated) ideal inner products
#
def compute_exact_innerps(rhs, ansatz_unbound, samples):
    
    num_samples   = len(samples)
    exact_innerps = [None]*num_samples
    
    for m in range(num_samples):
        ansatz_bound     = ansatz_unbound.bind_parameters(samples[m])
        innerp           = InnerProductStatevec(ansatz_bound, rhs)
        exact_innerps[m] = innerp.vdot_exact()
        
        
    return exact_innerps


# %% simplified result object
#
class MyResult():
    
    def __init__(self, counts, circuits=None):
        self._circuits = circuits
        self._counts   = counts
        
    
    def get_counts(self, experiment=None):
        
        if experiment is None:
            return self._counts
        
        if isinstance(experiment, int):
            return self._counts[experiment]
        
        if isinstance(experiment, QuantumCircuit):
            
            idx = None
            if isinstance(self._circuits, list):
                for q in range(len(self._circuits)):
                    if experiment.name == self._circuits[q].name:
                        idx = q
                        
            elif experiment.name != self._circuits.name:
                msg = "Circuit names do not match"
                raise ValueError(msg)
                
            else:
                return self._counts
                        
            if idx is None:
                msg = "Circuit not found"
                raise ValueError(msg)
                
            return self._counts[idx]
        
        msg = "Experiment type not recognized"
        raise ValueError(msg)


# %% fake backend that does a lookup
#
class LookupBackend(QuantumInstance):
    
    def __init__(self, *args, **kwargs):
        
        self._circuits  = kwargs.pop('circuits')
        self._new_shots = kwargs.pop('new_shots')
        self._results   = kwargs.pop('results')
        self._is_list   = kwargs.pop('is_list')
        
        if len(self._results) != len(self._new_shots):
            msg = "Lengths of result list and shots list do not match"
            raise ValueError(msg)
        
        QuantumInstance.__init__(self, *args, **kwargs)

        
    def execute(self, circuits, had_transpiled=True):
        
        # compare shots first
        current_shots =  self.run_config.shots
        idx           = None
        for i in range(len(self._new_shots)):
            if self._new_shots[i] == current_shots:
                idx = i
                break
            
        if idx is None:
            msg = "Shots match not found"
            raise ValueError(msg)
            
        # compare circuits next
        stored     = self._circuits
        num_stored = len(stored)
        
        if self._is_list:
            num_input = len(circuits)
            
            if num_stored != num_input:
                msg = "Length of stored circuit list does not match input"
                raise ValueError(msg)
            
            results = MyResult(self._results[idx], circuits)
            return results
        
        else:
            if type(circuits) is list:
                msg = "This circuit input cannot be a list"
                raise ValueError(msg)
            
            nomeas  = circuits.remove_final_measurements(inplace=False)

            test_qc = Statevector.from_instruction(nomeas)
            for i in range(num_stored):

                trial_qc = Statevector.from_instruction(stored[i])
                match    = trial_qc.equiv(test_qc)
                if match:
                    results = MyResult(self._results[idx][i], circuits)
                    return results
                
        msg = "Match not found"
        raise ValueError(msg)


# %% load results to fake backend
#
def compute_est_innerps_direct(innerp_driver,
                               num_samples,
                               sampled_probs):
    
    est_innerps = [None]*num_samples
    if len(sampled_probs) != num_samples:
        msg = "Length of counts and samples do not match"
        raise ValueError(msg)
    
    for m in range(num_samples):
        estimate       = innerp_driver.dot_from_counts(sampled_probs[m])
        est_innerps[m] = estimate
        
        
    return est_innerps
    

# %% compute estimated inner products
#
def compute_est_innerps_qae(results,
                            innerp_driver,
                            qae_method,
                            max_power,
                            iae_epsilon_target,
                            iae_alpha,
                            fae_delta,
                            fae_rescale,
                            new_shots,
                            samples):
    # QAE circuits
    device        = Aer.get_backend('aer_simulator')
    qi            = QuantumInstance(device, shots=100) # does not matter
    A             = innerp_driver.state_preparation()
    Q             = innerp_driver.Grover()
    
    # transpile (statevector fails otherwise)
    transpiler = TranspilerSabreMapomaticDD(device)
    A          = transpiler.transpile(A)
    Q          = transpiler.transpile(Q)
    
    oq            = innerp_driver.objective_qubits()
    is_good_state = innerp_driver.is_good_state
    problem       = EstimationProblem(state_preparation=A,
                                      grover_operator=Q,
                                      objective_qubits=oq,
                                      is_good_state=is_good_state)
    
    # Run the estimation with the fake backends
    is_list = False
    if qae_method == 'mlae':
        is_list = True
    
    num_samples = len(samples)
    est_innerps = [None]*num_samples
    num_circs   = max_power
    if qae_method == 'fae':
        num_circs = 2**(max_power-1)

    oracle_calls = []
    for m in range(num_samples):
        idx0          = m*num_circs
        idx1          = (m+1)*num_circs
        
        result_sample = []
        for i in range(len(results)):
            result_sample.append(results[i].get_counts()[idx0:idx1])
        
        A_bound       = A.bind_parameters(samples[m])
        Q_bound       = Q.bind_parameters(samples[m])
        est_problem   = EstimationProblem(state_preparation=A_bound,
                                          grover_operator=Q_bound,
                                          objective_qubits=oq,
                                          is_good_state=is_good_state)
        
        problem   = est_problem
        qae_circs = []
        if qae_method == 'mlae':
            estimator = MaximumLikelihoodAmplitudeEstimation(evaluation_schedule=max_power-1,
                                                             quantum_instance=qi)
            qae_circs = estimator.construct_circuits(problem, measurement=False)
        else:
            ks        = None
            estimator = None
            if qae_method == 'iae':
                ks        = list(range(max_power))
                estimator = IterativeAmplitudeEstimation(epsilon_target=iae_epsilon_target,
                                                         alpha=iae_alpha,
                                                         quantum_instance=qi)
            
            elif qae_method == 'fae':
                ks        = list(range(2**(max_power-1)))
                estimator = FasterAmplitudeEstimation(delta=fae_delta,
                                                      maxiter=max_power-1,
                                                      rescale=fae_rescale,
                                                      quantum_instance=qi)
                
                if fae_rescale:
                    problem = est_problem.rescale(0.25)
            for k in ks:        
                QkA = estimator.construct_circuit(problem, k, measurement=False)
                qae_circs.append(QkA)
            
        qi = LookupBackend(backend=device,
                           shots=new_shots[0],
                           circuits=qae_circs,
                           new_shots=new_shots,
                           results=result_sample,
                           is_list=is_list)

        estimator = None
        if qae_method == 'mlae':
            
            estimator = MaximumLikelihoodAmplitudeEstimation(evaluation_schedule=max_power-1,
                                                             quantum_instance=qi)
        elif qae_method == 'iae':
            estimator = IterativeAmplitudeEstimation(epsilon_target=iae_epsilon_target,
                                                     alpha=iae_alpha,
                                                     quantum_instance=qi)
        elif qae_method == 'fae':
            estimator = FasterAmplitudeEstimation(delta=fae_delta,
                                                  maxiter=max_power-1,
                                                  rescale=fae_rescale,
                                                  quantum_instance=qi)

        res            = estimator.estimate(est_problem)
        oracle_calls.append(res.num_oracle_queries)
        a              = res.estimation
        est_innerps[m] = innerp_driver.dot_from_amplitude(a)
        
    return est_innerps, oracle_calls


# %% compute errors
#
def compute_relative_errors(est_innerps, exact_innerps, squared=False):
    
    num_samples = len(exact_innerps)
    errors      = [None]*num_samples

    for m in range(num_samples):
        exact = exact_innerps[m]
        est   = est_innerps[m]
        if squared:
            exact = exact**2
            est   = est**2
        errors[m] = abs(est-exact)/abs(exact)
        
    return errors


def compute_absolute_errors(est_innerps, exact_innerps, squared=False):
    
    num_samples = len(exact_innerps)
    errors      = [None]*num_samples

    for m in range(num_samples):
        exact = exact_innerps[m]
        est   = est_innerps[m]
        if squared:
            exact = exact**2
            est   = est**2
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
              depths,
              cnots,
              rhs_type,
              innerp_type,
              qae,
              qae_method,
              max_power,
              oracle_calls,
              iae_epsilon_target,
              iae_alpha,
              fae_delta,
              fae_rescale,
              new_shots,
              num_samples,
              sabre,
              mapomatic,
              dd,
              squared,
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
            'rhs': [rhs_type],
            'innerp': [innerp_type],
            'qae': [qae],
            'qae_method': [qae_method],
            'max_power': [max_power],
            'oracle_calls': [oracle_calls],
            'iae_epsilon_target': [iae_epsilon_target],
            'iae_alpha': [iae_alpha],
            'fae_delta': [fae_delta],
            'fae_rescale': [fae_rescale],
            'new_shots': [new_shots],
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
            'squared': [squared],
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
    backend,
    archt,
    num_qubits,
    num_layers,
    rhs_type,
    innerp_type,
    qae,
    qae_method,
    max_power,
    iae_epsilon_target,
    iae_alpha,
    fae_delta,
    fae_rescale,
    shots,
    new_shots,
    num_samples,
    sabre,
    mapomatic,
    dd,
    mem,
    ansatz_unbound,
    rhs,
    innerp_unbound,
    trans_unbound,
    samples,
    trans_bound,
    results) = load_exps(args)
    
    mem           = args.mem
    squared       = args.squared
    relative      = args.relative
    sampled_probs = None
    verbose_mem   = args.verbose_mem

    if qae and mem:
        msg = "QAE and MEM currently don't work together, courtesy of Qiskit's \
ingenious software design..."
        raise ValueError(msg)
        
    if innerp_type=='Overlap' and squared:
        msg     = "Overlap already yields the squared inner product, resetting to False"
        squared = False
        warnings.warn(msg)
    
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
    
    exact_innerps = compute_exact_innerps(rhs, ansatz_unbound, samples)
    for m in range(num_samples):
        val = exact_innerps[m]
        if innerp_type == 'Overlap':
            exact_innerps[m] = val.real**2 + val.imag**2
        else:
            exact_innerps[m] = val.real
    
    innerp_driver = get_innerp_circuit_driver(innerp_type)
    innerp_driver = innerp_driver(ansatz_unbound, rhs)
    est_innerps   = None
    oracle_calls  = [shots]
    if not qae:
        est_innerps = compute_est_innerps_direct(innerp_driver,
                                                 num_samples,
                                                 sampled_probs)
    else:
        if qae_method == 'iae':
            msg  = "For IAE, max_power chosen at submission has to be high enough "
            msg += "for post processing, otherwise post processing will " 
            msg += "raise an error."
            warnings.warn(msg)
        est_innerps, oracle_calls = compute_est_innerps_qae(results,
                                                            innerp_driver,
                                                            qae_method,
                                                            max_power,
                                                            iae_epsilon_target,
                                                            iae_alpha,
                                                            fae_delta,
                                                            fae_rescale,
                                                            new_shots,
                                                            samples)
    
    compute_errors = compute_absolute_errors
    if relative:
        compute_errors = compute_relative_errors
        
    errors         = compute_errors(est_innerps, exact_innerps, squared)
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
              depths,
              cnots,
              rhs_type,
              innerp_type,
              qae,
              qae_method,
              max_power,
              oracle_calls,
              iae_epsilon_target,
              iae_alpha,
              fae_delta,
              fae_rescale,
              new_shots,
              num_samples,
              sabre,
              mapomatic,
              dd,
              squared,
              relative,
              mem,
              stats)
    

# %% run script
#
if __name__ == '__main__':
    main()