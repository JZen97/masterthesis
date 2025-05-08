# Functions for bootstrapping and generating bootstrap samples

import numpy as np
from sklearn.utils import resample
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
import json, pathlib

def generate_bootstrap_samples(n_nodes, n_iterations):
    ids = np.array([resample(range(n_nodes), n_samples=n_nodes) for _ in range(n_iterations)])
    return ids

# generate new bootstrap samples
#ids_val = generate_bootstrap_samples(10639, 1000)
#ids_test = generate_bootstrap_samples(11109, 1000)
# save ids
#np.save('bootstrap_ids_val.npy', ids_val)
#np.save('bootstrap_ids_test.npy', ids_test)

# or load previously defined bootstrap sample ids
#ids_val = np.load('bootstrap_ids_val.npy')
#ids_test = np.load('bootstrap_ids_test.npy')

def _eval(ids, input_tuple, score_fn, input_tuple2=None, score_fn_kwargs={}):
    sampled_inputs = [t[ids] for t in input_tuple]  # Subset inputs using bootstrapped indices
    #print(f"Evaluating on {len(ids)} samples")
    result = score_fn(*sampled_inputs, **score_fn_kwargs)
    #print(f"Score function output: {result}")
    return result

def empirical_bootstrap(input_tuple, score_fn, ids=None, n_iterations=1000, alpha=0.95, score_fn_kwargs={},threads=None, input_tuple2=None, ignore_nans=False, chunksize=50):
    '''
        performs empirical bootstrap https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf

        input_tuple: tuple of inputs for the score function typically something like (labels,predictions)
        score_function: scoring function that takes the individual entries of input tuple as argument e.g. f1_score
        id: list of previously sampled ids (if None new ids will be sampled)
        n_iterations: number of bootstrap iterations
        alpha: alpha-level for the confidence intervals
        score_fn_kwargs: additional (static) kwargs to be passed to the score_fn
        threads: number of threads (None uses os.cpu_count()); 0 no multithreading
        input_tuple2: if not None this is a second input of the same shape as input_tuple- in that case the function bootstraps the score difference between both inputs (this is just a convenience function- the same could be achieved by passing a tuple of the form (label,preds1,preds2) and computing the difference in the score_function itself)
        ignore_nans: ignore nans (e.g. no positives during during AUC evaluation) for score evaluation
        chunksize: process in chunks of size chunksize
    '''

    if(not(isinstance(input_tuple,tuple))):
        input_tuple = (input_tuple,)
    if(input_tuple2 is not None and not(isinstance(input_tuple2,tuple))):
        input_tuple2 = (input_tuple2,)

    score_point = score_fn(*input_tuple,**score_fn_kwargs) if input_tuple2 is None else score_fn(*input_tuple,**score_fn_kwargs)-score_fn(*input_tuple2,**score_fn_kwargs)

    if(n_iterations==0):
        return score_point,np.zeros(score_point.shape),np.zeros(score_point.shape),[]

    if(ids is None):
        ids = []
        for _ in range(n_iterations):
            ids.append(resample(range(len(input_tuple[0])), n_samples=len(input_tuple[0])))
        ids = np.array(ids)

    fn = partial(_eval,input_tuple=input_tuple,score_fn=score_fn,input_tuple2=input_tuple2,score_fn_kwargs=score_fn_kwargs)

    if(threads is not None and threads==0):
        results = []
        for sample_id in ids:
            results.append(fn(sample_id))
        results = np.array(results, dtype=np.float32)  # Convert to array after collection
    else:
        results=[]
        for istart in tqdm(np.arange(0,n_iterations,chunksize)):
            iend = min(n_iterations,istart+chunksize)
            pool = Pool(threads)
            results.append(np.array(pool.map(fn, ids[istart:iend])).astype(np.float32))
            pool.close()
            pool.join()

        results = np.concatenate(results,axis=0)

    percentile_fn = np.nanpercentile if ignore_nans else np.percentile

    score_diff = np.array(results)- score_point
    score_low = score_point + percentile_fn(score_diff, ((1.0-alpha)/2.0) * 100,axis=0)
    score_high = score_point + percentile_fn(score_diff, (alpha+((1.0-alpha)/2.0)) * 100,axis=0)
    std = np.std(score_diff)

    if(ignore_nans):#in this case return the number of nans in each score rather than the sampled ids (which could be different when evaluating several metrics at once)
        return score_point, score_low, score_high, np.sum(np.isnan(score_diff),axis=0)
    else:
        return score_point, score_low, score_high, std, ids

def MSE(targs, preds):
    score = np.mean((targs-preds)**2)
    return score

def gather_bootstrap_results(preds, targs, ids, out_path):
    target_names = ['PGA', 'PGV', 'SA03', 'SA10', 'SA30']
    results = {}

    # gather results for individual target variables
    for i, target_name in enumerate(target_names):
        pred, targ = preds[:, i], targs[:, i]
        input_tuple = (targ, pred)
        score_point, score_low, score_high, std, ids = empirical_bootstrap(input_tuple, MSE, ids=ids, n_iterations=1000, alpha=0.95, score_fn_kwargs={},threads=0, input_tuple2=None, ignore_nans=False, chunksize=1)
        results[target_name] = {}
        results[target_name]['score_point'] = float(score_point)
        results[target_name]['score_low'] = float(score_low)
        results[target_name]['score_high'] = float(score_high)
        results[target_name]['std'] = float(std)

    # gather results for all targets
    input_tuple = (targs, preds)
    score_point, score_low, score_high, std, ids = empirical_bootstrap(input_tuple, MSE, ids=ids, n_iterations=1000, alpha=0.95, score_fn_kwargs={},threads=0, input_tuple2=None, ignore_nans=False, chunksize=1)
    results['mean'] = {}
    results['mean']['score_point'] = float(score_point)
    results['mean']['score_low'] = float(score_low)
    results['mean']['score_high'] = float(score_high)
    results['mean']['std'] = float(std)

    # Save as JSON
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def gather_bootstrap_results_for_split(ids, in_path, out_path, split='test'):
    in_path = pathlib.Path(in_path.replace('SPLIT', split)).resolve()
    out_path = pathlib.Path(out_path.replace('SPLIT', split)).resolve()
    model_out = np.load(in_path)
    preds, targs = model_out['preds'], model_out['targs']
    results = gather_bootstrap_results(preds, targs, ids, out_path)
    return results

def apply_bootstrap_to_folder(model_output_folder, bootstrap_output_folder, ids_val, ids_test):
    model_output_folder = pathlib.Path(model_output_folder)
    bootstrap_output_folder = pathlib.Path(bootstrap_output_folder)

    for npz_file in list(model_output_folder.iterdir()):
        model_out = np.load(npz_file)
        preds, targs = model_out['preds'], model_out['targs']
        # get correct split bootstrap ids:
        if npz_file.name.endswith('test.npz'):
            ids = ids_test
        elif npz_file.name.endswith('val.npz'):
            ids = ids_val
        else:
            raise ValueError(f'Unknown file or split {npz_file.name}')

        # define output file path
        out_file_name = npz_file.name.replace('.npz', '.json')
        out_file_name = out_file_name.replace('_lr0.0001', '')
        out_file_path = bootstrap_output_folder / out_file_name
        results = gather_bootstrap_results(preds, targs, ids, out_file_path.resolve())
        print(f'{out_file_name} results: {results["mean"]["score_point"]} +- {results["mean"]["std"]} CI [{results["mean"]["score_low"]}, {results["mean"]["score_high"]}]')