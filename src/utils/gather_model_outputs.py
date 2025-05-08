# Utility functions for evaluating models and saving their outputs.

import torch
import torch_geometric
from tqdm.notebook import tqdm
import numpy as np

import os, sys, pathlib
src_path = os.path.abspath(os.path.join('.', '..', '..', 'src'))
sys.path.append(src_path)
from data.graph_instance import GIDatasetPyG

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# Paths
meta_csv_path = 'GraphInstance/events_meta_data.csv'
distances_npy_path = 'GraphInstance/station_distances_km.npy'
h5_path = 'GraphInstance/GraphInstance_events_gm.hdf5'
train_path = 'GraphInstance/GI_train.txt'
val_path = 'GraphInstance/GI_val.txt'
test_path = 'GraphInstance/GI_test.txt'

def aggregate_preds_targs(model, dl):
    # evaluates the model outputs for a single model and dataset configuration
    model.eval()
    preds = []
    targs = []
    with tqdm(total=len(dl)) as pbar:
        for idx, batch in enumerate(dl):
            # get data
            batch = batch.to(device)
            x = batch.x
            x1_ts = x[:, :3000].reshape(-1, 3, 1000) # the first 3000 (3 channels, 10s*100Hz=1000 values/channel) are time series, the rest are static features (position/elevation)
            x2_static = x[:, 3000:] # positione and elevation
            edge_index = batch.edge_index # normal adjacency matrix (edges)
            edge_weight = batch.edge_weight # edge weights
            pyg_batch = batch.batch # this defines which rows/columns belong to which sample in the batch (as adjacency matrices are put into a block matrix)
            # forward pass
            pred = model(x1_ts, x2_static, edge_index, edge_weight, pyg_batch)
            # if the output contains multiple elements, take only the first one (intensity measurement point prediction)
            if isinstance(pred, tuple):
                pred = pred[0]
            # get targets
            targ = torch.log10(batch.y) # ground truth intensity measurements
            # convert to numpy
            pred = pred.cpu().detach().numpy()
            targ = targ.cpu().detach().numpy()
            # append predictions and targets
            preds.append(pred)
            targs.append(targ)
            # update progress bar
            pbar.update(1)
    # combine these into a single numpy array along the node dimension
    preds = np.concatenate(preds, axis=0)
    targs = np.concatenate(targs, axis=0)
    return preds, targs

def export_preds_targs(model, state_dict_path, out_path, ds):
    # evaluates and saves the model outputs for different time series paddings
    # load pretrained state dict
    checkpoint = torch.load(state_dict_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    # update model state dict (strict=False -> mismatches in layer names are ignored (i.e. final layer))
    model.load_state_dict(state_dict, strict=True)

    # Configure dataloader and aggregate predictions
    for pad_ts_s in [0, 1, 2, 3, 4, 5]:
        # set time series start to a fixed time before the first arrival time
        ds.random_ts_padding = False
        ds.pad_ts_s = pad_ts_s
        # configure dataloader
        dl = torch_geometric.data.DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)

        # aggregate results
        preds, targs = aggregate_preds_targs(model, dl)

        # define output path
        out_path_s = pathlib.Path(str(out_path).replace('_results_', f'_pad{pad_ts_s}s_'))

        # save results
        np.savez(out_path_s, preds=preds, targs=targs)
        print('saved results to: {}'.format(out_path_s))
    return preds, targs

def export_preds_targs_dropout(model, state_dict_path, out_path_val=None, out_path_test=None, dropout_ratios=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    # evaluate and save model outputs for different time series paddings and station dropout ratios
    for dropout_ratio in dropout_ratios:
        if out_path_val:
            print(f'Gather results for validation dataset with dropout ratio: {dropout_ratio}')
            out_path_val_dropout = out_path_val.replace('DROPOUT', f'dropout{dropout_ratio}')
            # create dataset and dataloader
            val_ds = GIDatasetPyG(meta_csv_path=meta_csv_path,
                                  distances_npy_path=distances_npy_path,
                                  h5_path=h5_path,
                                  split_path=val_path,
                                  ts_length=10,
                                  dropout=dropout_ratio,
                                  edge_cutoff=0.0,
                                  graph_format='reduced',
                                  k_hop_reachability=2,
                                  pad_ts_s=5,
                                  deterministic=True,
                                  verbose=False)

            # gather results
            preds, targs = export_preds_targs(model, state_dict_path, out_path_val_dropout, val_ds)
        if out_path_test:
            print(f'Gather results for test dataset with dropout ratio: {dropout_ratio}')
            out_path_test_dropout = out_path_test.replace('DROPOUT', f'dropout{dropout_ratio}')
            # create dataset and dataloader
            test_ds = GIDatasetPyG(meta_csv_path=meta_csv_path,
                                   distances_npy_path=distances_npy_path,
                                   h5_path=h5_path,
                                   split_path=test_path,
                                   ts_length=10,
                                   dropout=dropout_ratio,
                                   edge_cutoff=0.0,
                                   graph_format='reduced',
                                   k_hop_reachability=2,
                                   pad_ts_s=5,
                                   deterministic=True,
                                   verbose=False)

            # gather results
            preds, targs = export_preds_targs(model, state_dict_path, out_path_test_dropout, test_ds)

def export_preds_targs_mixture_model(model, out_path, ds):
    # evaluate and save mixture model outputs for different time series paddings
    # Configure dataloader and aggregate predictions
    for pad_ts_s in [0, 1, 2, 3, 4, 5]:
        # set time series start to a fixed time before the first arrival time
        ds.random_ts_padding = False
        ds.pad_ts_s = pad_ts_s
        # configure dataloader
        dl = torch_geometric.data.DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)

        # aggregate results
        preds, targs = aggregate_preds_targs(model, dl)

        # define output path
        out_path_s = pathlib.Path(str(out_path).replace('_results_', f'_pad{pad_ts_s}s_'))

        # save results
        np.savez(out_path_s, preds=preds, targs=targs)
        print('saved results to: {}'.format(out_path_s))
    return preds, targs