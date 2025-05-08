# Experiment 3.1) Training a probabilistic model, predicting mean and variance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from tqdm import tqdm

import sys, time, pathlib, argparse, csv, itertools

src_path = '/fs/dss/home/lize4033/masterthesis/src/'
sys.path.append(src_path)
from data.graph_instance import GIDatasetPyG

parser = argparse.ArgumentParser(description='Train the model on EEW',
                                formatter_class=argparse.MetavarTypeHelpFormatter)
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--epochs', type=int, help='number of epochs')
parser.add_argument('--lambda_hsic', type=float, help='Weighting factor for the HSIC regularization loss')
parser.add_argument('--lambda_participation', type=float, help='Weighting factor for the participation regularization loss')
parser.add_argument('--participation_threshold', type=float, help='Threshold value for the participation loss')

args = parser.parse_args()

# ----------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------

class ModelExp6c(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=125, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=125, stride=2)

        # Graph Layers
        self.gcn1 = torch_geometric.nn.GCNConv(10051, 64, add_self_loops=True, normalize=True, bias=False)
        self.gcn2 = torch_geometric.nn.GCNConv(64, 64, add_self_loops=True, normalize=True, bias=False)

        # Output Layers
        self.fc_mean = nn.Linear(64, 5)  # PGA, PGV, SA03, SA10, SA30
        self.fc_variance = nn.Linear(64, 5)

    def forward(self, x1_ts, x2_static, edge_index, edge_weight):
        x = self.relu(self.conv1(x1_ts))  # out: bs x 32 x 438
        x = self.relu(self.conv2(x))  # out: bs x 64 x 157
        # flatten the cnn output
        x = torch.flatten(x, start_dim=1)  # out: bs x 10048
        # Concatenate output with static features
        x = torch.concat([x, x2_static], dim=1)  # out: bs x 10051
        # Apply graph layers
        x = self.relu(self.gcn1(x, edge_index=edge_index, edge_weight=edge_weight))  # out: bs x 64
        x = self.tanh(self.gcn2(x, edge_index=edge_index, edge_weight=edge_weight))  # out: bs x 64
        # Apply output layers
        # Node level predicitons
        mean = self.fc_mean(x)
        variance = self.fc_variance(x)
        # Enforce positivity of the variances. The 1e-6 is added for numerical stability
        variance = torch.log(1 + torch.exp(variance)) + 1e-6
        return mean, variance

class ModelExp6c_Ensemble(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([ModelExp6c() for _ in range(num_models)])

    def forward(self, *inputs):
        outputs = [model(*inputs) for model in self.models]
        # `outputs` will be a list of tuples: [(mean1, var1), (mean2, var2), ...]
        means, variances = zip(*outputs)
        means, variances = torch.stack(means, dim=2), torch.stack(variances, dim=2)
        # output has shape [nodes x targets x models]
        return means, variances

#----------------------------------------------------------------
# Training and Evaluation Functions
#----------------------------------------------------------------

def NLLLoss(mean, variance, targets):
    # calculate negative log likelihood loss
    loss = torch.sum(torch.log(variance) / 2 + ((targets - mean) ** 2) / (2 * variance))
    return loss

def get_kernel_matrix(data, kernel='rbf', sigma=1.0):
    """
    Compute the kernel matrix for HSIC loss.

    Parameters
    ----------
    data : torch.Tensor
        2D tensor of shape [d1, d2]. The kernel function is calculated in the d2 dimension between all pairs
        of elements in the d1 dimension, yielding a d1xd1 dimensional kernel matrix.
    kernel : str (default='rbf')
        kernel type (either linear or rbf).
    sigma : float (default=1.0)
        RBF kernel bandwidth.
    """
    if kernel == 'linear':
        return data @ data.T
    elif kernel == 'rbf':
        #print('data:', data)
        norms = (data**2).sum(1).reshape(-1, 1)
        #print('norms: ', norms)
        #print('norms.T: ', norms.T)
        #print('data @ data.T: ', data @ data.T)
        dists = norms - 2 * data @ data.T + norms.T
        #print('dists: ', dists)
        out = torch.exp(-dists / (2*sigma**2))
        #print('out: ', out, out.shape)
        return out
    else:
        raise ValueError('Unsupported kernel type.')

def get_hsic_loss (X, Y, kernel='rbf', sigma=1.0):
    """
    Calculate the (biased) HSIC loss between two sets of model outputs.

    Parameters
    ----------
    X : torch.tensor
        first model output.
    Y : torch.tensor
        second model output.
    kernel : str (default='rbf')
        kernel type (either linear or rbf).
    sigma : float (default=1.0)
        RBF kernel bandwidth.

    Returns
    -------
    hsic : torch.tensor
        HSIC loss between X and Y.
    """

    n = X.size(0)
    # centering matrix
    H = torch.eye(n) - torch.ones(n, n) / n
    print(H)
    #print((torch.ones(n,1) @ torch.ones(n,1).T)/n) # identical to the above definition of H
    # kernelize model outputs
    K = get_kernel_matrix(X, kernel, sigma)
    L = get_kernel_matrix(Y, kernel, sigma)
    # apply centering and calculate HSIC
    hsic = torch.trace(K @ H @ L @ H) /((n-1) ** 2)
    return hsic

def get_hsic_loss_unbiased(X, Y, kernel='rbf', sigma=1.0):
    """
    Calculate the (unbiased) HSIC loss between two sets of model outputs.

    Parameters
    ----------
    X : torch.tensor
        first model output.
    Y : torch.tensor
        second model output.
    kernel : str (default='rbf')
        kernel type (either linear or rbf).
    sigma : float (default=1.0)
        RBF kernel bandwidth.

    Returns
    -------
    hsic : torch.tensor
        HSIC loss between X and Y.
    """
    n = X.size(0)
    # kernelize model outputs
    K = get_kernel_matrix(X, kernel, sigma)
    L = get_kernel_matrix(Y, kernel, sigma)
    # set diagonals to zero
    K = K - torch.diag_embed(torch.diagonal(K))
    L = L - torch.diag_embed(torch.diagonal(L))

    ones = torch.ones(n,1, device=X.device)
    term_1 = torch.trace(K @ L)
    term_2 = (ones.T @ K @ ones * ones.T @ L @ ones) / ((n-1)*(n-2))
    term_3 = (-2 / (n-2)) * (ones.T @ K @ L @ ones)

    hsic = (term_1 + term_2 + term_3) / (n*(n-3))
    return hsic

def compute_ensemble_hsic_loss(variances, kernel='rbf', sigma=1.0, unbiased=True, normalize=True):
    """
    Compute the total HSIC loss between all pairs of model variance outputs.

    Parameters
    ----------
    variances : torch.Tensor
        Tensor of shape [nodes, targets, models] containing the predicted variances.
    kernel : str (default='rbf')
        'rbf' or 'linear' kernel for HSIC computation.
    sigma : float (default=1.0)
        RBF kernel bandwidth.
    unbiased : bool (default=True)
        Whether to use the unbiased HSIC estimator.
    normalize : bool (default=True)
        Whether to normalize the HSIC loss by the number of model combinations.

    Returns
    -------
    total_hsic : torch.Tensor
        Scalar HSIC loss summed across all model pairs.
    """
    # Get number of models
    num_models = variances.shape[-1]

    # Flatten node-target pairs to 2D: [nodes * targets, models]
    flat_vars = variances.view(-1, num_models)  # shape: [N*T, M]

    # Choose HSIC function
    hsic_fn = get_hsic_loss_unbiased if unbiased else get_hsic_loss

    # Accumulate HSIC across model pairs
    total_hsic = torch.tensor([0.0], device=variances.device)
    for i, j in itertools.combinations(range(num_models), 2):
        Xi = flat_vars[:, i].unsqueeze(1)  # shape [N*T, 1]
        Xj = flat_vars[:, j].unsqueeze(1)  # shape [N*T, 1]
        total_hsic += hsic_fn(Xi, Xj, kernel=kernel, sigma=sigma).squeeze()

    if normalize:
        total_hsic /= num_models * (num_models - 1) / 2

    return total_hsic

def get_participation_loss(weights, threshold=0.05):
    """
    Penalizes when models participate less than the threshold value on average.

    Parameters
    ----------
    weights : torch.Tensor
        weights of shape [nodes, targets, models] containing the weights for each model.
    threshold : float (default=0.05)
        weight threshold. If a model contributes less than this on average, it is penalized.

    Returns
    -------
    penalty : torch.Tensor
        the participation loss.
    """
    avg_weights = weights.mean(dim=(0,1)) # shape: [models]
    penalty = torch.relu(threshold - avg_weights).mean()
    return penalty


def evaluate_model(model, dl, criterion, participation_threshold):
    model.eval()
    total_node_loss = 0.
    total_node_mse_loss = 0.
    total_hsic_loss = 0.
    total_part_loss = 0.

    # keep track of weights/weighting factors of the individual models
    weights_lst = []

    with tqdm(total=len(dl)) as pbar:
        for idx, batch in enumerate(dl):
            # get data
            batch = batch.cuda()
            x = batch.x
            x1_ts = x[:, :3000].reshape(-1, 3, 1000) # the first 3000 (3 channels, 10s*100Hz=1000 values/channel) are time series, the rest are static features (position/elevation)
            x2_static = x[:, 3000:] # positione and elevation
            edge_index = batch.edge_index # normal adjacency matrix (edges)
            edge_weight = batch.edge_weight # edge weights
            pyg_batch = batch.batch # this defines which rows/columns belong to which sample in the batch (as adjacency matrices are put into a block matrix)
            # forward pass
            means, variances = model(x1_ts, x2_static, edge_index, edge_weight, pyg_batch)
            # get targets
            y = batch.y # ground truth intensity measurements
            #graph_y = batch.graph_y # epi dists (2), norm_factor (1), reference lat/lon (1)

            # Calculate model weights for each node and target
            weights = F.softmax(-variances, dim=2)

            # compute joint weighted mean and variance
            num_models = model.num_models
            mean = torch.sum(weights * means, dim=2) # sum across model dimension: [nodes x targets x models] -> [nodes x targets]
            variance = torch.sum(weights * (variances + means**2), dim=2) - mean**2

            # compute node level loss
            node_loss = NLLLoss(mean=mean, variance=variance, targets=torch.log10(y)) # node level loss (Intensity Measurements)
            # compute regularization losses
            hsic_loss = compute_ensemble_hsic_loss(variances, kernel='rbf', sigma=1.0, normalize=True)
            participation_loss = get_participation_loss(weights, threshold=participation_threshold)

            mse_loss = criterion(mean, torch.log10(y))
            # overall loss
            total_node_loss += node_loss.item()
            total_node_mse_loss += mse_loss.item()
            total_hsic_loss += hsic_loss.item()
            total_part_loss += participation_loss.item()

            # append weights
            weights_lst.append(weights)
            pbar.set_description(f'Evaluating...')
            pbar.update(1)

    # Compute statistics of the model weights
    all_weights = torch.cat(weights_lst, dim=0)
    # Mean and std across nodes: The mean tells us how much each model contributes to each target prediction on
    # average. The std tells us how much this fluctuates across nodes, i.e. a high std means the model focuses on
    # certain types of nodes.
    mean_weights_across_nodes = all_weights.mean(dim=0) # shape: [targets x models]
    std_weights_across_nodes = all_weights.std(dim=0) # shape: [targets x models]

    return total_node_loss, total_node_mse_loss, total_hsic_loss, total_part_loss, mean_weights_across_nodes, std_weights_across_nodes


def train_model(model, model_save_path, model_name, train_dl, val_dl, epochs, lr, lambda_hsic, lambda_participation, participation_threshold):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # list of losses (over the epochs) for training and validation split
    train_node_losses = []
    train_hsic_losses = []
    train_part_losses = []
    val_node_losses = []  # node level NLL loss
    val_node_mse_losses = []  # node level mse loss
    val_hsic_losses = []
    val_part_losses = []

    # for saving the model checkpoints
    last_model_checkpoint_path = None
    last_model_checkpoint_mse_path = None

    # Training
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_node_loss = 0.0
        total_hsic_loss = 0.0
        total_part_loss = 0.0

        with tqdm(total=len(train_dl)) as pbar:
            for idx, batch in enumerate(train_dl):
                # reset parameter gradients
                optimizer.zero_grad()
                # get data
                batch = batch.cuda()
                x = batch.x
                x1_ts = x[:, :3000].reshape(-1, 3, 1000) # the first 3000 (3 channels, 10s*100Hz=1000 values/channel) are time series, the rest are static features (position/elevation)
                x2_static = x[:, 3000:] # positione and elevation
                edge_index = batch.edge_index # normal adjacency matrix (edges)
                edge_weight = batch.edge_weight # edge weights
                pyg_batch = batch.batch # this defines which rows/columns belong to which sample in the batch (as adjacency matrices are put into a block matrix)
                # forward pass
                means, variances = model(x1_ts, x2_static, edge_index, edge_weight, pyg_batch)
                # get targets
                y = batch.y # ground truth intensity measurements
                #graph_y = batch.graph_y # epi dists (2), norm_factor (1), reference lat/lon (1)
                #epi_y = graph_y.reshape(train_dl.batch_size, -1)[:,:2] # only use the epicenter position for loss calculation

                # Calculate model weights for each node and target
                weights = F.softmax(-variances, dim=2)

                # compute joint weighted mean and variance
                num_models = model.num_models
                mean = torch.sum(weights * means, dim=2) # sum across model dimension: [nodes x targets x models] -> [nodes x targets]
                variance = torch.sum(weights * (variances + means**2), dim=2) - mean**2

                # compute node level loss
                node_loss = NLLLoss(mean=mean, variance=variance, targets=torch.log10(y)) # node level loss (Intensity Measurements)

                # compute regularization losses
                hsic_loss = compute_ensemble_hsic_loss(variances, kernel='rbf', sigma=1.0, normalize=True)
                participation_loss = get_participation_loss(weights, threshold=participation_threshold)


                # compute total loss
                total_loss = node_loss + lambda_hsic * hsic_loss + lambda_participation * participation_loss

                # backward pass
                total_loss.backward()
                # optimization step
                optimizer.step()
                # running variables
                total_node_loss += node_loss.item()
                total_hsic_loss += hsic_loss.item()
                total_part_loss += participation_loss.item()
                pbar.set_description(f'Epoch {epoch}. (NLL-loss: {node_loss.item()}) (HSIC-loss: {lambda_hsic}*{hsic_loss.item()}={lambda_hsic*hsic_loss.item()}) (Participation-loss: {lambda_participation}*{participation_loss.item()}={lambda_participation*participation_loss.item()}).')
                pbar.update(1)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # save training losses
        print(f'Epoch {epoch}, training duration: {epoch_duration:.2f}.Starting Evaluation...\n')
        train_node_losses.append(total_node_loss / len(train_dl))
        train_hsic_losses.append(total_hsic_loss / len(train_dl))
        train_part_losses.append(total_part_loss / len(train_dl))

        # Evaluation
        val_node_loss, val_node_mse_loss, val_hsic_loss, val_part_loss, mean_weights_across_nodes, std_weights_across_nodes = evaluate_model(model, val_dl, criterion, participation_threshold)
        val_node_losses.append(val_node_loss / len(val_dl))
        val_node_mse_losses.append(val_node_mse_loss / len(val_dl))
        val_hsic_losses.append(val_hsic_loss / len(val_dl))
        val_part_losses.append(val_part_loss / len(val_dl))

        # Plot message
        print(f'Evaluation: Train Loss (Node): {train_node_losses[-1]} | Val Loss (Node): {val_node_losses[-1]} | MSE Val Loss (Node): {val_node_mse_losses[-1]} | Val HSIC Loss: {lambda_hsic}*{val_hsic_losses[-1]}={lambda_hsic*val_hsic_losses[-1]} | Val Participation Loss: {lambda_participation}*{val_part_losses[-1]}={lambda_participation*val_part_losses[-1]}')

        # check if this is the best model so far:
        if (val_node_loss / len(val_dl)) <= min(val_node_losses):
            print('New best validation loss. Saving model...')
            # save the model state
            out_path = pathlib.Path(model_save_path) / f'{model_name}_epoch{epoch}_val_loss_{val_node_losses[-1]}_{val_node_mse_losses[-1]}_{val_hsic_losses[-1]}_{val_part_losses[-1]}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': total_node_loss,
                'val_loss': val_node_loss
            }, out_path)
            # remove the previously saved model which performed worse
            if last_model_checkpoint_path:
                pathlib.Path(last_model_checkpoint_path).unlink()
            # update model path
            last_model_checkpoint_path = str(out_path)

        # check if this is the best model so far:
        if (val_node_mse_loss / len(val_dl)) <= min(val_node_mse_losses):
            print('New best validation MSE loss. Saving model...')
            # save the model state
            out_path = pathlib.Path(model_save_path) / f'{model_name}_epoch{epoch}_val_loss_{val_node_losses[-1]}_{val_node_mse_losses[-1]}_{val_hsic_losses[-1]}_{val_part_losses[-1]}_best_mse.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': total_node_loss,
                'val_loss': val_node_loss,
                'mse_loss': val_node_mse_loss
            }, out_path)
            # remove the previously saved model which performed worse
            if last_model_checkpoint_mse_path:
                pathlib.Path(last_model_checkpoint_mse_path).unlink()
            # update model path
            last_model_checkpoint_mse_path = str(out_path)

        # Save losses and metrics
        csv_path = pathlib.Path(model_save_path) / f'{model_name}_losses.csv'
        if epoch == 0:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['epoch', 'train_node_loss', 'val_node_loss', 'val_node_mse_loss', 'val_hsic_loss', 'val_part_loss']
                # header for the mean and std of the weight across nodes (one value per model and target)
                target_names = ['pga', 'pgv', 'sa03', 'sa10', 'sa30']
                mean_header = [f'model_{m+1}_{target}_mean_w' for m in range(num_models) for target in target_names]
                std_header = [f'model_{m+1}_{target}_std_w' for m in range(num_models) for target in target_names]
                # add to header
                header.extend(mean_header)
                header.extend(std_header)
                # write header
                writer.writerow(header)
        # write evaluation results to csv file
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch, train_node_losses[-1], val_node_losses[-1], val_node_mse_losses[-1], val_hsic_losses[-1], val_part_losses[-1]]
            # get the means and stds of the weights across the nodes (1 value per model and target, for mean and std)
            flat_mean = mean_weights_across_nodes.t().flatten().tolist()  # flatten tensor to list
            flat_std = std_weights_across_nodes.t().flatten().tolist()
            # add to row
            row.extend(flat_mean)
            row.extend(flat_std)
            # write line to csv file
            writer.writerow(row)

    # After the training, save the last model state
    out_path = pathlib.Path(model_save_path) / f'{model_name}_epoch{epoch}_val_loss_{val_node_losses[-1]}_{val_node_mse_losses[-1]}_{val_hsic_losses[-1]}_{val_part_losses[-1]}_last.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_node_loss,
        'val_loss': val_node_loss
    }, out_path)

    return model

if __name__ == '__main__':
    #----------------------------------------------------------------
    # Dataset and Dataloader
    #----------------------------------------------------------------

    # Paths
    meta_csv_path = 'GraphInstance/events_meta_data.csv'
    distances_npy_path = 'GraphInstance/station_distances_km.npy'
    h5_path = 'GraphInstance/GraphInstance_events_gm.hdf5'
    train_path = 'GraphInstance/GI_train.txt'
    val_path = 'GraphInstance/GI_val.txt'
    test_path = 'GraphInstance/GI_test.txt'

    # Pytorch Geometric Dataset
    train_ds = GIDatasetPyG(meta_csv_path=meta_csv_path,
                            distances_npy_path=distances_npy_path,
                            h5_path=h5_path,
                            split_path=train_path,
                            ts_length=10,
                            edge_cutoff=0.0,
                            graph_format='reduced',
                            k_hop_reachability=2,
                            pad_ts_s=5,
                            verbose=False)

    val_ds = GIDatasetPyG(meta_csv_path=meta_csv_path,
                          distances_npy_path=distances_npy_path,
                          h5_path=h5_path,
                          split_path=val_path,
                          ts_length=10,
                          edge_cutoff=0.0,
                          graph_format='reduced',
                          k_hop_reachability=2,
                          pad_ts_s=5,
                          deterministic=True,
                          verbose=False)

    test_ds = GIDatasetPyG(meta_csv_path=meta_csv_path,
                           distances_npy_path=distances_npy_path,
                           h5_path=h5_path,
                           split_path=test_path,
                           ts_length=10,
                           edge_cutoff=0.0,
                           graph_format='reduced',
                           k_hop_reachability=2,
                           pad_ts_s=5,
                           deterministic=True,
                           verbose=False)

    # Dataloader
    train_dl = torch_geometric.data.DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    val_dl = torch_geometric.data.DataLoader(val_ds, batch_size=4, shuffle=False, drop_last=True)
    test_dl = torch_geometric.data.DataLoader(test_ds, batch_size=4, shuffle=False, drop_last=True)

    #----------------------------------------------------------------
    # Training
    #----------------------------------------------------------------

    model_exp_6c_ensemble = ModelExp6c_Ensemble(num_models=5).cuda()

    lr = args.lr
    epochs = args.epochs
    lambda_hsic = args.lambda_hsic
    lambda_participation = args.lambda_participation
    participation_threshold = args.participation_threshold


    model_save_path = 'adjust/this/path'
    model_name = f'ModelExp6c_ensemble_LAMhsic{lambda_hsic}_LAMpart{lambda_participation}_thresh{participation_threshold}'
    print(f'Starting Training with Learning rate: {lr} for {epochs} epochs. \n==============================================')
    model = train_model(model_exp_6c_ensemble, model_save_path, model_name, train_dl, val_dl, epochs, lr, lambda_hsic, lambda_participation, participation_threshold)
    print('Finished Training.')