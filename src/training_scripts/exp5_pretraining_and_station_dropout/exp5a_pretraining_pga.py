# Experiment 5a) Pretrain Model with PGA

import torch
import torch.nn as nn
import torch_geometric
from tqdm import tqdm

import sys, time, pathlib, argparse

src_path = 'adjust/this/path'
sys.path.append(src_path)
from data.graph_instance import GIDatasetPretrainPyG

parser = argparse.ArgumentParser(description='Train the model on EEW',
                                 formatter_class=argparse.MetavarTypeHelpFormatter)
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--epochs', type=int, help='number of epochs')
parser.add_argument('--dropout', type=float, help='fraction of stations to dropout')

args = parser.parse_args()


# ----------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------

class ModelExp5a(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=125, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=125, stride=2)

        # Graph Layers
        self.gcn1 = torch_geometric.nn.GCNConv(10051, 64, add_self_loops=True, normalize=True, bias=False)
        self.gcn2 = torch_geometric.nn.GCNConv(64, 64, add_self_loops=True, normalize=True, bias=False)

        # New output layer for PGA only
        self.fc_pga = nn.Linear(64, 1)

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
        # Node level predicitons (PGA)
        pga = self.fc_pga(x)
        return pga


# ----------------------------------------------------------------
# Training and Evaluation Functions
# ----------------------------------------------------------------

def evaluate_model(model, dl, criterion):
    model.eval()
    total_node_loss = 0.
    for idx, batch in enumerate(dl):
        # get data
        batch = batch.cuda()
        x = batch.x
        x1_ts = x[:, :3000].reshape(-1, 3,
                                    1000)  # the first 3000 (3 channels, 10s*100Hz=1000 values/channel) are time series, the rest are static features (position/elevation)
        x2_static = x[:, 3000:]  # positione and elevation
        edge_index = batch.edge_index  # normal adjacency matrix (edges)
        edge_weight = batch.edge_weight  # edge weights
        pyg_batch = batch.batch  # this defines which rows/columns belong to which sample in the batch (as adjacency matrices are put into a block matrix)
        # forward pass
        ims = model(x1_ts, x2_static, edge_index, edge_weight, pyg_batch)
        # get targets
        y = batch.y  # ground truth intensity measurements
        graph_y = batch.graph_y  # epi dists (2), norm_factor (1), reference lat/lon (1)
        # obtain mask of non-zero node targets
        non_zero_mask = batch.non_zero_mask
        # compute node and graph level loss
        node_loss = criterion(ims[non_zero_mask], torch.log10(y[non_zero_mask]))  # node level loss (Intensity Measurements)
        # overall loss
        total_node_loss += node_loss.item()
    return total_node_loss


def train_model(model, model_save_path, model_name, train_dl, val_dl, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # list of losses (over the epochs) for training and validation split
    train_node_losses = []
    val_node_losses = []

    # for saving the model checkpoints
    last_model_checkpoint_path = None

    # Training
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_node_loss = 0.0

        with tqdm(total=len(train_dl)) as pbar:
            for idx, batch in enumerate(train_dl):
                # reset parameter gradients
                optimizer.zero_grad()
                # get data
                batch = batch.cuda()
                x = batch.x
                x1_ts = x[:, :3000].reshape(-1, 3,
                                            1000)  # the first 3000 (3 channels, 10s*100Hz=1000 values/channel) are time series, the rest are static features (position/elevation)
                x2_static = x[:, 3000:]  # positione and elevation
                edge_index = batch.edge_index  # normal adjacency matrix (edges)
                edge_weight = batch.edge_weight  # edge weights
                pyg_batch = batch.batch  # this defines which rows/columns belong to which sample in the batch (as adjacency matrices are put into a block matrix)
                # forward pass
                ims = model(x1_ts, x2_static, edge_index, edge_weight, pyg_batch)
                # get targets
                y = batch.y  # ground truth intensity measurements
                graph_y = batch.graph_y  # epi dists (2), norm_factor (1), reference lat/lon (1)
                epi_y = graph_y.reshape(train_dl.batch_size, -1)[:,
                        :2]  # only use the epicenter position for loss calculation
                # obtain mask of non-zero node targets
                non_zero_mask = batch.non_zero_mask
                # compute node and graph level loss
                node_loss = criterion(ims[non_zero_mask], torch.log10(y[non_zero_mask]))  # node level loss (Intensity Measurements)
                # print(loss)
                # backward pass
                node_loss.backward()
                # optimization step
                optimizer.step()
                # running variables
                total_node_loss += node_loss.item()
                pbar.set_description(f'Epoch {epoch}. (Node-loss: {node_loss.item()}).')
                pbar.update(1)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # save training losses
        print(f'Epoch {epoch}, training duration: {epoch_duration:.2f}.Starting Evaluation...')
        train_node_losses.append(total_node_loss / len(train_dl))

        # Evaluation
        val_node_loss = evaluate_model(model, val_dl, criterion)
        val_node_losses.append(val_node_loss / len(val_dl))

        # Plot message
        print(f'Evaluation: Train Loss (Node): {train_node_losses[-1]} | Val Losses (Node): {val_node_losses[-1]}')

        # check if this is the best model so far:
        if (val_node_loss / len(val_dl)) <= min(val_node_losses):
            print('New best validation loss. Saving model...')
            # save the model state
            out_path = pathlib.Path(
                model_save_path) / f'{model_name}_epoch{epoch}_val_loss_{val_node_losses[-1]}_best.pt'
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

    # After the training, save the last model state
    out_path = pathlib.Path(model_save_path) / f'{model_name}_epoch{epoch}_val_loss_{val_node_losses[-1]}_last.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_node_loss,
        'val_loss': val_node_loss
    }, out_path)

    # Save losses
    out_path = pathlib.Path(model_save_path) / f'{model_name}_losses.csv'
    header = ['epoch,train_node_loss,val_node_loss\n']
    lines = [f'{i},{train_node_losses[i]},{val_node_losses[i]}\n' for i in range(epochs)]
    header.extend(lines)
    with open(out_path, 'w') as f:
        f.writelines(header)
    return model


if __name__ == '__main__':
    # ----------------------------------------------------------------
    # Dataset and Dataloader
    # ----------------------------------------------------------------

    # Paths
    meta_csv_path = 'GraphInstance/events_meta_data.csv'
    distances_npy_path = 'GraphInstance/station_distances_km.npy'
    h5_path = 'GraphInstance/GraphInstance_events_gm.hdf5'
    train_path = 'GraphInstance/GI_train.txt'
    val_path = 'GraphInstance/GI_val.txt'
    test_path = 'GraphInstance/GI_test.txt'

    dropout = args.dropout

    # Pytorch Geometric Dataset
    train_ds = GIDatasetPretrainPyG(meta_csv_path=meta_csv_path,
                                    distances_npy_path=distances_npy_path,
                                    h5_path=h5_path,
                                    split_path=train_path,
                                    ts_length=10,
                                    duration=20,
                                    dropout=dropout,
                                    edge_cutoff=0.0,
                                    graph_format='reduced',
                                    k_hop_reachability=2,
                                    pad_ts_s=0,
                                    verbose=False)

    val_ds = GIDatasetPretrainPyG(meta_csv_path=meta_csv_path,
                                  distances_npy_path=distances_npy_path,
                                  h5_path=h5_path,
                                  split_path=val_path,
                                  ts_length=10,
                                  duration=20,
                                  dropout=dropout,
                                  edge_cutoff=0.0,
                                  graph_format='reduced',
                                  k_hop_reachability=2,
                                  pad_ts_s=0,
                                  deterministic=True,
                                  verbose=False)

    test_ds = GIDatasetPretrainPyG(meta_csv_path=meta_csv_path,
                                   distances_npy_path=distances_npy_path,
                                   h5_path=h5_path,
                                   split_path=test_path,
                                   ts_length=10,
                                   duration=20,
                                   dropout=dropout,
                                   edge_cutoff=0.0,
                                   graph_format='reduced',
                                   k_hop_reachability=2,
                                   pad_ts_s=0,
                                   deterministic=True,
                                   verbose=False)

    # Dataloader
    train_dl = torch_geometric.data.DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    val_dl = torch_geometric.data.DataLoader(val_ds, batch_size=4, shuffle=False, drop_last=True)
    test_dl = torch_geometric.data.DataLoader(test_ds, batch_size=4, shuffle=False, drop_last=True)

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------

    # define model
    model = ModelExp5a().cuda()

    epochs = args.epochs
    lr = args.lr
    model_save_path = 'adjust/this/path'
    model_name = f'ModelExp5a_lr{lr}_dropout_{dropout}'
    print(
        f'Starting Pretraining with Learning rate: {lr} for {epochs} epochs (dropout: {dropout}). \n==============================================')
    model = train_model(model, model_save_path, model_name, train_dl, val_dl, epochs, lr)
    print('Finished Training.')
