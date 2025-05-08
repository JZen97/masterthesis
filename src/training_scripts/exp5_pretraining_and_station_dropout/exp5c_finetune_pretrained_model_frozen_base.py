# Experiment 5c) Finetune pretrained model with frozen base in the first 10 epochs

import torch
import torch.nn as nn
import torch_geometric
from tqdm import tqdm

import os, sys, pathlib, argparse, csv

src_path = 'adjust/this/path'
sys.path.append(src_path)
from data.graph_instance import GIDatasetPyG

parser = argparse.ArgumentParser(description='Train the model on EEW',
                                 formatter_class=argparse.MetavarTypeHelpFormatter)
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--epochs', type=int, help='number of epochs')
parser.add_argument('--dropout', type=float, help='fraction of stations to dropout')
parser.add_argument('--epochs_to_unfreeze', type=int, help='epochs until model base is unfrozen')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint path')
args = parser.parse_args()


# ----------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------

class ModelExp5c(nn.Module):
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
        self.fc_ims = nn.Linear(64, 5)  # PGA, PGV, SA03, SA10, SA30

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
        ims = self.fc_ims(x)
        return ims


# ----------------------------------------------------------------
# Training and Evaluation Functions
# ----------------------------------------------------------------
def evaluate_model(model, dl, criterion):
    model.eval()
    total_node_loss = 0.
    with tqdm(total=len(dl)) as pbar:
        for idx, batch in enumerate(dl):
            # get data
            batch = batch.cuda()
            x = batch.x
            x1_ts = x[:, :3000].reshape(-1, 3, 1000) # the first 3000 (3 channels, 10s*100Hz=1000 values/channel) are time series, the rest are static features (position/elevation)
            x2_static = x[:, 3000:] # position and elevation
            edge_index = batch.edge_index # normal adjacency matrix (edges)
            edge_weight = batch.edge_weight # edge weights
            pyg_batch = batch.batch # this defines which rows/columns belong to which sample in the batch (as adjacency matrices are put into a block matrix)
            # forward pass
            ims = model(x1_ts, x2_static, edge_index, edge_weight, pyg_batch)
            # get targets
            y = batch.y # ground truth intensity measurements
            graph_y = batch.graph_y # epi dists (2), norm_factor (1), reference lat/lon (1)
            # compute node and graph level loss
            node_loss = criterion(ims, torch.log10(y)) # node level loss (Intensity Measurements)
            # overall loss
            total_node_loss += node_loss.item()
            pbar.set_description(f'Evaluating...')
            pbar.update(1)
    return total_node_loss


def train_model(model, model_save_path, model_name, train_dl, val_dl, val_dl_dropout, epochs, lr, epochs_to_unfreeze, checkpoint_path=None):
    # load checkpoint if desired
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        is_finetuning = 'FT_'
        print(f'Loaded model parameters from {checkpoint_path}')
    else:
        start_epoch = 0
        print('Initializing model from scratch...')
        is_finetuning = ''
    stop_epoch = start_epoch + epochs
    print(f'Starting training for epochs {start_epoch} to {stop_epoch}...')
    last_model_checkpoint_path = None
    last_model_checkpoint_path_dropout = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # list of losses (over the epochs) for training and validation split
    train_node_losses = []
    val_node_losses = []
    val_node_losses_dropout = []

    # freeze all model base layers
    if not checkpoint_path:
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze final layer
        for param in model.fc_ims.parameters():
            param.requires_grad = True

    # Training
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        total_node_loss = 0.0

        # unfreeze model base
        if not checkpoint_path and epoch == epochs_to_unfreeze:
            for param in model.parameters():
                param.requires_grad = True

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
                ims = model(x1_ts, x2_static, edge_index, edge_weight, pyg_batch)
                # get targets
                y = batch.y # ground truth intensity measurements
                graph_y = batch.graph_y # epi dists (2), norm_factor (1), reference lat/lon (1)
                epi_y = graph_y.reshape(train_dl.batch_size, -1)[:,:2] # only use the epicenter position for loss calculation
                # compute node and graph level loss
                node_loss = criterion(ims, torch.log10(y)) # node level loss (Intensity Measurements)
                # backward pass
                node_loss.backward()
                # optimization step
                optimizer.step()
                # running variables
                total_node_loss += node_loss.item()
                pbar.set_description(f'Epoch {epoch}. (Node-loss: {node_loss.item()}).')
                pbar.update(1)

        # save training losses
        train_node_losses.append(total_node_loss / len(train_dl))

        # Evaluation
        val_node_loss = evaluate_model(model, val_dl, criterion)
        val_node_losses.append(val_node_loss / len(val_dl))

        val_node_loss_dropout = evaluate_model(model, val_dl_dropout, criterion)
        val_node_losses_dropout.append(val_node_loss_dropout / len(val_dl))

        # Plot message
        print(f'Evaluation: Train Loss (Node): {train_node_losses[-1]} | Val Losses (Node): {val_node_losses[-1]}')

        # check if this is the best model so far:
        if (val_node_loss / len(val_dl)) <= min(val_node_losses):
            print('New best validation loss. Saving model...')
            # save the model state
            out_path = pathlib.Path(model_save_path) / f'{is_finetuning}{model_name}_epoch{epoch}_val_loss_{val_node_losses[-1]}_best.pt'
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
        if (val_node_loss_dropout / len(val_dl)) <= min(val_node_losses_dropout):
            print('New best validation loss. Saving model...')
            # save the model state
            out_path = pathlib.Path(model_save_path) / f'{is_finetuning}{model_name}_epoch{epoch}_val_loss_{val_node_losses[-1]}_best_dropout.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': total_node_loss,
                'val_loss': val_node_loss
            }, out_path)
            # remove the previously saved model which performed worse
            if last_model_checkpoint_path_dropout:
                pathlib.Path(last_model_checkpoint_path_dropout).unlink()
            # update model path
            last_model_checkpoint_path_dropout = str(out_path)

        # Save losses and metrics
        csv_path = pathlib.Path(model_save_path) / f'{model_name}_losses.csv'
        if epoch == 0:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['epoch', 'train_node_loss', 'val_node_loss', 'val_node_loss_dropout']
                # write header
                writer.writerow(header)
        # write evaluation results to csv file
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch, train_node_losses[-1], val_node_losses[-1], val_node_losses_dropout[-1]]
            # write line to csv file
            writer.writerow(row)

    # After the training, save the last model state
    out_path = pathlib.Path(model_save_path) / f'{is_finetuning}{model_name}_epoch{epoch}_val_loss_{val_node_losses[-1]}_last.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_node_loss,
        'val_loss': val_node_loss
    }, out_path)

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
    train_ds = GIDatasetPyG(meta_csv_path=meta_csv_path,
                            distances_npy_path=distances_npy_path,
                            h5_path=h5_path,
                            split_path=train_path,
                            ts_length=10,
                            dropout=dropout,
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
                          dropout=0.0,
                          edge_cutoff=0.0,
                          graph_format='reduced',
                          k_hop_reachability=2,
                          pad_ts_s=5,
                          deterministic=True,
                          verbose=False)

    val_ds_dropout = GIDatasetPyG(meta_csv_path=meta_csv_path,
                          distances_npy_path=distances_npy_path,
                          h5_path=h5_path,
                          split_path=val_path,
                          ts_length=10,
                          dropout=dropout,
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
                           dropout=dropout,
                           edge_cutoff=0.0,
                           graph_format='reduced',
                           k_hop_reachability=2,
                           pad_ts_s=5,
                           deterministic=True,
                           verbose=False)

    # Dataloader
    train_dl = torch_geometric.data.DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    val_dl = torch_geometric.data.DataLoader(val_ds, batch_size=4, shuffle=False, drop_last=True)
    val_dl_dropout = torch_geometric.data.DataLoader(val_ds_dropout, batch_size=4, shuffle=False, drop_last=True)
    test_dl = torch_geometric.data.DataLoader(test_ds, batch_size=4, shuffle=False, drop_last=True)

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------

    # define model
    model = ModelExp5c().cuda()
    checkpoint = str(args.checkpoint)

    if not checkpoint:
        # load pretrained model weights
        weights_path = 'adjust/this/path'
        # get list of files
        files = os.listdir(weights_path)
        for file in files:
            if (file.endswith('_best.pt')) and (f'dropout_{dropout}' in file):
                weights_path = weights_path + file
                print(f'Loading model weights from: {weights_path}')
                break
        # load pretrained state dict
        checkpoint = torch.load(weights_path)
        pretrained_state_dict = checkpoint['model_state_dict']
        # update model state dict (strict=False -> mismatches in layer names are ignored (i.e. final layer))
        model.load_state_dict(pretrained_state_dict, strict=False)
        checkpoint = None # for compatibility with training function


    epochs = args.epochs
    lr = args.lr
    epochs_to_unfreeze = args.epochs_to_unfreeze

    model_save_path = 'adjust/this/path'
    model_name = f'ModelExp5c_lr{lr}_dropout_{dropout}_epochs_to_unfreeze{epochs_to_unfreeze}'
    print(
        f'Starting Finetuning with Learning rate: {lr} for {epochs} epochs (dropout: {dropout}, epochs to unfreeze: {epochs_to_unfreeze}). \n==============================================')
    model = train_model(model, model_save_path, model_name, train_dl, val_dl, val_dl_dropout, epochs, lr, epochs_to_unfreeze, checkpoint_path=checkpoint)
    print('Finished Training.')
