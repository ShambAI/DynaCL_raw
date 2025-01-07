import torch
from tqdm import tqdm
from src.datasets.harth import * 
from src.models.attention_model import *
from src.models.ts2vecencoder import *

import torch
import src.config, src.utils, src.models, src.hunt_data
from src.losses.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from src.loader.dataloader import HarDataset, ECGDataset, SleepDataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import wandb
import argparse
import random
import math
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split


from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import src.config, src.utils, src.models, src.data
import torchvision
from torch.utils.data import random_split

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_model(seed):
    set_seed(seed)
    model = TSEncoder(input_dims=args['feature_dim'], output_dims=args['out_features'])
    return model

def fit_lr(features, y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
        
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=0,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe

def main(train_loader, test_loader, seed, verbose=False):

    attn_model = initialize_model(seed)
    # Define loss function and optimizer

    if args['loss'] == 'HATCL_LOSS':
        cl_loss = HATCL_LOSS(temperature=args['temperature'])

    elif args['loss'] == 'LS_HATCL_LOSS':
        cl_loss = LS_HATCL_LOSS(temperature=args['temperature'])

    else:
        raise ValueError(f"Unsupported loss function: {args['loss']}")

    args['lr'] = float(args['lr'])

    if args['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    elif args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    elif args['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    elif args['optimizer'] == 'Adadelta':
        optimizer = torch.optim.Adadelta(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    else:
        optimizer = torch.optim.Adam(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.PATIENCE)



    # Wandb setup
    ds_name = os.path.realpath(ds_path).split('/')[-1]
    if config.WANDB:    
        proj_name = 'Dynamic_CL' + ds_name + str(seed)
        run_name = 'vanilla_cl'

        wandb_logger = WandbLogger(project=proj_name)
        
        # Initialize Wandb
        wandb.init(project=proj_name, name=run_name)
        wandb.watch(attn_model, log='all', log_freq=100)

        # Update Wandb config
        wandb.config.update(ds_args)
        wandb.config.update(args)
        wandb.config.update({
            'Algorithm': f'{run_name}',
            'Dataset': f'{ds_name}',
            'Train_DS_size': len(train_dataset),
            'Batch_Size': args["batch_size"],
            'Epochs': args["epochs"],
            'Patience': config.PATIENCE,
            'Seed': seed

        })
        wandb.run.name = run_name
        wandb.run.save()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_model.to(device)

    # Training and validation loop
    num_epochs = args['epochs']

    for epoch in tqdm(range(1, num_epochs+1)):
        # Training phase
        attn_model.train()  # Set the model to training mode
        train_running_loss = 0.0
        n_epoch_iters = 0

        for time_series, _ in train_loader:
            time_series = time_series.to(device) 

            # Forward pass
            out = attn_model(time_series)

            # Compute training loss
            features = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)

            features = features.squeeze(1)
        
            train_loss = cl_loss(features)

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Update training statistics
            n_epoch_iters += 1

            train_running_loss += train_loss.item()

       
        train_running_loss /= n_epoch_iters 

        if verbose:
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_running_loss:.4f}")

        # Log training loss to Wandb
        if config.WANDB:
            wandb.log({'Train Loss': train_running_loss, 'Epoch': epoch})


    attn_model.eval()  # Set the model to evaluation mode
    num_samples = len(train_loader.dataset)  # Total samples in the dataset
    data_tensor = torch.empty((num_samples, args['out_features']))
    label_tensor = torch.empty((num_samples,))

    # Counter to track position in preallocated tensors
    start_idx = 0

    for train_batch in train_loader:
        train_data, train_labels = train_batch
        
        # Move to device
        train_data = train_data.to(next(attn_model.parameters()).device)
        
        # Get representations
        with torch.no_grad():
            train_repr = attn_model(train_data)
        
        train_repr = F.max_pool1d(
                    train_repr.transpose(1, 2),
                    kernel_size = train_repr.size(1),
                ).transpose(1, 2)

        train_repr = train_repr.squeeze(1)
        
        # Get batch size
        batch_size = train_repr.size(0)
        
        # Fill preallocated tensors
        data_tensor[start_idx : start_idx + batch_size] = train_repr.cpu()
        label_tensor[start_idx : start_idx + batch_size] = train_labels.cpu()
        
        # Update index
        start_idx += batch_size



    #=======================================================================

    num_samples = len(test_loader.dataset)  # Total samples in the dataset
    test_data_tensor = torch.empty((num_samples, args['out_features']))
    test_label_tensor = torch.empty((num_samples,))

    # Counter to track position in preallocated tensors
    start_idx = 0

    for test_batch in test_loader:
        test_data, test_labels = test_batch
        
        # Move to device
        test_data = test_data.to(next(attn_model.parameters()).device)
        
        # Get representations
        with torch.no_grad():
            test_repr = attn_model(test_data)
        
        test_repr = F.max_pool1d(
                    test_repr.transpose(1, 2),
                    kernel_size = test_repr.size(1),
                ).transpose(1, 2)

        test_repr = test_repr.squeeze(1)
        
        # Get batch size
        batch_size = test_repr.size(0)
        
        # Fill preallocated tensors
        test_data_tensor[start_idx : start_idx + batch_size] = test_repr.cpu()
        test_label_tensor[start_idx : start_idx + batch_size] = test_labels.cpu()
        
        # Update index
        start_idx += batch_size

    
    clf = fit_lr(train_repr.detach().numpy(), train_labels)
    acc = clf.score(test_repr.detach().numpy(), test_labels)

    print(f"Accuracy: {acc:.4f} | batch: {args['batch_size']}, temp: {args['temperature']}, loss: {args['loss']}, "
      f"lr: {args['lr']}, feature_dim: {args['out_features']}, optimizer: {args['optimizer']}, seed: {args['seed']}")

    if epoch % config.SAVE_INTERVAL == 0:
        torch.save(attn_model.state_dict(), 
                f"models/{ds_name + str(seed)}_vanilla_epoch_{epoch}_b{args['batch_size']}_t{args['temperature']}_"
                f"l{args['loss']}_lr{args['lr']}_f{args['out_features']}_o{args['optimizer']}.pth")

    if config.WANDB:
        wandb.log({
            'Epoch': epoch,
            'Accuracy': round(acc, 2),
            'Final Loss': train_running_loss,
        })
    
    if config.WANDB:
        wandb.finish()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start Vanilla CL training.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='configs/sleepconfig.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default='data/sleepeeg')
    # parser.add_argument('-s', '--seed_value', required=False, type=int,
    #                     help='seed value.', default=42)
    
    parser.add_argument('-v', '--verbose_bool', required=False, type=bool,
                        help='verbose bool.', default=False)
    
    args = parser.parse_args()
    config_path = args.params_path
    # Read config
    config = src.config.Config(config_path)
    ds_path = args.dataset_path
    verbose = args.verbose_bool

    # Log in to Wandb
    if config.WANDB:
        wandb.login(key=config.WANDB_KEY)

    
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    
    for ds_args in src.utils.grid_search(config.DATASET_ARGS):
        # Iterate over all model configs if given
        for args in src.utils.grid_search(config.ALGORITHM_ARGS):
            
            seed = args['seed']
            # Set all seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # Create the dataset
            if config.DATASET == 'STFT':
                train_dataset = HarDataset(ds_path, window_size=ds_args['seq_length'], is_train=True)
                test_dataset = HarDataset(ds_path, window_size=ds_args['seq_length'], is_train=False)
                
            elif config.DATASET == 'ECG':
                train_dataset = ECGDataset(ds_path, is_train=True)
                test_dataset = ECGDataset(ds_path, is_train=False)

            elif config.DATASET == 'SLEEPEEG':
                train_dataset = SleepDataset(ds_path, is_train=True)
                test_dataset = SleepDataset(ds_path, is_train=False)

            else:
                raise ValueError(f"Unsupported DATASET: {config.DATASET}")
            
        
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args['batch_size'],
                # sampler=SequentialRandomSampler(train_ds, args['batch_size']),
                shuffle = True,
                num_workers=config.NUM_WORKERS,
                drop_last = True,
            )
            
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=args['batch_size'],
                # sampler=SequentialRandomSampler(valid_ds, args['batch_size']),
                shuffle = False,
                num_workers=config.NUM_WORKERS,
                drop_last = True,
            )
    
            main(train_loader, test_loader, seed, verbose)