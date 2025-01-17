import torch
from tqdm import tqdm
from src.datasets.harth import * 
from src.models.attention_model import *
from src.models.ts2vecencoder import *
import torch
from src.utils import take_per_row
import src.config, src.utils, src.models, src.hunt_data
from src.losses.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from src.loader.dataloader import SequentialRandomSampler, FlattenedDataset, STFTDataset, SLEEPDataset, KpiDataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from torch.nn import GRU, Linear, CrossEntropyLoss
import wandb
import argparse
import random
import math
from src.losses.contrastive import hierarchical_contrastive_loss
import torch.nn.functional as F



from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import src.config, src.utils, src.models, src.data
import torchvision
from torch.utils.data import random_split

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

def load_UCR(dataset):
    
    train_file = os.path.join('./data/UCRArchive_2018/', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('./data/UCRArchive_2018/', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels

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

def load_balanced_dataset(dataset, class_counts):
    # Initialize dictionary to store indices of each class
    class_indices = {label: [] for label in class_counts.keys()}
    
    # Populate class_indices with indices of each class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = label.item()  # Ensure the label is a scalar
        if label in class_indices:
            class_indices[label].append(idx)

    # Ensure each class has the required number of instances
    balanced_indices = []
    for label, count in class_counts.items():
        if len(class_indices[label]) >= count:
            balanced_indices.extend(random.sample(class_indices[label], count))
        else:
            raise ValueError(f"Not enough instances of class {label} to satisfy the requested count")

    # Create a subset of the dataset with the balanced indices
    balanced_subset = Subset(dataset, balanced_indices)
    return balanced_subset

# Function to apply t-SNE and visualize the results
def visualize_tsne(images, labels, class_names, model):

    # Evaluate the model and get features
    model.eval()
    with torch.no_grad():
        model_features = model(images)

    # Standardize the data before applying t-SNE
    scaler = StandardScaler()
    tsne = TSNE(n_components=2, init='random', learning_rate='auto')

    # Standardize model features before applying t-SNE
    standardized_model_features = scaler.fit_transform(model_features.view(-1, model_features.size(-1)).cpu().numpy())

    # Apply t-SNE to model features
    reduced_features_model = tsne.fit_transform(standardized_model_features)

    # Plot the results
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        indices = labels == i
        plt.scatter(reduced_features_model[indices, 0], reduced_features_model[indices, 1], label=class_names[i])

    plt.title(f't-SNE Visualization of TS2Vec Algorithm Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()

    return plt, model_features

def main(train_loader, seed):

    attn_model = initialize_model(seed)
    # Define loss function and optimizer

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
    if config.WANDB:
        ds_name = os.path.realpath(ds_path).split('/')[-1]
        proj_name = 'Dynamic_CL' + ds_name + str(seed)
        run_name = 'ts2vec'

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
            'Train_DS_size': len(dataset),
            'Batch_Size': args["batch_size"],
            'Epochs': args["epochs"],
            'Patience': config.PATIENCE,
            'Seed': seed

        })
        # Explicitly save the run
        wandb.run.name = run_name
        wandb.run.save()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_model.to(device)

    # Training and validation loop
    num_epochs = args['epochs']

    temporal_unit = 0
    max_train_length = 500
    for epoch in tqdm(range(1, num_epochs+1)):
        # Training phase
        attn_model.train()  # Set the model to training mode
        train_running_loss = 0.0

        for batch_idx, x in enumerate(train_loader):

            if max_train_length is not None and x.size(1) > max_train_length:
                window_offset = np.random.randint(x.size(1) - max_train_length + 1)
                x = x[:, window_offset : window_offset + max_train_length]
            x = x.to(device) 

            
            ts_l = x.size(1)
            crop_l = np.random.randint(low=2 ** (temporal_unit + 1), high=ts_l+1)
            crop_left = np.random.randint(ts_l - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
            crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

            x1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
            x2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)

            out1 = attn_model(x1)
            out2 = attn_model(x2)

            padding_size = abs(x2.size(1) - x1.size(1))  # Difference in the second dimension

            if out2.size(1) > out1.size(1):
                out1 = F.pad(out1, (0, 0, 0, padding_size))
            elif out1.size(1) > out2.size(1):
                out2 = F.pad(out2, (0, 0, 0, padding_size))

            # Calculate the loss
            train_loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=temporal_unit
                )

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Update training statistics
            train_running_loss += train_loss.item() * x.size(0)

        


        train_epoch_loss = train_running_loss / len(train_loader.dataset)
        # print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}")

        # Log training loss to Wandb
        if config.WANDB and batch_idx % 10 == 0:
            wandb.log({'Train Loss': train_epoch_loss, 'Epoch': epoch})
            
    
    torch.save(attn_model.feature_extractor.state_dict(), f'models/{ds_name + str(seed)}_ts2vec_model_epoch_{epoch}.pth')

    
    if config.WANDB:
        wandb.finish()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start Vanilla CL training.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='configs/ucr_config.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default='ECG5000')
    parser.add_argument('-s', '--seed_value', required=False, type=int,
                        help='seed value.', default=42)
    args = parser.parse_args()
    config_path = args.params_path
    # Read config
    config = src.config.Config(config_path)
    ds_path = args.dataset_path
    seed = int(args.seed_value)
    # Log in to Wandb
    # if config.WANDB:
    #     wandb.login(key=config.WANDB_KEY)

    # Set all seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    
    for ds_args in src.utils.grid_search(config.DATASET_ARGS):
        # Iterate over all model configs if given
        for args in src.utils.grid_search(config.ALGORITHM_ARGS):
            # Create the dataset
           
            if config.DATASET == 'UCR':
                dataset, _, _, _ = load_UCR(ds_path)

            else:
                raise ValueError(f"Unsupported DATASET: {config.DATASET}")

            
            
            
            train_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=args['batch_size'],
                shuffle = False,
                num_workers=config.NUM_WORKERS,
                drop_last = True,
            )
            


    main(train_loader, seed)