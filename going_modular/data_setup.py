import logging
import torch
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset
from torchvision import datasets, transforms
import random

# Normalization values for the different datasets
NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar10': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'cifar100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
}

# Configure a logger for this module if needed, or rely on root logger
logger = logging.getLogger(__name__)

def splitting_dataset(dataset, nb_clients):
    # Check if dataset is a Subset, if so, get the underlying dataset and indices
    original_dataset = dataset
    subset_indices = None
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        subset_indices = dataset.indices

    random.seed(42)
    
    # Group data by class, considering subset indices if applicable
    class_data = {}
    if subset_indices:
        logger.debug(f"Splitting subset with {len(subset_indices)} indices.")
        for idx in subset_indices:
            data, target = original_dataset[idx] # Access original dataset with subset index
            if target not in class_data:
                class_data[target] = []
            class_data[target].append((data, target))
    else: # Handle full dataset case
        logger.debug(f"Splitting full dataset with {len(dataset)} samples.")
        for data, target in dataset:
             if target not in class_data:
                 class_data[target] = []
             class_data[target].append((data, target))
    
    # Initialize client datasets
    clients_dataset = [[[], []] for _ in range(nb_clients)]
    
    # First pass: distribute samples from each class evenly
    for class_label, samples in class_data.items():
        samples = samples.copy()
        random.shuffle(samples)
        
        # Ensure each client gets at least one sample from each class if possible
        if len(samples) >= nb_clients:
            for client_idx in range(nb_clients):
                data, target = samples.pop()
                clients_dataset[client_idx][0].append(data)
                clients_dataset[client_idx][1].append(target)
        
        # Distribute remaining samples round-robin
        while samples:
            for client_idx in range(nb_clients):
                if not samples:
                    break
                data, target = samples.pop()
                clients_dataset[client_idx][0].append(data)
                clients_dataset[client_idx][1].append(target)
    
    # Verify and print distribution
    logger.info("\nOverall dataset distribution:")
    total_distributed = sum(len(x) for x, _ in clients_dataset)
    logger.info(f"Total samples distributed: {total_distributed}")
    if nb_clients > 0: logger.info(f"Average samples per client: {total_distributed/nb_clients:.2f}")
    logger.info(f"Number of clients: {nb_clients}")
    
    # Print per-client distribution
    logger.info("\nPer-client distribution:")
    empty_clients = []
    for i, (x, y) in enumerate(clients_dataset):
        if len(x) == 0:
            empty_clients.append(i)
            continue
            
        class_dist = {}
        for label in y:
            class_dist[label] = class_dist.get(label, 0) + 1
        logger.info(f"Client {i}:")
        logger.info(f"  Total samples: {len(x)}")
        logger.info(f"  Number of classes: {len(class_dist)}")
    
    # Handle empty clients by redistributing data
    if empty_clients:
        logger.warning(f"\nWarning: Found {len(empty_clients)} empty clients. Redistributing data...")
        for empty_client in empty_clients:
            # Find client with most samples
            max_client = max(range(nb_clients), 
                           key=lambda x: len(clients_dataset[x][0]) if x not in empty_clients else -1)
            
            # Transfer half of the samples
            n_transfer = len(clients_dataset[max_client][0]) // 2
            if n_transfer > 0: # Only transfer if there are samples
                clients_dataset[empty_client][0] = clients_dataset[max_client][0][:n_transfer]
                clients_dataset[empty_client][1] = clients_dataset[max_client][1][:n_transfer]
                clients_dataset[max_client][0] = clients_dataset[max_client][0][n_transfer:]
                clients_dataset[max_client][1] = clients_dataset[max_client][1][n_transfer:]
                logger.info(f"Transferred {n_transfer} samples from client {max_client} to client {empty_client}")
            else:
                 logger.warning(f"Cannot redistribute from client {max_client} as it has too few samples.")
    
    # Final verification
    client_sizes = [len(x) for x, _ in clients_dataset]
    size_difference = max(client_sizes) - min(client_sizes) if client_sizes else 0
    if size_difference > nb_clients:
        logger.warning(f"\nWarning: Uneven distribution detected. Size difference: {size_difference}")
        logger.warning(f"Client sizes: {client_sizes}")
    
    return clients_dataset

def load_dataset(name_dataset="cifar", data_root="./data/", number_of_clients=4, number_of_nodes=3, resize=None, max_samples_per_set=None):
    data_folder = f"{data_root}/{name_dataset}"

    list_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(**NORMALIZE_DICT[name_dataset])
    ]
    if resize is not None:
        list_transforms = [transforms.Resize((resize, resize))] + list_transforms
            
    transform = transforms.Compose(list_transforms)

    if name_dataset == "cifar10":
        full_dataset_train = datasets.CIFAR10(data_folder, train=True, download=True, transform=transform)
        full_dataset_test = datasets.CIFAR10(data_folder, train=False, download=True, transform=transform)
    elif name_dataset == "cifar100":
        full_dataset_train = datasets.CIFAR100(data_folder, train=True, download=True, transform=transform)
        full_dataset_test = datasets.CIFAR100(data_folder, train=False, download=True, transform=transform)
    elif name_dataset == "mnist":
        full_dataset_train = datasets.MNIST(data_folder, train=True, download=True, transform=transform)
        full_dataset_test = datasets.MNIST(data_folder, train=False, download=True, transform=transform)
    else:
        raise ValueError("The dataset name is not correct")
    
    logger.info(f"Loaded full {name_dataset} dataset: Train={len(full_dataset_train)}, Test={len(full_dataset_test)}")

    # --- Subsetting Logic ---
    if max_samples_per_set is not None and max_samples_per_set > 0:
        logger.info(f"Subsetting datasets to max {max_samples_per_set} samples each.")

        # Subset training data
        num_train_samples = len(full_dataset_train)
        train_indices = list(range(num_train_samples))
        random.shuffle(train_indices) # Shuffle before taking subset
        # Take the minimum of requested size and available size
        actual_train_subset_size = min(max_samples_per_set, num_train_samples)
        train_indices = train_indices[:actual_train_subset_size]
        dataset_train = Subset(full_dataset_train, train_indices) # Use torch Subset
        logger.info(f"Using {len(dataset_train)} training samples (subset).")

        # Subset testing data
        num_test_samples = len(full_dataset_test)
        test_indices = list(range(num_test_samples))
        random.shuffle(test_indices) # Shuffle before taking subset
        # Take the minimum of requested size and available size
        actual_test_subset_size = min(max_samples_per_set, num_test_samples)
        test_indices = test_indices[:actual_test_subset_size]
        dataset_test = Subset(full_dataset_test, test_indices) # Use torch Subset
        logger.info(f"Using {len(dataset_test)} testing samples (subset).")
    else:
        # Use the full datasets if no limit is specified or limit is invalid
        logger.info("Using full datasets (no subsetting applied).")
        dataset_train = full_dataset_train
        dataset_test = full_dataset_test
    # --- End Subsetting Logic ---

    # Split the (potentially subsetted) datasets
    client_train_sets = splitting_dataset(dataset_train, number_of_clients)
    client_test_sets = splitting_dataset(dataset_test, number_of_clients)
    node_test_sets = splitting_dataset(dataset_test, number_of_nodes) # Nodes also use the (subsetted) test set

    # Get classes from the original full dataset to ensure all classes are known
    classes = full_dataset_train.classes if hasattr(full_dataset_train, 'classes') else list(range(10)) # Fallback for MNIST

    return client_train_sets, client_test_sets, node_test_sets, classes

class Data(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)
