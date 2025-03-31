import time
import threading
import random
import logging
import json
# Add necessary imports for MNIST
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset # Added Subset

from pbft_node import PBFTNode
from pbft_client import PBFTClient

from going_modular.utils import initialize_parameters
# We might not use load_dataset from data_setup anymore
# from going_modular.data_setup import load_dataset
from config import settings

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import os
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PBFT-Network")

def send_concurrent_requests(clients, operations):
    """Send requests from multiple clients concurrently"""
    threads = []
    for i, (client, operation) in enumerate(zip(clients, operations)):
        thread = threading.Thread(target=client.send_request, args=(operation,))
        threads.append(thread)
        logger.info(f"Starting thread {i} for operation: {operation}")
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    logger.info("All concurrent requests sent")

def simulate_primary_failure(nodes):
    """Simulate primary node failure and trigger view change"""
    logger.info("Simulating primary node failure...")
    
    # Find the current primary node
    primary_node = None
    for node in nodes:
        if node.pbft.is_primary_node():
            primary_node = node
            break
    
    if not primary_node:
        logger.error("No primary node found!")
        return None
    
    logger.info(f"Current primary is node {primary_node.node_id}")
    
    # Stop the primary node
    primary_node.running = False
    primary_node.server_socket.close()
    logger.info(f"Primary node {primary_node.node_id} stopped")
    
    # Explicitly trigger view change on all other nodes
    current_view = primary_node.pbft.view
    new_view = current_view + 1
    logger.info(f"Triggering view change to view {new_view} on all nodes")
    
    # Create a view-change message
    view_change_msgs = []
    
    for node in nodes:
        if node != primary_node and node.running:
            # Force the node to initiate a view change
            try:
                # Create a view-change message
                view_change_msg = {
                    'type': 'view-change',
                    'new_view': new_view,
                    'last_seq': node.pbft.sequence_number,
                    'sender': node.node_id,
                    'prepared': {}  # Simplified for this example
                }
                view_change_msgs.append(view_change_msg)
                
                # Process it locally
                node.pbft.process_message(view_change_msg)
                logger.info(f"Triggered view change on node {node.node_id}")
            except Exception as e:
                logger.error(f"Error triggering view change on node {node.node_id}: {e}")
    
    # Broadcast all view-change messages to all nodes
    for node in nodes:
        if node != primary_node and node.running:
            for msg in view_change_msgs:
                try:
                    node.pbft.process_message(msg)
                except Exception as e:
                    logger.error(f"Error processing view-change on node {node.node_id}: {e}")
    
    # Wait for view change to complete
    logger.info("Waiting for view change to complete...")
    time.sleep(10)
    
    # Find the new primary
    new_primary = None
    for node in nodes:
        if node != primary_node and node.running and node.pbft.is_primary_node():
            new_primary = node
            break
    
    if new_primary:
        logger.info(f"New primary is node {new_primary.node_id}")
    else:
        logger.warning("No new primary node found after view change")
    
    return primary_node

def add_new_node(nodes, next_node_id, test_set, host='127.0.0.1', base_port=8000):
    """Add a new node to the PBFT network"""
    # Create node configuration
    port = base_port + next_node_id
    node_config = {
        'id': next_node_id,
        'host': host,
        'port': port,
        'test_set': test_set
    }
    
    # Get all existing node configs
    all_node_configs = []
    for node in nodes:
        if node.running:  # Only include running nodes
            all_node_configs.extend(node.nodes)
    
    # Remove duplicates
    unique_configs = []
    seen_ids = set()
    for config in all_node_configs:
        if config['id'] not in seen_ids:
            unique_configs.append(config)
            seen_ids.add(config['id'])
    
    # Fix: Pass the test_set parameter to the PBFTNode constructor
    node = PBFTNode(next_node_id, host, port, unique_configs, test_set)
    
    # Find the current primary node
    primary_node = None
    current_view = 0
    for n in nodes:
        if n.running and n.pbft.is_primary_node():
            primary_node = n
            current_view = n.pbft.view
            break
    
    # Add the new node to all existing nodes
    for existing_node in nodes:
        if existing_node.running:
            existing_node.add_node(next_node_id, host, port)
    
    # Explicitly sync the view from the primary
    if primary_node:
        logger.info(f"Syncing view {current_view} from primary node {primary_node.node_id} to new node {next_node_id}")
        view_sync = {
            'type': 'view-sync',
            'sender': primary_node.node_id,
            'view': current_view,
            'primary': primary_node.node_id
        }
        primary_node.send_message({'id': next_node_id, 'host': host, 'port': port}, view_sync)
        
        # Wait a bit for the view sync to take effect
        time.sleep(1)
    
    logger.info(f"Started new node {next_node_id} on {host}:{port}")
    
    return node, next_node_id + 1

def main():
    logging.basicConfig(level=logging.DEBUG)
    # We might not need training_barrier, length from initialize_parameters
    # training_barrier, length = initialize_parameters(settings)
    # print(training_barrier, length)

    num_nodes = settings['number_of_nodes']
    num_clients = 9
    base_port = 10000

    # --- MNIST Data Loading ---
    logger.info("Loading MNIST dataset...")
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization
    ])

    try:
        # Use data_root from settings for consistency
        data_root = settings.get('data_root', './data') # Default to ./data if not set
        os.makedirs(data_root, exist_ok=True) # Ensure data root exists
        full_train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=mnist_transform)
        full_test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=mnist_transform)
        logger.info(f"MNIST loaded: {len(full_train_dataset)} train, {len(full_test_dataset)} test images.")
    except Exception as e:
        logger.error(f"Failed to download or load MNIST dataset: {e}")
        logger.error("Please check your internet connection and data_root path in config.")
        return

    # --- Simple Data Splitting (Non-IID Example) ---
    # Creates disjoint subsets for each client/node. Replace with more sophisticated splitting if needed.
    client_train_sets = []
    client_test_sets = []
    node_test_sets = []

    train_indices = list(range(len(full_train_dataset)))
    test_indices = list(range(len(full_test_dataset)))
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    samples_per_client_train = len(train_indices) // num_clients
    samples_per_client_test = len(test_indices) // num_clients
    samples_per_node_test = len(test_indices) // num_nodes # Simple split for nodes

    logger.info(f"Splitting data: ~{samples_per_client_train} train/client, ~{samples_per_client_test} test/client, ~{samples_per_node_test} test/node")

    all_train_data = []
    all_test_data = []
    for i in range(num_clients):
        start_idx = i * samples_per_client_train
        end_idx = start_idx + samples_per_client_train if i < num_clients - 1 else len(train_indices) # Handle remainder
        client_train_subset_indices = train_indices[start_idx:end_idx]
        # It's often better to pass indices or Subsets to clients if they handle DataLoader internally
        # But PBFTClient currently expects tensors, so we load them here. This can be memory intensive.
        client_train_subset = Subset(full_train_dataset, client_train_subset_indices)
        # Load all data from subset into memory (adjust batch_size if memory becomes an issue)
        temp_loader = DataLoader(client_train_subset, batch_size=len(client_train_subset))
        x_train_client, y_train_client = next(iter(temp_loader))
        client_train_sets.append((x_train_client, y_train_client))
        all_train_data.append(len(client_train_subset_indices))


        start_idx = i * samples_per_client_test
        end_idx = start_idx + samples_per_client_test if i < num_clients - 1 else len(test_indices) # Handle remainder
        client_test_subset_indices = test_indices[start_idx:end_idx]
        client_test_subset = Subset(full_test_dataset, client_test_subset_indices)
        temp_loader = DataLoader(client_test_subset, batch_size=len(client_test_subset))
        x_test_client, y_test_client = next(iter(temp_loader))
        client_test_sets.append((x_test_client, y_test_client))
        all_test_data.append(len(client_test_subset_indices))

    # Create Node Test sets
    current_test_idx = 0
    for i in range(num_nodes):
        num_samples = samples_per_node_test
        if i == num_nodes - 1: # Give remainder to last node
             num_samples = len(test_indices) - current_test_idx
        end_idx = current_test_idx + num_samples
        node_test_subset_indices = test_indices[current_test_idx:end_idx]
        node_test_subset = Subset(full_test_dataset, node_test_subset_indices)
        temp_loader = DataLoader(node_test_subset, batch_size=len(node_test_subset))
        x_test_node, y_test_node = next(iter(temp_loader))
        node_test_sets.append((x_test_node, y_test_node))
        current_test_idx = end_idx

    logger.info(f"Data split: Train sizes={all_train_data}, Test sizes={all_test_data}")
    # --- End MNIST Data Loading & Splitting ---

    # Ensure we have enough data splits
    if len(client_train_sets) < num_clients or len(client_test_sets) < num_clients or len(node_test_sets) < num_nodes:
        logger.error(f"Insufficient data splits after processing MNIST.")
        return

    nodes_config = []
    for i in range(num_nodes):
        nodes_config.append({
            'id': i,
            'host': 'localhost',
            'port': base_port + i
        })
    
    # Start nodes
    nodes = []
    for i in range(num_nodes):
        node = PBFTNode(
            node_id=i,
            host='localhost',
            port=base_port + i,
            nodes_config=nodes_config,
            test_set=node_test_sets[i] # Pass the split node test set
        )
        nodes.append(node)
        logger.info(f"Started node {i} on port {base_port + i}")
    
    # Give nodes time to start AND for the primary to create the initial global model
    logger.info("Waiting for nodes to start and initial global model creation (v1)...")
    time.sleep(5)
    logger.info("Initial wait finished.")
    
    # Create clients
    clients = []
    for i in range(num_clients):
        # Use the pre-loaded tensors
        train_data_tensors = client_train_sets[i]
        test_data_tensors = client_test_sets[i]
        logger.info(f"Client {i} using {len(train_data_tensors[0])} training samples and {len(test_data_tensors[0])} test samples.")
        client = PBFTClient(
            client_id=f"client{i}", 
            nodes_config=nodes_config,
            client_train_set=train_data_tensors,
            client_test_set=test_data_tensors
            )
        clients.append(client)
    logger.info(f"Created {len(clients)} clients.")
    
    # Start a timer to periodically check for censored requests
    def start_censorship_check():
        while True:
            try:
                for node in nodes:
                    if node.running and not node.pbft.is_primary_node():
                        node.check_for_censored_requests()
            except Exception as e:
                logger.error(f"Error in censorship check: {e}")
            time.sleep(10)  # Check every 10 seconds
    
    censorship_thread = threading.Thread(target=start_censorship_check)
    censorship_thread.daemon = True
    censorship_thread.start()
    
    try:
        next_node_id = num_nodes
        cluster_size = 3
        last_processed_global_model_version = 0

        # --- Helper Function for Cluster Training ---
        def run_cluster_training(cluster_clients, cluster_id, base_global_model_version):
            client_ids = [c.client_id for c in cluster_clients]
            logger.info(f"--- Starting Cluster Training [Cluster {cluster_id}, Base v{base_global_model_version}] Clients: {client_ids} ---")

            threads = []
            results = {} # Use thread-safe collection if needed

            def train_client_thread(client):
                logger.info(f"[Cluster {cluster_id}] Starting training thread for {client.client_id} based on v{base_global_model_version}")
                weights, loss, accuracy, version_used = client.train(send_update=False, return_weights=True)

                # Important: Verify the client actually got the intended global model version
                if version_used != base_global_model_version:
                     logger.warning(f"[Cluster {cluster_id}] Mismatch! Client {client.client_id} trained on v{version_used} "
                                    f"but expected v{base_global_model_version}. Aborting this client's result.")
                     results[client.client_id] = None # Treat as failure
                elif weights is not None:
                    results[client.client_id] = {
                        'weights': weights, 'loss': loss, 'accuracy': accuracy, 'version_used': version_used
                    }
                    logger.info(f"[Cluster {cluster_id}] Training thread finished for {client.client_id} (v{version_used}). Acc: {accuracy*100:.2f}%")
                else:
                    results[client.client_id] = None # Indicate failure
                    logger.warning(f"[Cluster {cluster_id}] Training thread failed for {client.client_id}")

            for client in cluster_clients:
                thread = threading.Thread(target=train_client_thread, args=(client,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            logger.info(f"--- Cluster Training Finished [Cluster {cluster_id}] ---")

            # Check results and aggregate
            successful_weights = {}
            all_successful_in_cluster = True
            for client_id in client_ids:
                result = results.get(client_id)
                if result:
                    successful_weights[client_id] = result['weights']
                    logger.info(f"  [Cluster {cluster_id}] Client {client_id}: Success (v{result['version_used']})")
                else:
                    logger.warning(f"  [Cluster {cluster_id}] Client {client_id}: Failed or Version Mismatch")
                    all_successful_in_cluster = False

            if all_successful_in_cluster and successful_weights:
                logger.info(f"[Cluster {cluster_id}] All clients trained successfully. Aggregating...")
                # --- Simulate Aggregation (Simple Averaging) ---
                aggregated_weights = {}
                first_client_weights = list(successful_weights.values())[0]
                for key in first_client_weights:
                     aggregated_weights[key] = np.zeros_like(first_client_weights[key], dtype=np.float64)
                num_clients_in_agg = len(successful_weights)
                for client_weights in successful_weights.values():
                     for key in aggregated_weights:
                         if key in client_weights:
                             aggregated_weights[key] += client_weights[key].astype(np.float64)
                for key in aggregated_weights:
                     aggregated_weights[key] /= num_clients_in_agg
                     try: # Cast back
                         original_dtype = first_client_weights[key].dtype
                         if original_dtype != np.float64: aggregated_weights[key] = aggregated_weights[key].astype(original_dtype)
                     except Exception as e: logger.warning(f"Could not cast key {key}: {e}")

                # --- Save Aggregated Model ---
                agg_dir = "models/aggregated"
                os.makedirs(agg_dir, exist_ok=True)
                timestamp = int(time.time())
                agg_model_filename = f"cluster_{cluster_id}_v{base_global_model_version}_aggregated_{timestamp}.npz"
                agg_model_path = os.path.join(agg_dir, agg_model_filename)
                np.savez(agg_model_path, **aggregated_weights)
                logger.info(f"[Cluster {cluster_id}] Saved aggregated model to: {agg_model_path}")

                # Calculate hash
                agg_model_hash = ""
                with open(agg_model_path, 'rb') as f: agg_model_hash = hashlib.sha256(f.read()).hexdigest()

                # --- Send Validation Request ---
                client_ids_str = json.dumps(client_ids)
                operation = (f"CLUSTER_TRAIN_VALIDATE cluster_id={cluster_id} "
                             f"aggregated_model_path='{agg_model_path}' "
                             f"aggregated_model_hash='{agg_model_hash}' "
                             f"client_ids='{client_ids_str}' "
                             f"global_model_version={base_global_model_version}") # Use the base version

                logger.info(f"[Cluster {cluster_id}] Sending validation request: {operation}")
                # Use the first client of the *entire simulation* (client 0) to send the request
                clients[0].send_request(operation)
                time.sleep(2) # Small delay after sending request
            else:
                logger.warning(f"[Cluster {cluster_id}] Training incomplete or failed. No aggregation or validation request sent.")
        # --- End Helper Function ---

        while True:
            print("\nPBFT Blockchain Network Menu:")
            print("1. Add a new element to the chain (Manual SET/GET/DELETE)")
            print("2. Check if all nodes have the same state")
            print("3. View state of a specific node")
            print("4. Send concurrent requests from multiple clients")
            print("5. View blockchain of a specific node")
            print("6. Compare blockchains across nodes")
            print("7. Save blockchain to file")
            # Option 8 (original Exit) is removed
            print("9. Simulate primary node failure")
            print("10. Add a new node to the network")
            print("11. Simulate selective censorship")
            print("12. Train clients (Individual) - Note: Uses client 0") # Clarified which client
            print("13. Check for New Model & Trigger Random Cluster Training") # Updated description
            print("14. Exit")
            
            choice = input(f"Enter your choice (1-14): ")
            
            if choice == '1':
                operation_type = input("Enter operation type (SET/GET/DELETE): ").upper()
                if operation_type in ["SET", "GET", "DELETE"]:
                    if operation_type == "SET":
                        key = input("Enter key: ")
                        value = input("Enter value: ")
                        operation = f"SET {key} {value}"
                    else:
                        key = input("Enter key: ")
                        operation = f"{operation_type} {key}"
                    
                    clients[0].send_request(operation)
                    print(f"Request sent: {operation}")
                    time.sleep(2)  # Wait for propagation
                else:
                    print("Invalid operation type. Please use SET, GET, or DELETE.")
            
            elif choice == '2':
                # Check if all nodes have the same state
                states = []
                for i, node in enumerate(nodes):
                    states.append((i, node.get_state()))
                
                print("\nState comparison across nodes:")
                reference_state = states[0][1]
                all_same = True
                
                for node_id, state in states:
                    is_same = (state == reference_state)
                    if not is_same:
                        all_same = False
                    print(f"Node {node_id}: {'SAME' if is_same else 'DIFFERENT'}")
                
                if all_same:
                    print("\n✅ All nodes have the same state - Consensus achieved!")
                else:
                    print("\n❌ Nodes have different states - Consensus failed!")
                
                print("\nDetailed state for each node:")
                for node_id, state in states:
                    print(f"Node {node_id}: {state}")
            
            elif choice == '3':
                node_id = input("Enter node ID (0-3): ")
                try:
                    node_id = int(node_id)
                    if 0 <= node_id < len(nodes):
                        print(f"\nState of Node {node_id}:")
                        print(nodes[node_id].get_state())
                        print(f"Last executed sequence: {nodes[node_id].last_executed_seq}")
                        print(f"Is primary: {nodes[node_id].is_primary}")
                    else:
                        print(f"Invalid node ID. Please enter a number between 0 and {len(nodes)-1}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '4':
                # Send concurrent requests from multiple clients
                num_requests = input("Enter number of concurrent requests (2-4): ")
                try:
                    num_requests = int(num_requests)
                    if 2 <= num_requests <= 4:
                        concurrent_operations = []
                        for i in range(num_requests):
                            print(f"\nRequest {i+1}:")
                            operation_type = input("Enter operation type (SET/GET/DELETE): ").upper()
                            if operation_type in ["SET", "GET", "DELETE"]:
                                if operation_type == "SET":
                                    key = input("Enter key: ")
                                    value = input("Enter value: ")
                                    operation = f"SET {key} {value}"
                                else:
                                    key = input("Enter key: ")
                                    operation = f"{operation_type} {key}"
                                concurrent_operations.append(operation)
                            else:
                                print("Invalid operation type. Skipping this request.")
                        
                        if concurrent_operations:
                            print("\nSending concurrent requests...")
                            send_concurrent_requests(clients[:len(concurrent_operations)], concurrent_operations)
                            print("Concurrent requests sent. Waiting for processing...")
                            time.sleep(3)  # Wait for processing
                    else:
                        print("Please enter a number between 2 and 4.")
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '5':
                # View blockchain of a specific node
                node_id = input("Enter node ID (0-3): ")
                try:
                    node_id = int(node_id)
                    if 0 <= node_id < len(nodes):
                        blockchain = nodes[node_id].get_blockchain()
                        print(f"\nBlockchain of Node {node_id}:")
                        print(f"Chain length: {blockchain.len_chain}")
                        
                        view_details = input("View detailed block information? (y/n): ").lower()
                        if view_details == 'y':
                            blockchain.print_blockchain()
                        else:
                            # Print summary
                            for i, block in enumerate(blockchain.blocks):
                                operations_count = len(block.data) if isinstance(block.data, list) else 1
                                print(f"Block #{i}: {operations_count} operations, hash: {block.current_hash[:10]}...")
                    else:
                        print(f"Invalid node ID. Please enter a number between 0 and {len(nodes)-1}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '6':
                # Compare blockchains across nodes
                blockchains = []
                for i, node in enumerate(nodes):
                    blockchains.append((i, node.get_blockchain()))
                
                print("\nBlockchain comparison across nodes:")
                reference_chain = blockchains[0][1]
                all_same = True
                
                for node_id, blockchain in blockchains:
                    # Compare chain length
                    length_same = (blockchain.len_chain == reference_chain.len_chain)
                    
                    # Compare last block hash
                    last_block_same = False
                    if length_same and blockchain.len_chain > 0:
                        last_block_same = (blockchain.blocks[-1].current_hash == 
                                          reference_chain.blocks[-1].current_hash)
                    
                    is_same = length_same and last_block_same
                    if not is_same:
                        all_same = False
                    
                    print(f"Node {node_id}: {'SAME' if is_same else 'DIFFERENT'} " +
                          f"(Length: {blockchain.len_chain}, " +
                          f"Last block hash: {blockchain.blocks[-1].current_hash[:10]}...)")
                
                if all_same:
                    print("\n✅ All nodes have the same blockchain - Consensus achieved!")
                else:
                    print("\n❌ Nodes have different blockchains - Consensus failed!")
            
            elif choice == '7':
                # Save blockchain to file
                node_id = input("Enter node ID to save blockchain from (0-3): ")
                try:
                    node_id = int(node_id)
                    if 0 <= node_id < len(nodes):
                        filename = input("Enter filename to save to: ")
                        if not filename:
                            filename = f"blockchain_node_{node_id}.txt"
                        
                        blockchain = nodes[node_id].get_blockchain()
                        blockchain.save_chain_to_file(filename)
                        print(f"Blockchain saved to {filename}")
                        
                        # Also save as JSON
                        json_filename = f"{filename.split('.')[0]}.json"
                        with open(json_filename, "w") as f:
                            f.write(blockchain.to_json())
                        print(f"Blockchain also saved as JSON to {json_filename}")
                    else:
                        print(f"Invalid node ID. Please enter a number between 0 and {len(nodes)-1}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '9':
                failed_primary = simulate_primary_failure(nodes)
                if failed_primary is not None:
                    print(f"Primary node {failed_primary.node_id} has been stopped. Waiting for view change...")
                    time.sleep(10)  # Wait for view change to occur
                    
                    # Check new primary
                    new_primary = None
                    for i, node in enumerate(nodes):
                        if i != failed_primary.node_id and node.pbft.is_primary_node():
                            new_primary = i
                            break
                    
                    if new_primary is not None:
                        print(f"View change successful! New primary is node {new_primary}")
                    else:
                        print("View change may not have completed yet")
                else:
                    print("Could not identify primary node")
            
            elif choice == '10':
                new_node, next_node_id = add_new_node(nodes, next_node_id, test_set=node_test_sets[0])
                nodes.append(new_node)
                print(f"Added new node with ID {new_node.node_id}")
            
            elif choice == '11':
                # Find the primary node
                primary_node = None
                for node in nodes:
                    if node.pbft.is_primary_node():
                        primary_node = node
                        break
                
                if primary_node:
                    print(f"Making primary node {primary_node.node_id} selectively malicious")
                    
                    # Create a list to track which requests to censor
                    primary_node.censored_keys = []
                    
                    # Ask which key to censor
                    key_to_censor = input("Enter a key that the primary should censor (e.g., 'key1'): ")
                    primary_node.censored_keys.append(key_to_censor)
                    
                    # Override the handle_request method
                    def selective_malicious_handle_request(self, message):
                        operation = message.get('operation', '')
                        request_id = message.get('request_id', '')
                        
                        # Check if this operation contains a censored key
                        should_censor = False
                        for censored_key in self.censored_keys:
                            if censored_key in operation:
                                should_censor = True
                                break
                        
                        if should_censor:
                            self.logger.info(f"MALICIOUS PRIMARY: Selectively censoring request {request_id} containing {self.censored_keys}")
                            # Still store the request but don't process it
                            self.pbft.request_log[request_id] = message
                            if not hasattr(self.pbft, 'request_timestamps'):
                                self.pbft.request_timestamps = {}
                            self.pbft.request_timestamps[request_id] = time.time()
                            
                            # Make sure all backup nodes also know about this request
                            for node in nodes:
                                if node.node_id != self.node_id and node.running:
                                    if not hasattr(node.pbft, 'request_timestamps'):
                                        node.pbft.request_timestamps = {}
                                    node.pbft.request_timestamps[request_id] = time.time()
                                    node.pbft.request_log[request_id] = message
                        
                            return
                        else:
                            # Process normally
                            self.logger.info(f"MALICIOUS PRIMARY: Processing non-censored request {request_id}")
                            # Call the original method
                            self.pbft.handle_request(message)
                    
                    import types
                    primary_node.handle_request = types.MethodType(selective_malicious_handle_request, primary_node)
                    
                    print(f"Primary will now censor requests containing '{key_to_censor}'")
                    print("Send requests with and without this key to test selective censorship detection")
                else:
                    print("Could not find primary node")
            
            elif choice == '12':
                # Train a single client (e.g., client 0) - Kept for simple testing
                client_to_train = clients[0]
                print(f"Starting training for {client_to_train.client_id}...")
                # Use send_update=True here if you want individual updates to also trigger aggregation
                model_path, loss, accuracy, version_used = client_to_train.train(send_update=True)
                if model_path:
                    print(f"Training complete for {client_to_train.client_id} (based on v{version_used}).")
                    print(f"  Model saved to: {model_path}")
                    print(f"  Final Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")
                    # Note: Update request was sent automatically by client.train if send_update=True
                else:
                    print(f"Training failed for {client_to_train.client_id}.")
                time.sleep(2) # Allow time for update request propagation
            
            elif choice == '13': # Check for new model and trigger training
                logger.info("Checking for new global model version...")
                current_global_model_version = -1
                try:
                    # Query node 0's state for the global model info
                    # Ensure node 0 is running before querying
                    if nodes and nodes[0].running:
                        node_to_query = nodes[0]
                        node_state = node_to_query.get_state()
                        if 'global_model' in node_state:
                            model_info_str = node_state['global_model']
                            model_info = json.loads(model_info_str)
                            current_global_model_version = model_info.get('version', -1)
                            logger.info(f"Node 0 reported current global model version: {current_global_model_version}")
                        else:
                            logger.warning("Node 0 state does not contain 'global_model' info yet.")
                    else:
                         logger.error("Node 0 is not running or not available.")

                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding global model JSON from Node 0 state: {e}")
                except Exception as e:
                    logger.error(f"Error querying node 0 for global model version: {e}")

                # Compare with the last version we processed
                if current_global_model_version > 0 and current_global_model_version > last_processed_global_model_version:
                    logger.info(f"✅ New global model version detected: v{current_global_model_version} "
                                f"(previously processed v{last_processed_global_model_version}). Triggering cluster training.")
                    last_processed_global_model_version = current_global_model_version

                    # --- Randomize Client Order for Cluster Formation ---
                    logger.info("Shuffling clients for random cluster formation...")
                    clients_copy = clients[:] # Create a copy to shuffle
                    random.shuffle(clients_copy)
                    logger.debug(f"Shuffled client order: {[c.client_id for c in clients_copy]}")
                    # --- End Randomization ---

                    # Trigger training for all clusters based on this new version
                    num_clusters = len(clients_copy) // cluster_size # Use the shuffled list length
                    logger.info(f"Attempting to train {num_clusters} clusters of size {cluster_size}.")
                    for i in range(num_clusters):
                        start_index = i * cluster_size
                        end_index = start_index + cluster_size
                        # Use the shuffled list for slicing
                        cluster_clients_for_run = clients_copy[start_index:end_index]

                        if len(cluster_clients_for_run) == cluster_size:
                             # Run training for this cluster sequentially
                             logger.info(f"--- Starting run_cluster_training for Cluster {i} (Clients: {[c.client_id for c in cluster_clients_for_run]}) ---")
                             run_cluster_training(cluster_clients_for_run,
                                                  cluster_id=i,
                                                  base_global_model_version=current_global_model_version)
                             logger.info(f"--- Finished run_cluster_training for Cluster {i} ---")
                             time.sleep(1) # Small delay between starting clusters
                        else:
                             # This condition might be less likely if num_clients is a multiple of cluster_size
                             logger.warning(f"Could not form cluster {i}, only {len(cluster_clients_for_run)} clients available in slice [{start_index}:{end_index}].")

                    logger.info(f"All cluster training rounds initiated for global model v{current_global_model_version}.")

                elif current_global_model_version == -1:
                     logger.info("Could not determine current global model version from Node 0.")
                else:
                    logger.info(f"No new global model version detected. Current version v{current_global_model_version} is not newer than last processed v{last_processed_global_model_version}.")

            elif choice == '14': # Renumbered Exit
                break

            else:
                # Adjust the invalid choice message
                print(f"Invalid choice. Please enter a number between 1 and 14.")
    
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down PBFT network...")
        for node in nodes:
            node.stop()

if __name__ == "__main__":
    main() 