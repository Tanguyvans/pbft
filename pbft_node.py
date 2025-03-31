import socket
import threading
import json
import time
import hashlib
import os
from typing import Dict, List
import logging
import queue
import traceback
import numpy as np
import torch
import torch.nn as nn
import re

from going_modular.model import Net
from flowerclient import FlowerClient

from blockchain import Blockchain
from block import Block
from pbft import PBFT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PBFTNode:
    def __init__(self, node_id: int, host: str, port: int, nodes_config: List[Dict], test_set):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.nodes = nodes_config
        self.logger = logging.getLogger(f"Node-{self.node_id}")
        
        # Initialize the PBFT consensus protocol
        self.pbft = PBFT(node_id, len(nodes_config), self)
        
        # Node state
        self.state = {}  # Simple key-value store as the state
        self.last_executed_seq = 0
        self.execution_queue = queue.PriorityQueue()  # Queue for ordered execution
        self.executed_requests = set()  # Track executed request IDs

        x_test, y_test = test_set

        self.flower_client = FlowerClient.node(
            x_test=x_test, 
            y_test=y_test
        )
        
        # Initialize blockchain with a deterministic genesis block
        self.blockchain = Blockchain()
        
        # Locks for thread safety
        self.state_lock = threading.Lock()
        self.blockchain_lock = threading.Lock()
        
        # Start server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        
        self.running = True
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Start execution thread
        self.execution_thread = threading.Thread(target=self.process_execution_queue)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        
        self.logger.info(f"Node {self.node_id} started on {self.host}:{self.port}")

        # Add a longer delay before the primary node creates the initial global model
        if self.pbft.is_primary_node():
            def delayed_initial_model():
                # Create second genesis block with initial global model
                self.logger.info(f"Primary node {self.node_id} creating initial global model")
                
                # Store request in PBFT module with proper digest
                request_id = "initial_global_model"
                operation = "CREATE_GLOBAL_MODEL"
                digest = hashlib.sha256(f"{request_id}:{operation}".encode()).hexdigest()
                
                request = {
                    'type': 'request',
                    'client_id': "system",
                    'timestamp': int(time.time() * 1000),
                    'operation': operation,
                    'digest': digest,
                    'request_id': request_id
                }
                
                # First make sure all nodes are aware of this request
                self.broadcast(request)
                self.pbft.start_consensus(request_id)
                
                self.logger.info(f"Primary status: {self.node_id} is the primary node")
            
            # Schedule the creation of the initial model after a longer delay
            threading.Timer(3.0, delayed_initial_model).start()
        
        # Model aggregation parameters
        self.update_threshold = 3  # Number of updates before aggregation
        
        # Track model updates
        self.pending_updates = []  # List of validated updates since last aggregation
        self.global_model_version = 1  # Current global model version
    
    def start_server(self):
        """Accept incoming connections and handle them in separate threads"""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket, addr))
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                self.logger.error(f"Error accepting connection: {e}")
                if not self.running:
                    break
    
    def handle_client(self, client_socket, addr):
        """Handle incoming messages from clients or other nodes"""
        try:
            data = b""
            while self.running:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                try:
                    # Try to parse the message
                    message = json.loads(data.decode('utf-8'))
                    self.process_message(message)
                    data = b""
                except json.JSONDecodeError:
                    # Incomplete message, continue receiving
                    continue
        except Exception as e:
            self.logger.error(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()
    
    def send_message(self, target_node: Dict, message: Dict):
        """Send a message to a specific node"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)  # Set a timeout for connection attempts
            s.connect((target_node['host'], target_node['port']))
            s.sendall(json.dumps(message).encode('utf-8'))
            s.close()
        except ConnectionRefusedError:
            # More concise error for expected failures
            if not hasattr(self, 'failed_nodes'):
                self.failed_nodes = set()
            
            # Only log the first time we detect a node is down
            if target_node['id'] not in self.failed_nodes:
                self.logger.warning(f"Node {target_node['id']} appears to be down")
                self.failed_nodes.add(target_node['id'])
        except Exception as e:
            self.logger.error(f"Error sending message to {target_node['id']}: {e}")
    
    def broadcast(self, message: Dict, exclude_self=False):
        """Broadcast a message to all nodes"""
        for node in self.nodes:
            if exclude_self and node['id'] == self.node_id:
                continue
            self.send_message(node, message)
    
    def process_message(self, message: Dict):
        """Process incoming messages based on their type"""
        msg_type = message.get('type')
        
        if msg_type == 'request':
            self.handle_request(message)
        elif msg_type in ['pre-prepare', 'prepare', 'commit', 'view-change', 'new-view', 'heartbeat']:
            self.pbft.process_message(message)
        elif msg_type == 'block-sync':
            self.handle_block_sync(message)
        elif msg_type == 'state-sync':
            self.handle_state_sync(message)
        elif msg_type == 'view-sync':
            self.handle_view_sync(message)
        elif msg_type == 'node-join':
            self.handle_node_join(message)
        elif msg_type == 'model-request':
            self.handle_model_request(message)
        elif msg_type == 'validation-result':
            self.handle_validation_result(message)
        elif msg_type == 'validation-failed':
            self.handle_validation_failed(message)
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")
    
    def handle_request(self, message: Dict):
        """Handle client request"""
        self.logger.info(f"Received client request: {message}")
        
        # Store the request
        client_id = message.get('client_id')
        timestamp = message.get('timestamp')
        operation = message.get('operation')
        request_id = f"{client_id}:{timestamp}"
        
        # Check if this is a model update and extract version information
        if operation.startswith('UPDATE_MODEL '):
            # Get the global model version this update is based on
            global_model_version = message.get('global_model_version', 1)
            
            # Add version to the operation string if not already there
            if ' v' not in operation:
                operation = f"{operation} v{global_model_version}"
                message['operation'] = operation
        
        # Calculate request digest (including cluster validation requests)
        request_data = f"{client_id}:{timestamp}:{operation}"
        digest = hashlib.sha256(request_data.encode()).hexdigest()
        
        # Store request in PBFT module
        self.pbft.store_request(request_id, {
            'client_id': client_id,
            'timestamp': timestamp,
            'operation': operation,
            'digest': digest
        })
        
        # If this node is the primary, initiate the PBFT protocol
        if self.pbft.is_primary_node():
            self.pbft.start_consensus(request_id)
    
    def execute_operation(self, sequence: int, request: Dict):
        """Execute the operation and update the state"""
        operation = request.get('operation', '')
        request_id = request.get('request_id', '')
        view = request.get('view', 0)
        validation_key = f"v{view}-s{sequence}"
        
        self.logger.info(f"Executing operation for seq {sequence}, view {view}: {operation}")
        
        # Check if this operation has been validated
        if hasattr(self, 'validated_operations') and validation_key in self.validated_operations:
            if not self.validated_operations[validation_key]:
                self.logger.warning(f"Operation for {validation_key} failed validation, skipping execution")
                return "VALIDATION_FAILED"
        
        # For operations that need validation but haven't been validated yet
        if (operation.startswith('UPDATE_MODEL ') or operation.startswith('SET ')) and (not hasattr(self, 'validated_operations') or validation_key not in self.validated_operations):
            # Store for later execution after validation
            if not hasattr(self, 'pending_validations'):
                self.pending_validations = {}
            
            self.logger.info(f"Operation for {validation_key} waiting for validation")
            self.pending_validations[validation_key] = request
            
            # Trigger validation check
            self.validate_operation(sequence, request)
            return "PENDING_VALIDATION"
        
        # Check if this is a duplicate global model update
        if operation.startswith('UPDATE_GLOBAL_MODEL '):
            parts = operation.split(' ', 3)
            if len(parts) >= 4:
                version = int(parts[3])
                
                # Check if we've already processed this version
                with self.state_lock:
                    try:
                        current_model_info = json.loads(self.state.get('global_model', '{}'))
                        current_version = current_model_info.get('version', 0)
                        
                        # If we've already processed this version or a newer one, skip
                        if current_version >= version:
                            self.logger.warning(f"Skipping duplicate global model update for version {version}, current version is {current_version}")
                            return f"SKIPPED_DUPLICATE: Global model v{version} already processed"
                    except:
                        pass
        
        result = None
        state_changed = False
        
        try:
            if operation.startswith('UPDATE_MODEL '):
                # Format: UPDATE_MODEL model_path model_hash loss accuracy [vX]
                parts = operation.split(' ')
                if len(parts) >= 5:
                    model_path = parts[1]
                    model_hash = parts[2]
                    training_loss = float(parts[3])
                    training_accuracy = float(parts[4])
                    
                    # Extract version information if present
                    update_version = 1  # Default version
                    for part in parts[5:]:
                        if part.startswith('v') and part[1:].isdigit():
                            update_version = int(part[1:])
                            break
                    
                    # Extract client ID from request
                    client_id = request.get('client_id', 'unknown')
                    
                    # Verify the model file exists
                    if not os.path.exists(model_path):
                        self.logger.error(f"Model file not found: {model_path}")
                        result = f"ERROR: Model file not found: {model_path}"
                    else:
                        # Verify the model hash
                        with open(model_path, 'rb') as f:
                            file_content = f.read()
                            actual_hash = hashlib.sha256(file_content).hexdigest()
                        
                        if actual_hash != model_hash:
                            self.logger.warning(f"Model hash verification failed!")
                            self.logger.warning(f"Expected: {model_hash}")
                            self.logger.warning(f"Actual: {actual_hash}")
                            result = "ERROR: Model hash verification failed"
                        else:
                            # The model is already in NPZ format, no need to convert
                            
                            # Create model metadata for blockchain
                            model_update = {
                                'type': 'model_update',
                                'client_id': client_id,
                                'timestamp': int(time.time()),
                                'storage_path': model_path,
                                'hash': model_hash,
                                'training_loss': training_loss,
                                'training_accuracy': training_accuracy * 100,  # Multiply by 100 for clarity
                                'version': update_version,  # Add version information
                                'based_on_global': self.global_model_version  # Track which global model this is based on
                            }
                            
                            # Add to state
                            with self.state_lock:
                                # Store in a list of model updates
                                if 'model_updates' not in self.state:
                                    self.state['model_updates'] = []
                                
                                self.state['model_updates'].append(model_update)
                                result = f"MODEL_UPDATED: {client_id}, accuracy: {training_accuracy * 100:.2f}%, version: v{update_version}"
                                state_changed = True
                                self.logger.info(f"Added model update v{update_version} from client {client_id} to blockchain")
                                
                                # Add to pending updates for aggregation, grouped by version
                                update_info = {
                                    'client_id': client_id,
                                    'model_path': model_path,
                                    'model_hash': model_hash,
                                    'training_loss': training_loss,
                                    'training_accuracy': training_accuracy,
                                    'timestamp': int(time.time()),
                                    'version': update_version
                                }
                                
                                # Group updates by version
                                if not hasattr(self, 'pending_updates_by_version'):
                                    self.pending_updates_by_version = {}
                                
                                if update_version not in self.pending_updates_by_version:
                                    self.pending_updates_by_version[update_version] = []
                                
                                self.pending_updates_by_version[update_version].append(update_info)
                                
                                # Also add to the regular pending updates list for backward compatibility
                                self.pending_updates.append(update_info)
                                
                                # Log the update count for this version
                                version_updates = len(self.pending_updates_by_version[update_version])
                                self.logger.info(f"Added model update to pending list for version v{update_version}. Current count: {version_updates}/{self.update_threshold}")
                                
                                # Check if we should trigger aggregation for this version
                                if version_updates >= self.update_threshold:
                                    if self.pbft.is_primary_node():
                                        self.logger.info(f"Threshold reached for version v{update_version}! Triggering model aggregation with {version_updates} updates")
                                        # Use threading to avoid blocking the execution queue
                                        aggregation_thread = threading.Thread(target=self.aggregate_models, args=(update_version,))
                                        aggregation_thread.daemon = True
                                        aggregation_thread.start()
                                    else:
                                        self.logger.info(f"Threshold reached for version v{update_version} but this is not the primary node. Waiting for primary to initiate aggregation.")
                else:
                    result = "ERROR: Invalid UPDATE_MODEL format"
            
            elif operation == 'CREATE_GLOBAL_MODEL':
                self.logger.info("Creating initial global model")
                
                # Create models directory if it doesn't exist
                models_dir = "models"
                npz_dir = "models/npz"
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                if not os.path.exists(npz_dir):
                    os.makedirs(npz_dir)
                
                # Create model file using NPZ format
                timestamp = int(time.time())
                model_filename = f"global_model_v1_{timestamp}.npz"
                model_path = os.path.join(npz_dir, model_filename)
                
                # Get the model's state dict
                state_dict = self.flower_client.model.state_dict()
                
                # Convert PyTorch tensors to numpy arrays
                numpy_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
                
                # Save the model weights
                np.savez(model_path, **numpy_dict)
                time.sleep(0.1) # Add a small delay to ensure file write completes
                
                # Calculate hash of the model file
                model_hash = ""
                with open(model_path, 'rb') as f: # Ensure file is properly closed after reading
                    model_hash = hashlib.sha256(f.read()).hexdigest()
                
                self.logger.info(f"Global model saved to {model_path} with hash {model_hash}")
                
                # Create model metadata for blockchain (separate from the model file)
                model_data = {
                    'type': 'initial_model',
                    'version': 1,
                    'created_by': f"node-{self.node_id}",
                    'timestamp': timestamp,
                    'storage_path': model_path,
                    'hash': model_hash,
                    'architecture': 'mobilenet_v2',
                    'num_classes': 10
                }
                
                with self.state_lock:
                    self.state['global_model'] = json.dumps(model_data)
                    result = "GLOBAL_MODEL_CREATED"
                    state_changed = True
                    self.logger.info(f"Created initial global model: {model_data}")
            
            elif operation.startswith('SET '):
                parts = operation.split(' ', 2)  # Split only on the first two spaces
                
                if len(parts) == 3:
                    key, value = parts[1], parts[2]
                    
                    with self.state_lock:
                        self.state[key] = value
                        self.logger.info(f"SET {key} = {value}")
                        result = f"SET {key} = {value}"
                        state_changed = True
                        self.logger.info(f"State changed: {state_changed}")
                
            elif operation.startswith('GET '):
                parts = operation.split(' ', 2)  # Split only on the first two spaces
                
                if len(parts) == 2:
                    key = parts[1]
                    with self.state_lock:
                        value = self.state.get(key, "NULL")
                        self.logger.info(f"GET {key} = {value}")
                        result = f"GET {key} = {value}"
                    # GET operations don't modify state, so no new block needed
                
            elif operation.startswith('DELETE '):
                parts = operation.split(' ', 2)  # Split only on the first two spaces
                
                if len(parts) == 2:
                    key = parts[1]
                    with self.state_lock:
                        if key in self.state:
                            del self.state[key]
                            self.logger.info(f"DELETE {key}")
                            result = f"DELETE {key} = SUCCESS"
                            state_changed = True
                            self.logger.info(f"State changed: {state_changed}")
                        else:
                            result = f"DELETE {key} = KEY_NOT_FOUND"
            elif operation.startswith('UPDATE_GLOBAL_MODEL '):
                # Format: UPDATE_GLOBAL_MODEL model_path model_hash version
                parts = operation.split(' ', 3)
                if len(parts) >= 4:
                    model_path = parts[1]
                    model_hash = parts[2]
                    version = int(parts[3])
                    
                    # Verify the model file exists
                    if not os.path.exists(model_path):
                        self.logger.error(f"Global model file not found: {model_path}")
                        result = f"ERROR: Global model file not found: {model_path}"
                    else:
                        # Verify the model hash
                        with open(model_path, 'rb') as f:
                            file_content = f.read()
                            actual_hash = hashlib.sha256(file_content).hexdigest()
                        
                        if actual_hash != model_hash:
                            self.logger.warning(f"Global model hash verification failed!")
                            self.logger.warning(f"Expected: {model_hash}")
                            self.logger.warning(f"Actual: {actual_hash}")
                            result = "ERROR: Global model hash verification failed"
                        else:
                            # Get the full model data from the request
                            model_data = request.get('model_data', {})
                            if not model_data:
                                # If model_data not in request, create basic metadata
                                model_data = {
                                    'type': 'aggregated_model',
                                    'version': version,
                                    'created_by': request.get('client_id', f"node-{self.node_id}"),
                                    'timestamp': int(time.time()),
                                    'storage_path': model_path,
                                    'hash': model_hash,
                                    'architecture': 'mobilenet_v2',
                                    'num_classes': 10
                                }
                            
                            # Update state with new global model
                            with self.state_lock:
                                old_model_info = None
                                try:
                                    old_model_info = json.loads(self.state.get('global_model', '{}'))
                                except:
                                    pass
                                
                                self.state['global_model'] = json.dumps(model_data)
                                
                                # Update local tracking
                                self.global_model_version = version
                                self.pending_updates = []  # Clear pending updates
                                
                                result = f"GLOBAL_MODEL_UPDATED: version {version}"
                                state_changed = True
                                
                                # Log the update
                                old_version = old_model_info.get('version', 0) if old_model_info else 0
                                self.logger.info(f"Updated global model from v{old_version} to v{version}")
            elif operation.startswith('CLUSTER_TRAIN_VALIDATE '):
                # --- Use Regex for Robust Parsing ---
                # Pattern to find key='value' or key=number pairs
                # \w+ matches the key (word characters)
                # = matches the equals sign
                # (?:...) is a non-capturing group for alternation
                # '[^']*' matches anything inside single quotes (non-greedy)
                # \d+ matches one or more digits (for IDs/versions)
                pattern = r"(\w+)=(?:'([^']*)'|(\d+))"
                
                params = {}
                try:
                    # Find all key=value pairs after the initial command
                    command_part, args_part = operation.split(' ', 1)
                    for match in re.finditer(pattern, args_part):
                        key = match.group(1)
                        # Value can be in group 2 (string) or group 3 (number)
                        value = match.group(2) if match.group(2) is not None else match.group(3)
                        params[key] = value
                except ValueError: # Handle case where there are no args after command
                    self.logger.error("Could not split operation into command and args.")
                    params = {} # Ensure params is empty dict
                except Exception as e:
                    self.logger.error(f"Error parsing CLUSTER_TRAIN_VALIDATE args with regex: {e}")
                    params = {} # Ensure params is empty dict


                cluster_id = None
                agg_model_path = None
                agg_model_hash = None
                client_ids_list = None
                global_model_version = None

                try:
                    cluster_id = int(params.get('cluster_id')) if 'cluster_id' in params else None
                    agg_model_path = params.get('aggregated_model_path')
                    agg_model_hash = params.get('aggregated_model_hash')
                    client_ids_str = params.get('client_ids')
                    global_model_version = int(params.get('global_model_version')) if 'global_model_version' in params else None

                    if client_ids_str:
                        client_ids_list = json.loads(client_ids_str)
                    else:
                         self.logger.warning("client_ids string not found in parsed params.")


                except (TypeError, ValueError, json.JSONDecodeError) as e:
                    self.logger.error(f"Error converting parsed CLUSTER_TRAIN_VALIDATE params: {e}")
                    # Set potentially problematic variables back to None
                    if isinstance(e, (TypeError, ValueError)):
                         if 'cluster_id' not in params or not isinstance(params.get('cluster_id'), str) or not params.get('cluster_id').isdigit(): cluster_id = None
                         if 'global_model_version' not in params or not isinstance(params.get('global_model_version'), str) or not params.get('global_model_version').isdigit(): global_model_version = None
                    if isinstance(e, json.JSONDecodeError): client_ids_list = None


                # --- Validation and Processing (using parsed values) ---
                if cluster_id is not None and agg_model_path and agg_model_hash and client_ids_list and global_model_version is not None:
                    self.logger.info(f"Processing cluster validation request for Cluster ID: {cluster_id}, based on global v{global_model_version}")
                    self.logger.info(f"  Aggregated Model Path: {agg_model_path}")
                    self.logger.info(f"  Aggregated Model Hash: {agg_model_hash}")
                    self.logger.info(f"  Contributing Clients: {client_ids_list}")

                    # --- Perform Validation ---
                    is_cluster_valid = False
                    if not os.path.exists(agg_model_path):
                        self.logger.error(f"Validation FAILED: Aggregated model file not found: {agg_model_path}")
                        result = f"ERROR: Aggregated model file not found: {agg_model_path}"
                    else:
                         # Verify hash
                         with open(agg_model_path, 'rb') as f:
                             actual_hash = hashlib.sha256(f.read()).hexdigest()
                         if actual_hash != agg_model_hash:
                             self.logger.warning(f"Validation FAILED: Aggregated model hash mismatch!")
                             self.logger.warning(f"  Expected: {agg_model_hash}")
                             self.logger.warning(f"  Actual:   {actual_hash}")
                             result = "ERROR: Aggregated model hash mismatch"
                         else:
                             self.logger.info("✅ Aggregated model hash verified.")
                             is_cluster_valid = True

                    # --- If Valid, Add to Pending Aggregations ---
                    if is_cluster_valid:
                        update_info = {
                            'type': 'cluster_aggregation', 
                            'client_id': f"cluster_{cluster_id}", 
                            'model_path': agg_model_path,
                            'model_hash': agg_model_hash,
                            'client_ids': client_ids_list, 
                            'training_loss': None, 
                            'training_accuracy': None,
                            'timestamp': int(time.time()),
                            'version': global_model_version 
                        }
                        with self.state_lock:
                            if not hasattr(self, 'pending_updates_by_version'): self.pending_updates_by_version = {}
                            if global_model_version not in self.pending_updates_by_version: self.pending_updates_by_version[global_model_version] = []
                            self.pending_updates_by_version[global_model_version].append(update_info)
                            self.pending_updates.append(update_info)
                            version_updates = len(self.pending_updates_by_version[global_model_version])
                            self.logger.info(f"Added validated cluster result (based on v{global_model_version}) to pending list. "
                                             f"Current count for v{global_model_version}: {version_updates}/{self.update_threshold}")
                            if 'cluster_validations' not in self.state: self.state['cluster_validations'] = {}
                            state_key = f"cluster_{cluster_id}_seq_{sequence}"
                            self.state['cluster_validations'][state_key] = {
                                'status': 'VALIDATED', 
                                'aggregated_model_path': agg_model_path,
                                'aggregated_model_hash': agg_model_hash,
                                'client_ids': client_ids_list,
                                'based_on_version': global_model_version,
                                'timestamp': time.time()
                            }
                            result = f"CLUSTER_VALIDATED: cluster_id={cluster_id}, added to pending aggregation v{global_model_version}"
                            state_changed = True

                        # Check if we should trigger aggregation
                        if version_updates >= self.update_threshold:
                            if self.pbft.is_primary_node():
                                self.logger.info(f"Threshold reached for version v{global_model_version}! Triggering model aggregation including cluster result.")
                                aggregation_thread = threading.Thread(target=self.aggregate_models, args=(global_model_version,))
                                aggregation_thread.daemon = True
                                aggregation_thread.start()
                            else:
                                self.logger.info(f"Threshold reached for v{global_model_version} but not primary. Waiting for primary.")
                        
                    # else: result is already set by the failing validation check

                else: # Parsing failed or essential parameters missing
                    result = f"ERROR: Invalid CLUSTER_TRAIN_VALIDATE format or missing essential parameters. Parsed params: {params}"
                    self.logger.error(result)
            else:
                self.logger.warning(f"Unknown operation format: {operation}")
                result = f"UNKNOWN_OPERATION: {operation}"
        except Exception as e:
            self.logger.error(f"Error executing operation: {e}")
            result = f"ERROR: {str(e)}"
        
        # Create a new block for state-changing operations (including CLUSTER_TRAIN_VALIDATE)
        if state_changed:
            self.logger.info(f"Creating block for operation: {operation}, sequence: {sequence}, view: {view}")
            with self.blockchain_lock:
                # Create a unique block ID that includes both view and sequence
                block_id = f"v{view}-s{sequence}"
                
                # Check if a block for this ID already exists
                block_exists = False
                for block in self.blockchain.blocks:
                    if block.data.get('block_id') == block_id:
                        block_exists = True
                        self.logger.info(f"Block for {block_id} already exists, skipping creation")
                        break
                
                if not block_exists:
                    if self.pbft.is_primary_node():
                        self.logger.info(f"Primary node creating block for {block_id}")
                    
                    # --- Prepare data for the block ---
                    block_data = {
                            'operation': operation,
                            'sequence': sequence,
                            'view': view,
                            'block_id': block_id,
                            'request_id': request_id,
                            'result': result,
                    }

                    # --- Conditionally add specific data and state snapshots ---
                    if operation.startswith('CLUSTER_TRAIN_VALIDATE '):
                        # Add the list of client IDs that formed this cluster
                        if 'client_ids_list' in locals() and client_ids_list is not None:
                            block_data['cluster_client_ids'] = client_ids_list
                            self.logger.debug(f"Adding client IDs to block {block_id}: {client_ids_list}")
                        else:
                            # This case shouldn't happen if parsing worked, but good to handle
                            self.logger.warning(f"Could not find client_ids_list when creating block for {block_id}, adding empty list.")
                            block_data['cluster_client_ids'] = []
                        # Include only the relevant part of the state for clusters
                        block_data['state_snapshot'] = { 'cluster_validations': self.state.get('cluster_validations', {}).copy() }
                        block_model_type = "cluster-validation-request"
                    elif operation.startswith('UPDATE_MODEL '):
                        # Potentially add model update specific info if needed later
                        block_data['state_snapshot'] = self.state.copy() # Or specific parts if state grows large
                        block_model_type = "model-update"
                    elif operation == 'CREATE_GLOBAL_MODEL' or operation.startswith('UPDATE_GLOBAL_MODEL '):
                        block_data['state_snapshot'] = {'global_model': self.state.get('global_model')} # Only global model info
                        block_model_type = "global-model-creation-or-update"
                    else: # Default for SET/GET/DELETE etc.
                        block_data['state_snapshot'] = self.state.copy() # Full state snapshot
                        block_model_type = "key-value-operation"


                    # --- Create the block ---
                    new_block = self.blockchain.create_block(
                        data=block_data, # Use the prepared data dictionary
                        model_type=block_model_type,
                        storage_reference=f"op-{block_id}",
                        calculated_hash=hashlib.sha256(json.dumps(block_data, sort_keys=True).encode()).hexdigest(), # Hash the actual block data
                        participants=[str(self.node_id)]
                    )
                    self.blockchain.add_block(new_block)
                    self.logger.info(f"Created block #{new_block.index} for {block_id}, hash: {new_block.current_hash[:10]}...")

                    # If this node is the primary, broadcast the new block to all nodes
                    if self.pbft.is_primary_node():
                        self.logger.info(f"Broadcasting block #{new_block.index} to all nodes")
                        block_sync = {
                            'type': 'block-sync',
                            'sender': self.node_id,
                            'block': new_block.to_dict(),
                            'sequence': sequence,
                            'view': view
                        }
                        self.broadcast(block_sync)
        
        self.logger.info(f"Current state: {self.state}")
        return result
    
    def process_execution_queue(self):
        """Process the execution queue in order"""
        while self.running:
            try:
                # Check if there's anything to execute
                if self.execution_queue.empty():
                    time.sleep(0.1)
                    continue
                
                # Get the next item from the queue
                seq, request = self.execution_queue.get()
                
                # Get the view for this request
                view = request.get('view', self.pbft.view)
                
                # Check if we've already executed this request
                request_id = request.get('request_id', '')
                if request_id and request_id in self.executed_requests:
                    self.logger.info(f"Request {request_id} already executed, skipping")
                    continue
                
                # Check if we've already executed this sequence in this view
                # We need to track executed sequences per view
                view_seq_key = f"v{view}-s{seq}"
                if hasattr(self, 'executed_view_seqs') and view_seq_key in self.executed_view_seqs:
                    self.logger.info(f"Sequence {seq} in view {view} already executed, skipping")
                    continue
                
                # For global model updates, check if we've already processed this version
                operation = request.get('operation', '')
                if operation.startswith('UPDATE_GLOBAL_MODEL '):
                    parts = operation.split(' ', 3)
                    if len(parts) >= 4:
                        version = int(parts[3])
                        
                        # Check if we've already processed this version
                        with self.state_lock:
                            try:
                                current_model_info = json.loads(self.state.get('global_model', '{}'))
                                current_version = current_model_info.get('version', 0)
                                
                                # If we've already processed this version or a newer one, skip
                                if current_version >= version:
                                    self.logger.warning(f"Skipping duplicate global model update for version {version}, current version is {current_version}")
                                    
                                    # Mark as executed to avoid reprocessing
                                    self.last_executed_seq = seq
                                    if request_id:
                                        self.executed_requests.add(request_id)
                                    
                                    # Track executed sequences per view
                                    if not hasattr(self, 'executed_view_seqs'):
                                        self.executed_view_seqs = set()
                                    self.executed_view_seqs.add(view_seq_key)
                                    
                                    continue
                            except:
                                pass
                
                # Execute the operation
                result = self.execute_operation(seq, {
                    'operation': operation,
                    'request_id': request_id,
                    'view': view
                })
                
                # Mark as executed
                self.last_executed_seq = seq
                if request_id:
                    self.executed_requests.add(request_id)
                
                # Track executed sequences per view
                if not hasattr(self, 'executed_view_seqs'):
                    self.executed_view_seqs = set()
                self.executed_view_seqs.add(view_seq_key)
                
            except Exception as e:
                self.logger.error(f"Error processing execution queue: {e}")
                time.sleep(0.1)
    
    def handle_block_sync(self, message: Dict):
        """Handle block synchronization from the primary node"""
        sender = message.get('sender')
        block_data = message.get('block')
        sequence = message.get('sequence')
        
        # Only accept blocks from the primary node
        primary_id = self.pbft.view % len(self.nodes)
        if sender != primary_id and not self.pbft.is_primary_node():
            self.logger.warning(f"Ignoring block from non-primary node {sender}")
            return
        
        self.logger.info(f"Received block sync from primary node {sender} for sequence {sequence}")
        
        # Check if we already have a block with this index
        block_index = block_data.get('index')
        with self.blockchain_lock:
            if block_index < len(self.blockchain.blocks):
                existing_block = self.blockchain.blocks[block_index]
                if existing_block.current_hash == block_data.get('current_hash'):
                    self.logger.info(f"Block #{block_index} already exists with same hash, skipping")
                    return
        
        # Create a new block with the exact same properties
        new_block = Block(
            index=block_data.get('index'),
            data=block_data.get('data'),
            model_type=block_data.get('model_type'),
            storage_reference=block_data.get('storage_reference'),
            calculated_hash=block_data.get('calculated_hash'),
            participants=block_data.get('participants'),
            previous_hash=block_data.get('previous_hash')
        )
        
        # Set the exact same timestamp and nonce to ensure identical hash
        new_block.timestamp = block_data.get('timestamp')
        new_block.nonce = block_data.get('nonce')
        
        # Add or replace the block in our blockchain
        with self.blockchain_lock:
            if block_index < len(self.blockchain.blocks):
                self.blockchain.blocks[block_index] = new_block
                self.logger.info(f"Replaced block #{block_index} with block from primary, hash: {new_block.current_hash[:10]}...")
            else:
                # Make sure we're adding blocks in order
                if block_index == len(self.blockchain.blocks):
                    self.blockchain.blocks.append(new_block)
                    self.logger.info(f"Added block #{block_index} from primary, hash: {new_block.current_hash[:10]}...")
                else:
                    self.logger.warning(f"Received out-of-order block #{block_index}, expected {len(self.blockchain.blocks)}")
            
            # Apply the operation from the block
            operation_data = new_block.data
            if operation_data:
                with self.state_lock:
                    op_type = operation_data.get('type')
                    key = operation_data.get('key')
                    
                    if op_type == 'SET':
                        value = operation_data.get('value')
                        self.state[key] = value
                        self.logger.info(f"Applied SET {key} = {value} from block")
                    
                    elif op_type == 'DELETE':
                        if key in self.state:
                            del self.state[key]
                            self.logger.info(f"Applied DELETE {key} from block")
    
    def get_state(self):
        """Return a copy of the current state"""
        with self.state_lock:
            return self.state.copy()
    
    def get_blockchain(self):
        """Return the blockchain"""
        with self.blockchain_lock:
            return self.blockchain
    
    def stop(self):
        """Stop the node"""
        self.running = False
        self.pbft.cleanup()  # Clean up PBFT timers
        self.server_socket.close()
        self.logger.info(f"Node {self.node_id} stopped")
    
    def on_consensus_reached(self, sequence: int, request: Dict):
        """Called by PBFT when consensus is reached for a request"""
        request_id = request.get('request_id', '')
        operation = request.get('operation', '')
        
        self.logger.info(f"CONSENSUS REACHED for operation: {operation} (seq: {sequence})")
        
        # Add to execution queue with priority based on sequence number
        self.execution_queue.put((sequence, {
            'request_id': request_id,
            'operation': operation,
            'view': self.pbft.view  # Include the view
        }))
    
    def on_view_change(self, new_view):
        """Handle view change notification from PBFT"""
        self.logger.info(f"View changed to {new_view}")
        
        # If this node is the new primary, start sending heartbeats
        if self.pbft.is_primary_node():
            self.logger.info(f"This node is now the primary for view {new_view}")
            # Start heartbeat timer if not already running
            self.pbft.start_heartbeat_timer()
        else:
            self.logger.info(f"This node is a backup for view {new_view}")
    
    def add_node(self, node_id: int, host: str, port: int):
        """Add a new node to the network"""
        # Check if the node already exists
        for node in self.nodes:
            if node['id'] == node_id:
                self.logger.warning(f"Node {node_id} already exists in the network")
                return
        
        # Add the node to the configuration
        new_node = {'id': node_id, 'host': host, 'port': port}
        self.nodes.append(new_node)
        
        # Update PBFT with new node count
        self.pbft.update_total_nodes(len(self.nodes))
        
        # If this is the primary node, sync blockchain and state to the new node
        if self.pbft.is_primary_node():
            self.logger.info(f"Syncing blockchain to node {node_id}")
            
            # First, send the current view to ensure the new node has the correct view
            view_sync = {
                'type': 'view-sync',
                'sender': self.node_id,
                'view': self.pbft.view,
                'primary': self.pbft.view % len(self.nodes)
            }
            self.send_message(new_node, view_sync)
            
            # Then sync all blocks
            for block in self.blockchain.blocks:
                block_sync = {
                    'type': 'block-sync',
                    'sender': self.node_id,
                    'block': block.to_dict(),
                    'sequence': block.index,
                    'view': self.pbft.view  # Include current view
                }
                self.send_message(new_node, block_sync)
            
            # Sync state
            state_sync = {
                'type': 'state-sync',
                'sender': self.node_id,
                'state': self.state,
                'last_executed_seq': self.last_executed_seq
            }
            self.send_message(new_node, state_sync)
            
            self.logger.info(f"Blockchain and state sync to node {node_id} completed")

    def handle_view_sync(self, message: Dict):
        """Handle view synchronization from the primary node"""
        sender = message.get('sender')
        view = message.get('view')
        primary = message.get('primary')
        
        self.logger.info(f"Received view sync from node {sender}: view={view}, primary={primary}")
        
        # Update our PBFT view
        self.pbft.view = view
        
        # Update primary status based on the new view
        new_primary_id = view % len(self.nodes)
        self.pbft._is_primary = (self.node_id == new_primary_id)
        
        self.logger.info(f"Updated view to {view}, primary status: {self.pbft.is_primary_node()}")
        
        # If we're joining after a view change, we need to notify all nodes about our presence
        if view > 0:
            self.logger.info(f"Joining after view change (view={view}), notifying all nodes")
            join_msg = {
                'type': 'node-join',
                'sender': self.node_id,
                'view': view
            }
            self.broadcast(join_msg)

    def handle_state_sync(self, message: Dict):
        """Handle state synchronization from another node"""
        sender = message.get('sender')
        state = message.get('state')
        last_executed_seq = message.get('last_executed_seq')
        view = message.get('view', 0)
        
        self.logger.info(f"Received state sync from node {sender}")
        
        # Update our view if needed
        if view > self.pbft.view:
            self.logger.info(f"Updating view from {self.pbft.view} to {view}")
            self.pbft.view = view
            
            # Update primary status
            new_primary_id = view % len(self.nodes)
            self.pbft._is_primary = (self.node_id == new_primary_id)
            
            self.logger.info(f"Updated primary status: {self.pbft.is_primary_node()}")
        
        # Update our state
        with self.state_lock:
            self.state = state
            self.last_executed_seq = last_executed_seq
            self.logger.info(f"Updated state from primary: {self.state}")
            self.logger.info(f"Updated last_executed_seq to {last_executed_seq}")

    def handle_node_join(self, message: Dict):
        """Handle notification that a new node has joined the network"""
        sender = message.get('sender')
        view = message.get('view')
        
        self.logger.info(f"Node {sender} has joined the network with view {view}")
        
        # If we're the primary, send our current state to the new node
        if self.pbft.is_primary_node():
            self.logger.info(f"Sending current state to new node {sender}")
            
            # Find the node in our configuration
            target_node = None
            for node in self.nodes:
                if node['id'] == sender:
                    target_node = node
                    break
            
            if target_node:
                # Send current state
                state_sync = {
                    'type': 'state-sync',
                    'sender': self.node_id,
                    'state': self.state,
                    'last_executed_seq': self.last_executed_seq,
                    'view': self.pbft.view
                }
                self.send_message(target_node, state_sync)
                
                # Send all blocks
                for block in self.blockchain.blocks:
                    block_sync = {
                        'type': 'block-sync',
                        'sender': self.node_id,
                        'block': block.to_dict(),
                        'sequence': block.index,
                        'view': self.pbft.view
                    }
                    self.send_message(target_node, block_sync)
                
                self.logger.info(f"Sent state and blockchain to new node {sender}")

    def handle_model_request(self, message: Dict):
        """Handle a request for the global model"""
        client_id = message.get('client_id', '')
        request_id = message.get('request_id', '')
        
        self.logger.info(f"Handling model request from client {client_id}")
        
        # Check if we have global model info in our state
        global_model_info = None
        with self.state_lock:
            if 'global_model' in self.state:
                try:
                    global_model_info = json.loads(self.state['global_model'])
                except:
                    self.logger.error("Failed to parse global model info from state")
        
        if not global_model_info:
            self.logger.warning("No global model information available")
            response = {
                'type': 'model-response',
                'status': 'error',
                'message': 'No global model available',
                'request_id': request_id
            }
        else:
            try:
                # Get the model file path and hash
                model_path = global_model_info.get('storage_path')
                model_hash = global_model_info.get('hash')
                
                if not model_path or not os.path.exists(model_path):
                    self.logger.error(f"Model file not found: {model_path}")
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                # Send only the model location and hash
                response = {
                    'type': 'model-response',
                    'status': 'success',
                    'version': global_model_info.get('version', 1),
                    'model_path': model_path,
                    'model_hash': model_hash,
                    'architecture': global_model_info.get('architecture', 'mobilenet_v2'),
                    'num_classes': global_model_info.get('num_classes', 10),
                    'request_id': request_id
                }
                
                self.logger.info(f"Sending global model info to client {client_id}")
            
            except Exception as e:
                self.logger.error(f"Error preparing model response: {e}")
                response = {
                    'type': 'model-response',
                    'status': 'error',
                    'message': str(e),
                    'request_id': request_id
                }
        
        # Send the response back to the client
        try:
            # Get the client socket from the current connection handler
            # This is stored in the thread_local storage when handling client connections
            if hasattr(threading.current_thread(), 'client_socket'):
                client_socket = threading.current_thread().client_socket
                client_socket.sendall(json.dumps(response).encode('utf-8'))
                self.logger.info(f"Sent model response to client {client_id}")
            else:
                # If we can't find the client socket in the current thread,
                # we need to send the response back through the server socket
                self.logger.info(f"No client socket in current thread, sending response through server")
                
                # The client should have sent its connection info
                client_host = message.get('client_host', 'localhost')
                client_port = message.get('client_port', 0)
                
                if client_port > 0:
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.connect((client_host, client_port))
                        s.sendall(json.dumps(response).encode('utf-8'))
                        s.close()
                        self.logger.info(f"Sent model response to client {client_id} at {client_host}:{client_port}")
                    except Exception as e:
                        self.logger.error(f"Error sending response to client: {e}")
                else:
                    self.logger.error(f"Cannot send response: no client socket and no valid client port")
        
        except Exception as e:
            self.logger.error(f"Error sending model response: {e}")

    def check_for_censored_requests(self):
        """Check if any requests have been censored by the primary"""
        # Only backup nodes check for censorship
        if self.pbft.is_primary_node() or self.pbft.in_view_change:
            return
        
        if not hasattr(self.pbft, 'request_timestamps'):
            self.pbft.request_timestamps = {}
        
        current_time = time.time()
        censored_requests = []
        
        # Check for requests that have been pending too long
        for request_id, timestamp in list(self.pbft.request_timestamps.items()):
            # Skip requests that have been executed
            if request_id in self.executed_requests:
                self.logger.debug(f"Request {request_id} has been executed, removing from tracking")
                del self.pbft.request_timestamps[request_id]
                continue
            
            # Check if request has been pending too long (15 seconds)
            if current_time - timestamp > 15:  # Reduced from 30 to 15 seconds for faster detection
                censored_requests.append(request_id)
                self.logger.warning(f"Request {request_id} appears to be censored by the primary (pending for {int(current_time - timestamp)} seconds)")
        
        # If we found censored requests, collect evidence
        if censored_requests:
            self.logger.warning(f"Detected {len(censored_requests)} censored requests: {censored_requests}")
            
            # Track censorship evidence
            if not hasattr(self, 'censorship_evidence'):
                self.censorship_evidence = {}
            
            # Add all censored requests as unreported evidence
            for request_id in censored_requests:
                self.censorship_evidence[request_id] = {
                    'detected_at': current_time,
                    'reported': False  # Always set to False to ensure it's counted
                }
                self.logger.info(f"Added request {request_id} to censorship evidence as unreported")
            
            # Check if we have enough evidence to trigger a view change
            self.check_censorship_evidence()
        else:
            self.logger.debug("No censored requests detected")

    def check_censorship_evidence(self):
        """Check if we have enough evidence to trigger a view change"""
        if not hasattr(self, 'censorship_evidence'):
            self.censorship_evidence = {}
            return
        
        # Count unreported censored requests
        unreported = [req_id for req_id, evidence in self.censorship_evidence.items() 
                     if not evidence['reported']]
        
        self.logger.info(f"Found {len(unreported)} unreported censored requests, need {self.pbft.f + 1} to trigger view change")
        
        if len(unreported) >= 1:  # Changed from f+1 to 1 for testing
            self.logger.warning(f"Found evidence of {len(unreported)} censored requests - initiating view change")
            
            # Mark these as reported
            for req_id in unreported:
                self.censorship_evidence[req_id]['reported'] = True
            
            # Initiate view change
            if not self.pbft.in_view_change:
                new_view = self.pbft.view + 1
                view_change_msg = {
                    'type': 'view-change',
                    'new_view': new_view,
                    'last_seq': self.pbft.sequence_number,
                    'sender': self.node_id,
                    'prepared': {},
                    'reason': 'selective_censorship',
                    'censored_requests': unreported
                }
                
                self.pbft.in_view_change = True
                self.pbft.process_message(view_change_msg)
                self.broadcast(view_change_msg)
                self.logger.warning(f"Initiated view change to view {new_view} due to selective censorship")

    def create_primary_block(self, data, model_type="primary-created"):
        """Create a block directly from the primary node (bypassing consensus)"""
        if not self.pbft.is_primary_node():
            self.logger.warning("Only primary nodes can create blocks directly")
            return None
        
        self.logger.info(f"Primary node creating block with data: {data}")
        
        with self.blockchain_lock:
            # Create a new block
            new_block = self.blockchain.create_block(
                data=data,
                model_type=model_type,
                storage_reference=f"primary-{int(time.time())}",
                calculated_hash=hashlib.sha256(str(data).encode()).hexdigest(),
                participants=[str(self.node_id)]
            )
            
            # Add the block to our blockchain
            self.blockchain.add_block(new_block)
            self.logger.info(f"Created block #{new_block.index} as primary node")
            
            # Broadcast the block to all other nodes
            block_sync = {
                'type': 'block-sync',
                'sender': self.node_id,
                'block': new_block.to_dict(),
                'sequence': 0,  # Not part of consensus
                'view': self.pbft.view
            }
            self.broadcast(block_sync, exclude_self=True)
            
            return new_block

    def validate_operation(self, sequence: int, request: Dict):
        """Perform validation check for an operation"""
        operation = request.get('operation', '')
        request_id = request.get('request_id', '')
        view = request.get('view', 0)
        
        self.logger.info(f"Validating operation for seq {sequence}, view {view}: {operation}")
        
        # Default to valid
        is_valid = True
        
        try:
            # Check if this is a model update operation
            if operation.startswith('UPDATE_MODEL '):
                # Format: UPDATE_MODEL model_path model_hash loss accuracy
                parts = operation.split(' ', 4)
                if len(parts) >= 5:
                    model_path = parts[1]
                    model_hash = parts[2]
                    update_loss = float(parts[3])
                    update_accuracy = float(parts[4])
                    
                    # Extract client ID from request
                    client_id = request.get('client_id', 'unknown')
                    
                    self.logger.info(f"Validating model update from {client_id}")
                    self.logger.info(f"Update model metrics - Loss: {update_loss:.4f}, Accuracy: {update_accuracy*100:.2f}%")
                    
                    # Get the global model info
                    global_model_info = None
                    with self.state_lock:
                        if 'global_model' in self.state:
                            try:
                                global_model_info = json.loads(self.state['global_model'])
                            except:
                                self.logger.error("Failed to parse global model info from state")
                    
                    if global_model_info:
                        # Evaluate the global model on our test set if we haven't already
                        if not hasattr(self, 'global_model_metrics'):
                            self.global_model_metrics = {}
                        
                        global_model_path = global_model_info.get('storage_path')
                        
                        # If we haven't evaluated this global model yet, do it now
                        if global_model_path not in self.global_model_metrics:
                            self.logger.info(f"Evaluating global model {global_model_path} for validation")
                            
                            # Load and evaluate the global model
                            try:
                                # Load the model
                                device = torch.device("cpu")
                                
                                # Initialize model architecture
                                architecture = global_model_info.get('architecture', 'mobilenet_v2')
                                model = Net(num_classes=10, arch=architecture).to(device)
                                
                                # Load weights based on file extension
                                if global_model_path.endswith('.npz'):
                                    # Load NPZ file
                                    npz_data = np.load(global_model_path, allow_pickle=True)
                                    
                                    # Convert numpy arrays to PyTorch tensors
                                    state_dict = {}
                                    for key in npz_data.files:
                                        # Skip any non-tensor metadata
                                        try:
                                            tensor = torch.from_numpy(npz_data[key])
                                            state_dict[key] = tensor
                                        except TypeError:
                                            self.logger.warning(f"Skipping non-tensor key in NPZ file: {key}")
                                    
                                    # Load state dict
                                    model.load_state_dict(state_dict)
                                
                                # Evaluate the model
                                model.eval()
                                criterion = nn.CrossEntropyLoss()
                                
                                # Get test data from flower client
                                x_test = self.flower_client.x_test
                                y_test = self.flower_client.y_test
                                
                                # Convert to tensors if needed
                                if not isinstance(x_test, torch.Tensor):
                                    x_test = torch.tensor(x_test)
                                if not isinstance(y_test, torch.Tensor):
                                    y_test = torch.tensor(y_test)
                                
                                # Move to device
                                x_test = x_test.to(device)
                                y_test = y_test.to(device)
                                
                                # Evaluate
                                with torch.no_grad():
                                    outputs = model(x_test)
                                    loss = criterion(outputs, y_test)
                                    _, predicted = torch.max(outputs.data, 1)
                                    total = y_test.size(0)
                                    correct = (predicted == y_test).sum().item()
                                    
                                    global_loss = loss.item()
                                    global_accuracy = correct / total
                                
                                # Store metrics
                                self.global_model_metrics[global_model_path] = {
                                    'loss': global_loss,
                                    'accuracy': global_accuracy
                                }
                                
                                self.logger.info(f"Global model metrics - Loss: {global_loss:.4f}, Accuracy: {global_accuracy*100:.2f}%")
                            
                            except Exception as e:
                                self.logger.error(f"Error evaluating global model: {e}")

                                traceback.print_exc()
                                # Default to accepting the update if we can't evaluate the global model
                                self.global_model_metrics[global_model_path] = {
                                    'loss': float('inf'),
                                    'accuracy': 0.0
                                }
                        
                        # Compare update model with global model
                        global_metrics = self.global_model_metrics[global_model_path]
                        global_loss = global_metrics['loss']
                        global_accuracy = global_metrics['accuracy']
                        
                        self.logger.info(f"Comparing models:")
                        self.logger.info(f"  Global model: Loss={global_loss:.4f}, Accuracy={global_accuracy*100:.2f}%")
                        self.logger.info(f"  Update model: Loss={update_loss:.4f}, Accuracy={update_accuracy*100:.2f}%")
                        
                        # Validation rule: update model must have lower loss or higher accuracy
                        is_valid = (update_loss < global_loss) or (update_accuracy > global_accuracy)
                        
                        if is_valid:
                            self.logger.info(f"✅ Model update from {client_id} VALIDATED - Improves on global model")
                        else:
                            self.logger.warning(f"❌ Model update from {client_id} REJECTED - Does not improve on global model")
                    else:
                        self.logger.warning("No global model info found, defaulting to accepting the update")
            
            # For SET operations, keep the existing validation logic
            elif operation.startswith('SET ') and len(operation.split(' ', 2)) == 3:
                key, value = operation.split(' ', 2)[1:]
                
                # Check if value is numeric
                try:
                    value_int = int(value)
                    # Validation rule: value must be greater than node_id
                    is_valid = value_int > self.node_id
                    self.logger.info(f"Validation check: value {value_int} > node_id {self.node_id} = {is_valid}")
                except ValueError:
                    # Non-numeric values are always valid
                    self.logger.info(f"Non-numeric value '{value}', using default validation (valid)")
            else:
                # Non-SET operations are always valid
                self.logger.info(f"Operation type '{operation.split(' ')[0]}' doesn't require validation")
        except Exception as e:
            self.logger.error(f"Error during validation: {e}")
            # Default to valid in case of errors
            is_valid = True
        
        # Create validation key for tracking
        validation_key = f"v{view}-s{sequence}"
        
        # Record validation result
        if not hasattr(self, 'validations'):
            self.validations = {}
        self.validations[validation_key] = is_valid
        
        # Broadcast validation result to other nodes
        validation_msg = {
            'type': 'validation-result',
            'sender': self.node_id,
            'sequence': sequence,
            'view': view,
            'request_id': request_id,
            'is_valid': is_valid
        }
        self.broadcast(validation_msg)
        
        # Also process our own validation result
        self.handle_validation_result(validation_msg)

        # Log validation votes for model updates
        if operation.startswith('UPDATE_MODEL '):
            # Log validation votes for model updates
            parts = operation.split(' ', 4)
            if len(parts) >= 5:
                model_path = parts[1]
                client_id = request.get('client_id', 'unknown')
                
                # Create a validation log entry
                if not hasattr(self, 'model_validation_log'):
                    self.model_validation_log = {}
                
                if validation_key not in self.model_validation_log:
                    self.model_validation_log[validation_key] = {
                        'client_id': client_id,
                        'model_path': model_path,
                        'votes': {},
                        'result': None
                    }
                
                # Add this node's vote
                self.model_validation_log[validation_key]['votes'][self.node_id] = is_valid
                
                # Log the current voting status
                votes_for = sum(1 for v in self.model_validation_log[validation_key]['votes'].values() if v)
                votes_against = len(self.model_validation_log[validation_key]['votes']) - votes_for
                
                self.logger.info(f"Model validation votes for {client_id}: {votes_for} FOR, {votes_against} AGAINST")

    def handle_validation_result(self, message: Dict):
        """Handle validation results from other nodes"""
        sender = message.get('sender')
        sequence = message.get('sequence')
        view = message.get('view')
        is_valid = message.get('is_valid')
        request_id = message.get('request_id')
        
        validation_key = f"v{view}-s{sequence}"
        self.logger.info(f"Received validation result from node {sender} for seq {sequence}: {is_valid}")
        
        # Store validation result
        if not hasattr(self, 'validation_results'):
            self.validation_results = {}
        
        if validation_key not in self.validation_results:
            self.validation_results[validation_key] = {}
        
        self.validation_results[validation_key][sender] = is_valid
        
        # Count valid votes and total votes
        valid_count = sum(1 for v in self.validation_results[validation_key].values() if v)
        total_count = len(self.validation_results[validation_key])
        
        # Calculate required valid votes for consensus (2f+1)
        required_valid = 2 * self.pbft.f + 1
        
        self.logger.info(f"Validation status for {validation_key}: {valid_count}/{total_count} valid, need {required_valid}")
        
        # Track votes for model updates in the log
        if hasattr(self, 'model_validation_log') and validation_key in self.model_validation_log:
            self.model_validation_log[validation_key]['votes'][sender] = is_valid
            
            # Update vote counts in log
            votes_for = sum(1 for v in self.model_validation_log[validation_key]['votes'].values() if v)
            votes_against = len(self.model_validation_log[validation_key]['votes']) - votes_for
            
            self.logger.info(f"Updated model validation votes: {votes_for} FOR, {votes_against} AGAINST")
        
        # Case 1: We have enough valid votes to mark as valid
        if valid_count >= required_valid:
            self.logger.info(f"✅ Validation consensus reached for {validation_key}: VALID ({valid_count} ≥ {required_valid})")
            self._mark_operation_as_valid(validation_key, sequence, request_id)
            
            # Update model validation log
            if hasattr(self, 'model_validation_log') and validation_key in self.model_validation_log:
                self.model_validation_log[validation_key]['result'] = True
                self.logger.info(f"Model update ACCEPTED by consensus")
        
        # Case 2: It's mathematically impossible to reach enough valid votes
        elif valid_count + (len(self.nodes) - total_count) < required_valid:
            self.logger.warning(f"❌ Validation consensus reached for {validation_key}: INVALID (max possible: {valid_count + (len(self.nodes) - total_count)} < {required_valid})")
            self._mark_operation_as_invalid(validation_key, sequence, view, request_id)
            
            # Update model validation log
            if hasattr(self, 'model_validation_log') and validation_key in self.model_validation_log:
                self.model_validation_log[validation_key]['result'] = False
                self.logger.warning(f"Model update REJECTED by consensus")
        
        # Case 3: We've heard from all nodes and don't have enough valid votes
        elif total_count == len(self.nodes) and valid_count < required_valid:
            self.logger.warning(f"❌ Validation consensus reached for {validation_key}: INVALID (all nodes reported, {valid_count} < {required_valid})")
            self._mark_operation_as_invalid(validation_key, sequence, view, request_id)
            
            # Update model validation log
            if hasattr(self, 'model_validation_log') and validation_key in self.model_validation_log:
                self.model_validation_log[validation_key]['result'] = False
                self.logger.warning(f"Model update REJECTED by consensus")
        
        # Otherwise: Still waiting for more votes

    def _mark_operation_as_valid(self, validation_key, sequence, request_id):
        """Mark an operation as valid and execute it if pending"""
        # Store the validation result
        if not hasattr(self, 'validated_operations'):
            self.validated_operations = {}
        self.validated_operations[validation_key] = True
        
        # Check if this operation is waiting for validation
        if hasattr(self, 'pending_validations') and validation_key in self.pending_validations:
            request = self.pending_validations[validation_key]
            self.logger.info(f"Executing validated operation for {validation_key}")
            self.execute_operation(sequence, request)
            del self.pending_validations[validation_key]

    def _mark_operation_as_invalid(self, validation_key, sequence, view, request_id):
        """Mark an operation as invalid and clean up"""
        # Store the validation result
        if not hasattr(self, 'validated_operations'):
            self.validated_operations = {}
        self.validated_operations[validation_key] = False
        
        # Remove from pending validations if it exists
        if hasattr(self, 'pending_validations') and validation_key in self.pending_validations:
            self.logger.warning(f"Removing invalid operation for {validation_key}")
            del self.pending_validations[validation_key]
        
        # If we're the primary, broadcast a validation-failed message
        if self.pbft.is_primary_node():
            validation_failed_msg = {
                'type': 'validation-failed',
                'sender': self.node_id,
                'sequence': sequence,
                'view': view,
                'request_id': request_id
            }
            self.broadcast(validation_failed_msg)

    def handle_validation_failed(self, message: Dict):
        """Handle notification that an operation failed validation"""
        sequence = message.get('sequence')
        view = message.get('view')
        request_id = message.get('request_id')
        
        validation_key = f"v{view}-s{sequence}"
        self.logger.warning(f"Received validation-failed for {validation_key}")
        
        # Mark this operation as invalid
        if not hasattr(self, 'validated_operations'):
            self.validated_operations = {}
        self.validated_operations[validation_key] = False
        
        # Remove from pending validations if it exists
        if hasattr(self, 'pending_validations') and validation_key in self.pending_validations:
            self.logger.warning(f"Removing invalid operation for {validation_key}")
            del self.pending_validations[validation_key]
        
        # If we've already executed this operation, we need to roll it back
        # This is a simplified approach - in a real system, you'd need a more robust rollback mechanism
        operation = None
        for block in self.blockchain.blocks:
            if block.data.get('block_id') == validation_key:
                operation = block.data.get('operation')
                break
        
        if operation and operation.startswith('SET '):
            parts = operation.split(' ', 2)
            if len(parts) == 3:
                key = parts[1]
                with self.state_lock:
                    if key in self.state:
                        self.logger.warning(f"Rolling back invalid operation: {operation}")
                        del self.state[key]

    def aggregate_models(self, version=None):
        """Aggregate model updates into a new global model using simple averaging
        
        Args:
            version: Optional specific version of updates to aggregate
        """
        if not self.pbft.is_primary_node():
            self.logger.warning("Only primary node can perform aggregation")
            return
        
        # If no specific version is provided, use the current global model version
        if version is None:
            version = self.global_model_version
        
        # Get updates for the specified version
        if hasattr(self, 'pending_updates_by_version') and version in self.pending_updates_by_version:
            updates_to_aggregate = self.pending_updates_by_version[version]
            # Filter out any potential None entries if errors occurred during addition
            updates_to_aggregate = [u for u in updates_to_aggregate if u is not None] 
        else:
            # Fall back is less useful now, rely on version specific
             self.logger.warning(f"No pending updates found for specified version v{version} to aggregate.")
             return
             # updates_to_aggregate = self.pending_updates # Old fallback

        if not updates_to_aggregate:
             self.logger.warning(f"No valid updates to aggregate for version v{version}.")
             return

        self.logger.info(f"Starting model aggregation for version v{version} with {len(updates_to_aggregate)} updates")

        try:
            self.logger.debug(f"[AGG_V{version}] Starting try block.")
            
            # --- Get Current Global Model Info (Needed for architecture etc.) ---
            global_model_info = None
            with self.state_lock:
                if 'global_model' in self.state:
                    try:
                        global_model_info = json.loads(self.state['global_model'])
                    except Exception as e:
                        self.logger.error(f"[AGG_V{version}] Failed to parse current global model info: {e}")
                        # Decide how to handle this - maybe return? For now, log and continue cautiously.
                if global_model_info is None:
                    self.logger.warning(f"[AGG_V{version}] Cannot find current global model info. Using defaults.")
                    # Set defaults if info is missing
                    global_model_info = {'architecture': 'mobilenet_v2', 'num_classes': 10}


            # --- Create Directories ---
            models_dir = "models"
            npz_dir = "models/npz"
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(npz_dir, exist_ok=True)

            # --- Load Update Model Weights ---
            self.logger.info(f"Loading {len(updates_to_aggregate)} update model(s)/result(s)")
            update_weights = []
            
            for update in updates_to_aggregate:
                model_path = update.get('model_path') # Use .get for safety
                update_type = update.get('type', 'model_update') # Default to individual update
                
                if not model_path or not os.path.exists(model_path):
                    self.logger.warning(f"Update model file not found: {model_path}, skipping this update.")
                    continue
                
                # Verify hash before loading (important for cluster results too)
                expected_hash = update.get('model_hash')
                if expected_hash:
                     with open(model_path, 'rb') as f:
                         actual_hash = hashlib.sha256(f.read()).hexdigest()
                     if actual_hash != expected_hash:
                         self.logger.warning(f"[AGG_V{version}] Hash mismatch for {model_path}! Skipping.")
                         continue
                
                # Load the update model weights
                try:
                    self.logger.info(f"[AGG_V{version}] Loading weights from {model_path} (Type: {update_type})")
                    update_npz = np.load(model_path, allow_pickle=True)
                    model_weights = {}
                    for key in update_npz.files:
                        try: model_weights[key] = update_npz[key]
                        except: self.logger.warning(f"[AGG_V{version}] Skipping non-tensor key {key} in {model_path}")
                    
                    if model_weights:
                        update_weights.append(model_weights)
                    else:
                         self.logger.warning(f"[AGG_V{version}] No valid weights found in {model_path}")

                except Exception as e:
                     self.logger.error(f"[AGG_V{version}] Error loading weights from {model_path}: {e}")
                     continue # Skip this update if loading fails

            if not update_weights:
                self.logger.warning(f"[AGG_V{version}] No valid weights loaded, aborting aggregation.")
                return
            
            self.logger.debug(f"[AGG_V{version}] Averaging weights.")
            # --- Simple Averaging ---
            aggregated_weights = {}
            first_weights = update_weights[0]
            num_models_averaged = len(update_weights)

            # Initialize with zeros (using float64 for accumulation)
            for key in first_weights:
                 aggregated_weights[key] = np.zeros_like(first_weights[key], dtype=np.float64)

            # Sum weights
            for weights in update_weights:
                for key in aggregated_weights:
                    if key in weights:
                        aggregated_weights[key] += weights[key].astype(np.float64)

            # Average weights
            for key in aggregated_weights:
                aggregated_weights[key] /= num_models_averaged
                # Cast back to original dtype if possible
                try:
                    original_dtype = first_weights[key].dtype
                    if original_dtype != np.float64:
                        aggregated_weights[key] = aggregated_weights[key].astype(original_dtype)
                except Exception as e:
                    self.logger.warning(f"[AGG_V{version}] Could not cast aggregated key {key} back to original dtype: {e}")


            # --- Save New Aggregated Model ---
            new_version = version + 1
            timestamp = int(time.time())
            new_model_filename = f"global_model_v{new_version}_{timestamp}.npz"
            new_model_path = os.path.join(npz_dir, new_model_filename)

            self.logger.debug(f"[AGG_V{version}] Saving new model file: {new_model_path}")
            np.savez(new_model_path, **aggregated_weights)

            # --- Calculate Hash ---
            new_model_hash = ""
            with open(new_model_path, 'rb') as f:
                new_model_hash = hashlib.sha256(f.read()).hexdigest()

            self.logger.debug(f"[AGG_V{version}] Creating metadata.")
            # Create model metadata for the new global model
            new_model_data = {
                 'type': 'aggregated_model',
                 'version': new_version,
                 'created_by': f"node-{self.node_id}",
                 'timestamp': timestamp,
                 'storage_path': new_model_path,
                 'hash': new_model_hash,
                 'architecture': global_model_info.get('architecture', 'mobilenet_v2'),
                 'num_classes': global_model_info.get('num_classes', 10),
                 'aggregated_from_version': version,
                 'num_updates_aggregated': len(updates_to_aggregate),
                 'aggregated_from_types': [u.get('type', 'model_update') for u in updates_to_aggregate],
                 'based_on_updates_details': updates_to_aggregate # Potentially large, consider summarizing if needed
            }

            self.logger.debug(f"[AGG_V{version}] Calling create_new_global_model_consensus.")
            self.create_new_global_model_consensus(new_model_data)
            
            # --- Cleanup ---
            # Clear the specific version's pending updates after successful aggregation attempt
            # (Consensus will handle the actual state update)
            if hasattr(self, 'pending_updates_by_version') and version in self.pending_updates_by_version:
                 self.logger.info(f"[AGG_V{version}] Clearing pending updates after initiating aggregation consensus.")
                 self.pending_updates_by_version[version] = []

            self.logger.info(f"[AGG_V{version}] Aggregation process completed successfully. Initiated consensus for v{new_version}.")
            
        except Exception as e:
             self.logger.error(f"[AGG_V{version}] Error during model aggregation: {e}")
             traceback.print_exc()
             # Ensure cleanup happens if possible

    def create_new_global_model_consensus(self, model_data):
        """Create a consensus request for the new global model"""
        if not self.pbft.is_primary_node():
            self.logger.warning("Only primary node can initiate global model consensus")
            return
        
        # Create a unique request ID that includes the version
        version = model_data['version']
        timestamp = model_data['timestamp']
        request_id = f"global_model_v{version}_{timestamp}"
        
        # Check if we've already initiated consensus for this version
        if hasattr(self, 'global_model_consensus_versions') and version in self.global_model_consensus_versions:
            self.logger.warning(f"[CONSENSUS_V{version}] Already initiated, skipping.")
            return
        
        # Track this version to avoid duplicates
        if not hasattr(self, 'global_model_consensus_versions'):
            self.global_model_consensus_versions = set()
        self.global_model_consensus_versions.add(version)
        
        # Create operation string
        operation = f"UPDATE_GLOBAL_MODEL {model_data['storage_path']} {model_data['hash']} {model_data['version']}"
        
        # Create request
        request = {
            'type': 'request',
            'client_id': f"node-{self.node_id}",
            'timestamp': timestamp,
            'operation': operation,
            'request_id': request_id,
            'model_data': model_data  # Include full model data
        }
        
        # Store request in PBFT module
        digest = hashlib.sha256(f"{request_id}:{operation}".encode()).hexdigest()
        self.pbft.store_request(request_id, {
            'client_id': f"node-{self.node_id}",
            'timestamp': timestamp,
            'operation': operation,
            'digest': digest
        })
        
        # Broadcast request to all nodes
        self.logger.debug(f"[CONSENSUS_V{version}] Broadcasting request.")
        self.broadcast(request)
        
        # Start consensus
        self.logger.debug(f"[CONSENSUS_V{version}] Calling pbft.start_consensus.")
        self.pbft.start_consensus(request_id)
        
        self.logger.info(f"[CONSENSUS_V{version}] Initiated consensus for new global model v{version}")
