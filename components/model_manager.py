import torch
import os
import json
import hashlib
import time
import traceback
import numpy as np
import threading
import socket
from typing import Dict
import logging
class ModelManager:
    def __init__(self, test_set):
        self.models = {}
        self.test_set = test_set
        self.logger = logging.getLogger(f"Node-{self.node.node_id}")

    def load_model(self, model_path):
        self.models[model_path] = torch.load(model_path)

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

    def aggregate_models(self, version=None):
        """Aggregate model updates into a new global model using simple averaging
        
        Args:
            version: Optional specific version of updates to aggregate
        """
        if not self.node.pbft.is_primary_node():
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
        self.message_handler.broadcast(request)
        
        # Start consensus
        self.logger.debug(f"[CONSENSUS_V{version}] Calling pbft.start_consensus.")
        self.pbft.start_consensus(request_id)
        
        self.logger.info(f"[CONSENSUS_V{version}] Initiated consensus for new global model v{version}")
