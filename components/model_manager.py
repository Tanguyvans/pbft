import torch
import os
import json
import hashlib
import time
import traceback
import numpy as np
import threading
import socket
from typing import Dict, Optional, List, Tuple
import logging
import torch.nn as nn

from going_modular.model import Net

class ModelManager:
    def __init__(self, node, test_set, flower_client):
        self.node = node
        self.logger = node.logger
        self.test_set = test_set
        self.flower_client = flower_client
        
        # Initialize model state
        self.global_model_version = 1
        self.metrics_cache = {}
        
        # Setup directories
        self.models_dir = "models"
        self.npz_dir = "models/npz"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.npz_dir, exist_ok=True)
        
        # Prepare test data
        self.device = torch.device("cpu")
        self.test_data = self._prepare_test_data()

    def handle_operation(self, operation: str, request: Dict, sequence: int) -> Tuple[str, bool]:
        """Main entry point for handling model-related operations"""
        try:
            if operation.startswith('UPDATE_MODEL '):
                return self._handle_model_update(operation, request)
            elif operation == 'CREATE_GLOBAL_MODEL':
                return self._handle_create_global()
            elif operation.startswith('UPDATE_GLOBAL_MODEL '):
                return self._handle_update_global(operation, request)
            elif operation.startswith('CLUSTER_TRAIN_VALIDATE '):
                return self._handle_cluster_validation(operation, request, sequence)
            else:
                return f"ERROR: Unknown operation type: {operation}", False
        except Exception as e:
            self.logger.error(f"Error handling operation: {e}")
            return f"ERROR: {str(e)}", False

    def _handle_model_update(self, operation: str, request: Dict) -> Tuple[str, bool]:
        """Handle individual model updates from clients"""
        parts = operation.split(' ')
        if len(parts) < 5:
            return "ERROR: Invalid UPDATE_MODEL format", False
            
        model_path = parts[1]
        model_hash = parts[2]
        training_loss = float(parts[3])
        training_accuracy = float(parts[4])
        client_id = request.get('client_id', 'unknown')
        
        # Extract version information
        update_version = 1
        for part in parts[5:]:
            if part.startswith('v') and part[1:].isdigit():
                update_version = int(part[1:])
                break

        # Verify model file and hash
        if not self._verify_model_file(model_path, model_hash):
            return f"ERROR: Model verification failed for {model_path}", False

        # Create model metadata
        model_update = {
            'type': 'model_update',
            'client_id': client_id,
            'timestamp': int(time.time()),
            'storage_path': model_path,
            'hash': model_hash,
            'training_loss': training_loss,
            'training_accuracy': training_accuracy * 100,
            'version': update_version,
            'based_on_global': self.global_model_version
        }

        # Store update in state
        with self.node.state_lock:
            if 'model_updates' not in self.node.state:
                self.node.state['model_updates'] = []
            self.node.state['model_updates'].append(model_update)
            
            return f"MODEL_UPDATED: {client_id}, accuracy: {training_accuracy * 100:.2f}%, version: v{update_version}", True

    def _handle_create_global(self) -> Tuple[str, bool]:
        """Handle creation of initial global model"""
        try:
            # --- Simple Save with Delay ---
            timestamp = int(time.time())
            model_filename = f"global_model_v1_{timestamp}.npz"
            final_model_path = os.path.join(self.npz_dir, model_filename)
            saved_successfully = False

            try:
                # Ensure the target directory exists
                os.makedirs(self.npz_dir, exist_ok=True)

                # Get model weights
                state_dict = self.flower_client.model.state_dict()
                numpy_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}

                # Save directly using np.savez
                np.savez(final_model_path, **numpy_dict)
                self.logger.info(f"Initial global model saved to: {final_model_path}")

                # *** Add a small delay to allow filesystem writes to complete ***
                time.sleep(0.2)
                self.logger.debug("Short delay after np.savez complete.")

                saved_successfully = True

            except Exception as e:
                 self.logger.error(f"Failed to save initial global model NPZ: {e}")
                 # Clean up potentially partially written file if it exists
                 if os.path.exists(final_model_path):
                     try:
                         os.remove(final_model_path)
                     except OSError as rm_err:
                         self.logger.error(f"Error removing file {final_model_path}: {rm_err}")
                 return f"ERROR: Failed to save initial model file", False

            # --- Calculate hash ---
            model_hash = ""
            try:
                with open(final_model_path, 'rb') as f:
                    model_hash = hashlib.sha256(f.read()).hexdigest()
                self.logger.info(f"Calculated hash for {final_model_path}: {model_hash[:10]}...")
            except Exception as e:
                self.logger.error(f"Failed to read model file for hashing: {e}")
                return f"ERROR: Failed read model file after saving", False

            # --- Create metadata ---
            model_data = {
                'type': 'initial_model',
                'version': 1,
                'created_by': f"node-{self.node.node_id}",
                'timestamp': timestamp,
                'storage_path': final_model_path,
                'hash': model_hash,
                'architecture': 'mobilenet_v2', # TODO: Get dynamically if needed
                'num_classes': 10 # TODO: Get dynamically if needed
            }

            # --- Update state ---
            with self.node.state_lock:
                self.node.state['global_model'] = json.dumps(model_data)
            self.logger.info(f"Node state updated with global model v1 info.")

            return "GLOBAL_MODEL_CREATED", True

        except Exception as e:
            self.logger.error(f"Error creating global model: {e}")
            traceback.print_exc() # Print traceback for unexpected errors
            return f"ERROR: {str(e)}", False

    def _handle_update_global(self, operation: str, request: Dict) -> Tuple[str, bool]:
        """Handle global model updates after aggregation"""
        parts = operation.split(' ')
        if len(parts) < 4:
            return "ERROR: Invalid UPDATE_GLOBAL_MODEL format", False
            
        model_path = parts[1]
        model_hash = parts[2]
        version = int(parts[3])
        
        if not self._verify_model_file(model_path, model_hash):
            return f"ERROR: Global model verification failed", False
            
        model_data = request.get('model_data', {})
        if not model_data:
            model_data = {
                'type': 'aggregated_model',
                'version': version,
                'created_by': request.get('client_id', f"node-{self.node.node_id}"),
                'timestamp': int(time.time()),
                'storage_path': model_path,
                'hash': model_hash,
                'architecture': 'mobilenet_v2',
                'num_classes': 10
            }
            
        with self.node.state_lock:
            self.node.state['global_model'] = json.dumps(model_data)
            self.global_model_version = version
            return f"GLOBAL_MODEL_UPDATED: version {version}", True

    def _verify_model_file(self, model_path: str, expected_hash: str) -> bool:
        """Verify model file exists and hash matches"""
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return False
            
        with open(model_path, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        if actual_hash != expected_hash:
            self.logger.warning(f"Model hash verification failed!")
            self.logger.warning(f"Expected: {expected_hash}")
            self.logger.warning(f"Actual: {actual_hash}")
            return False
        return True

    def _prepare_test_data(self):
        """Prepare test data for model evaluation"""
        try:
            # Unpack test set directly from constructor argument
            x_test, y_test = self.test_set
            
            # Convert to tensors if they aren't already
            if not isinstance(x_test, torch.Tensor):
                if isinstance(x_test, np.ndarray):
                    x_test = torch.from_numpy(x_test)
                else:
                    x_test = torch.tensor(np.array(x_test))
                
            if not isinstance(y_test, torch.Tensor):
                if isinstance(y_test, np.ndarray):
                    y_test = torch.from_numpy(y_test)
                else:
                    y_test = torch.tensor(np.array(y_test))
            
            # Move to device
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)
            
            self.logger.info(f"Test data prepared: x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
            return {'x_test': x_test, 'y_test': y_test}
            
        except Exception as e:
            self.logger.error(f"Error preparing test data: {e}")
            self.logger.error(f"x_test type: {type(x_test)}, y_test type: {type(y_test)}")
            if isinstance(x_test, np.ndarray):
                self.logger.error(f"x_test shape: {x_test.shape}, dtype: {x_test.dtype}")
            if isinstance(y_test, np.ndarray):
                self.logger.error(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
            raise
    
    def evaluate_model(self, model_path: str) -> Dict[str, float]:
        """Evaluate a model and return its metrics"""
        # Check cache first
        if model_path in self.metrics_cache:
            return self.metrics_cache[model_path]
            
        try:
            # Load model
            model = Net(num_classes=10).to(self.device)
            state_dict = dict(np.load(model_path, allow_pickle=True))
            model.load_state_dict({k: torch.from_numpy(v) for k, v in state_dict.items()})
            
            # Prepare data if not already done
            if not hasattr(self, 'test_data') or self.test_data is None:
                self.test_data = self._prepare_test_data()
            
            # Set model to evaluation mode
            model.eval()
            
            # Initialize metrics
            total_loss = 0.0
            correct = 0
            total = 0
            criterion = nn.CrossEntropyLoss()
            batch_size = 32  # Can be adjusted based on your needs
            
            # Evaluate in batches
            with torch.no_grad():
                x_test = self.test_data['x_test']
                y_test = self.test_data['y_test']
                
                for i in range(0, len(x_test), batch_size):
                    batch_x = x_test[i:i + batch_size]
                    batch_y = y_test[i:i + batch_size]
                    
                    # Forward pass
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    # Accumulate loss
                    total_loss += loss.item() * len(batch_x)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
            # Calculate final metrics
            avg_loss = total_loss / total
            accuracy = correct / total
            
            # Log evaluation results
            self.logger.info(f"Model Evaluation Results for {model_path}:")
            self.logger.info(f"- Test Loss: {avg_loss:.4f}")
            self.logger.info(f"- Test Accuracy: {accuracy:.4f}")
            
            # Cache and return metrics
            metrics = {'loss': avg_loss, 'accuracy': accuracy}
            self.metrics_cache[model_path] = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_path}: {e}")
            self.logger.error(traceback.format_exc())
            return {'loss': float('inf'), 'accuracy': 0.0}
    
    def validate_model_update(self, model_path: str, model_hash: str, 
                            reported_loss: float, reported_accuracy: float) -> bool:
        """Validate a model update by evaluating it on our test dataset"""
        try:
            # First verify the model file exists and hash matches
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            if actual_hash != model_hash:
                self.logger.error(f"Model hash mismatch. Expected: {model_hash}, Got: {actual_hash}")
                return False

            # Evaluate the update model on our test dataset
            update_metrics = self.evaluate_model(model_path)
            actual_loss = update_metrics['loss']
            actual_accuracy = update_metrics['accuracy']
            
            # Get and evaluate current global model for comparison
            global_model_info = self._get_global_model_info()
            if not global_model_info:
                self.logger.warning("No global model info available, accepting update")
                return True
            
            global_model_path = global_model_info.get('storage_path')
            global_metrics = self.evaluate_model(global_model_path)
            
            # Log all metrics for comparison
            self.logger.info("Model Validation Metrics:")
            self.logger.info(f"Global Model - Loss: {global_metrics['loss']:.4f}, Accuracy: {global_metrics['accuracy']:.4f}")
            self.logger.info(f"Update Model - Reported - Loss: {reported_loss:.4f}, Accuracy: {reported_accuracy:.4f}")
            self.logger.info(f"Update Model - Actual   - Loss: {actual_loss:.4f}, Accuracy: {actual_accuracy:.4f}")
            
            # Validation checks:
            # 1. Verify honesty: reported metrics should be reasonably close to actual metrics
            honesty_threshold = 0.1  # 10% tolerance
            is_honest = (
                abs(reported_loss - actual_loss) <= honesty_threshold * reported_loss and
                abs(reported_accuracy - actual_accuracy) <= honesty_threshold
            )
            
            # 2. Performance check: actual metrics should be comparable or better than global model
            performance_threshold = 0.95  # Allow up to 5% degradation
            is_performing = (
                actual_accuracy >= global_metrics['accuracy'] * performance_threshold and
                actual_loss <= global_metrics['loss'] * (1 + honesty_threshold)  # Allow 10% worse loss
            )
            
            # Log validation checks
            self.logger.info("Validation Checks:")
            self.logger.info(f"Honesty Check: {'PASS' if is_honest else 'FAIL'}")
            self.logger.info(f"Performance Check: {'PASS' if is_performing else 'FAIL'}")
            
            if not is_honest:
                self.logger.warning("Model update rejected: Reported metrics don't match actual performance")
                self.logger.warning(f"Metric gaps - Loss: {abs(reported_loss - actual_loss):.4f}, "
                                  f"Accuracy: {abs(reported_accuracy - actual_accuracy):.4f}")
            
            if not is_performing:
                self.logger.warning("Model update rejected: Performance below acceptable threshold")
                self.logger.warning(f"Required min accuracy: {global_metrics['accuracy'] * performance_threshold:.4f}")
                self.logger.warning(f"Required max loss: {global_metrics['loss'] * (1 + honesty_threshold):.4f}")
            
            # Both checks must pass for validation
            is_valid = is_honest and is_performing
            
            self.logger.info(f"Final validation result: {'VALID' if is_valid else 'INVALID'}")
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Error in model validation: {e}")
            traceback.print_exc()
            return False
    
    def _get_global_model_info(self) -> Optional[Dict]:
        """Get current global model information"""
        try:
            with self.node.state_lock:
                if 'global_model' in self.node.state:
                    return json.loads(self.node.state['global_model'])
            return None
        except Exception as e:
            self.logger.error(f"Error getting global model info: {e}")
            return None

    def load_model(self, model_path):
        self.models[model_path] = torch.load(model_path)

    def handle_model_request(self, message: Dict):
        """Handle a request for the global model"""
        client_id = message.get('client_id', '')
        request_id = message.get('request_id', '')
        
        self.logger.info(f"Handling model request from client {client_id}")
        
        # Check if we have global model info in our state
        global_model_info = None
        with self.node.state_lock:
            if 'global_model' in self.node.state:
                try:
                    global_model_info = json.loads(self.node.state['global_model'])
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
            with self.node.state_lock:
                if 'global_model' in self.node.state:
                    try:
                        global_model_info = json.loads(self.node.state['global_model'])
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
            'client_id': f"node-{self.node.node_id}",
            'timestamp': timestamp,
            'operation': operation,
            'request_id': request_id,
            'model_data': model_data  # Include full model data
        }
        
        # Store request in PBFT module
        digest = hashlib.sha256(f"{request_id}:{operation}".encode()).hexdigest()
        self.pbft.store_request(request_id, {
            'client_id': f"node-{self.node.node_id}",
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
