import time
import socket
import json
import logging
import hashlib
import traceback
from sklearn.model_selection import train_test_split
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from going_modular.model import Net, SimpleMNISTNet
from torch.utils.data import DataLoader, TensorDataset

from flowerclient import FlowerClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PBFTClient:
    def __init__(self, client_id: str, nodes_config: list, client_train_set, client_test_set):
        self.client_id = client_id
        self.nodes = nodes_config
        self.logger = logging.getLogger(f"Client-{self.client_id}")
        self.request_count = 0


        x_train, y_train = client_train_set
        x_test, y_test = client_test_set        

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,
                                                    stratify=None)
        
        print(f"the length of the train set is: {len(x_train)}")
        print(f"the length of the val set is: {len(x_val)}")
        print(f"the length of the test set is: {len(x_test)}")

        self.flower_client = FlowerClient.client(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test
        )
    
    def send_request(self, operation: str):
        """Send a request to the PBFT network"""
        self.request_count += 1
        timestamp = int(time.time() * 1000)
        
        request = {
            'type': 'request',
            'client_id': self.client_id,
            'timestamp': timestamp,
            'operation': operation,
            'request_id': f"{self.client_id}:{timestamp}"
        }
        
        # Try to find the primary node first
        primary_found = False
        for node in self.nodes:
            try:
                self.logger.info(f"Sending request to node {node['id']}: {operation}")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)  # Short timeout for primary check
                s.connect((node['host'], node['port']))
                
            except Exception:
                # Skip failed nodes
                continue
        
        # If primary not found or request to primary failed, send to all nodes
        if not primary_found:
            self.logger.warning("Primary node not found, sending to all available nodes")
            success_count = 0
            for node in self.nodes:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect((node['host'], node['port']))
                    s.sendall(json.dumps(request).encode('utf-8'))
                    s.close()
                    success_count += 1
                except Exception:
                    # Skip failed nodes
                    continue
            
            return success_count > 0
        
        # Log the full request for debugging
        self.logger.info(f"Sending request with client_id: {self.client_id}")
        self.logger.info(f"Full request: {json.dumps(request)}")
        
        return True 

    def get_global_model(self):
        """Request the current global model info from the network"""
        self.logger.info("Requesting global model from the network")
        
        # Create a unique request ID
        timestamp = int(time.time() * 1000)
        request_id = f"{self.client_id}:model:{timestamp}"
        
        # Create a digest for the request
        operation = "GET_GLOBAL_MODEL"
        digest = hashlib.sha256(f"{operation}:{request_id}".encode()).hexdigest()
        
        # Set up a socket server to receive the response
        client_port = 50000 + int(self.client_id.replace('client', ''))
        response_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        response_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            response_socket.bind(('localhost', client_port))
            response_socket.listen(5)
            response_socket.settimeout(2.0)  # Short timeout for accepting connections
            
            self.logger.info(f"Client listening for responses on port {client_port}")
            
            # Include client connection info for response
            request = {
                'type': 'model-request',
                'client_id': self.client_id,
                'timestamp': timestamp,
                'operation': operation,
                'request_id': request_id,
                'digest': digest,
                'client_host': 'localhost',
                'client_port': client_port
            }
            
            # Send request to all nodes
            for node in self.nodes:
                try:
                    self.logger.info(f"Requesting global model from node {node['id']}")
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1.0)  # Short timeout for sending request
                    s.connect((node['host'], node['port']))
                    s.sendall(json.dumps(request).encode('utf-8'))
                    s.close()
                except Exception as e:
                    self.logger.error(f"Error sending request to node {node['id']}: {e}")
                    continue
            
            # Wait for response from any node
            model_info = None
            start_time = time.time()
            got_response = False
            
            while time.time() - start_time < 5.0:  # Wait up to 5 seconds for a response
                try:
                    client_socket, addr = response_socket.accept()
                    self.logger.info(f"Received response from {addr}")
                    
                    response_data = b""
                    while True:
                        chunk = client_socket.recv(8192)
                        if not chunk:
                            break
                        response_data += chunk
                    
                    client_socket.close()

                    print("the response data is: ", response_data)
                    
                    if response_data:
                        try:
                            response = json.loads(response_data.decode('utf-8'))
                            self.logger.info(f"Parsed response: {response}")
                            
                            if response.get('status') == 'success' and not got_response:
                                self.logger.info(f"Received global model info from {addr}")
                                model_info = response
                                got_response = True
                                # Don't break here - keep accepting connections to avoid refused connections
                            elif response.get('status') == 'error':
                                self.logger.error(f"Error from node: {response.get('message')}")
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse response as JSON: {e}")
                            self.logger.debug(f"Raw response: {response_data}")
                except socket.timeout:
                    # If we already got a response, we can stop waiting
                    if got_response:
                        break
                    continue
                except Exception as e:
                    self.logger.error(f"Error receiving response: {e}")
                    continue
            
            if model_info:
                self.logger.info(f"Successfully received model info: {model_info}")
                # Store the model info for future reference
                self.global_model_info = model_info
                
                # Log the version information
                version = model_info.get('version', 1)
                self.logger.info(f"Global model version: v{version}")
            else:
                self.logger.error("Did not receive valid model info from any node")
            
            return model_info
        
        finally:
            # Always close the response socket
            time.sleep(0.5)  # Give other nodes a chance to connect before closing
            response_socket.close()

    def train(self, send_update=True, return_weights=False):
        """Train a local model based on the global model"""
        global_model_version_used = None # Variable to store the version
        try:
            # First, get the global model info
            model_info = self.get_global_model()
            
            if not model_info:
                self.logger.error("Cannot train without global model info")
                # Return version as None too
                return None, None, None, None 
            
            self.logger.info(f"Received global model info: {model_info}")
            
            # Extract model path and hash
            model_path = model_info.get('model_path')
            expected_hash = model_info.get('model_hash')
            architecture = model_info.get('architecture', 'simple_mnist_cnn')
            
            # Get the global model version - THIS IS IMPORTANT
            global_model_version_used = model_info.get('version', 1) # Store the version
            
            # Verify the model file exists
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return None, None, None, None
            
            # Verify the model hash
            with open(model_path, 'rb') as f:
                file_content = f.read()
                actual_hash = hashlib.sha256(file_content).hexdigest()

            if actual_hash != expected_hash:
                self.logger.warning(f"Model hash verification failed!")
                self.logger.warning(f"Expected: {expected_hash}")
                self.logger.warning(f"Actual: {actual_hash}")
                self.logger.warning("Model may have been tampered with or corrupted")
                self.logger.warning("Continuing with unverified model...")
            else:
                self.logger.info("Model hash verification successful!")
            
            # Evaluate the global model before training
            self.logger.info("Evaluating global model before training")
            pre_train_loss, pre_train_accuracy = self.evaluate_model(model_path=model_path, architecture=architecture)
            
            print("Step 1: Loading model architecture...")            
            # Force CPU usage to avoid MPS bus errors
            device = torch.device("cpu")
            print("Using CPU device for training (avoiding MPS due to stability issues)")
            
            # Initialize model
            if architecture == 'simple_mnist_cnn':
                 model = SimpleMNISTNet(num_classes=10).to(device)
            elif architecture == 'mobilenet_v2':
                 model = Net(num_classes=10, arch=architecture).to(device)
            else:
                 self.logger.error(f"Unknown architecture '{architecture}' during training init.")
                 return None, None, None, None
            
            print("Step 2: Loading model weights...")
            # Load the model weights based on file extension
            try:
                if model_path.endswith('.npz'):
                    # Load NPZ file
                    npz_data = np.load(model_path, allow_pickle=True)
                    
                    # Convert numpy arrays to PyTorch tensors
                    state_dict = {}
                    for key in npz_data.files:
                        # Skip any non-tensor metadata
                        try:
                            tensor = torch.from_numpy(npz_data[key])
                            state_dict[key] = tensor
                        except TypeError:
                            print(f"Skipping non-tensor key in NPZ file: {key}")
                    
                    # Check if we have a valid state dict
                    if state_dict:
                        try:
                            model.load_state_dict(state_dict)
                            print("Successfully loaded model weights from NPZ")
                        except Exception as e:
                            print(f"Error loading state dict: {e}")
                    else:
                        print("No valid tensors found in NPZ file")
                else:
                    print(f"Unsupported model format: {model_path}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print("Continuing with randomly initialized weights")
            
            print("Step 3: Preparing data...")
            # Prepare your training data - move to CPU
            x_train = self.flower_client.train_loader.dataset.tensors[0].to(device)
            y_train = self.flower_client.train_loader.dataset.tensors[1].to(device)
            
            # Create data loader with smaller batch size
            batch_size = 16  # Smaller batch size to avoid memory issues
            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            print("Step 4: Setting up training...")
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            
            # Add learning rate scheduler for better convergence
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
            
            print("Step 5: Starting training...")
            # Training loop
            model.train()
            num_epochs = 1  # Increased number of epochs for better learning
            best_loss = float('inf')
            best_model_state = None
            
            try:
                for epoch in range(num_epochs):
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    
                    for i, (inputs, labels) in enumerate(train_loader):
                        # Zero the gradients
                        optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()
                        
                        # Print statistics
                        running_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if i % 5 == 0:  # More frequent updates
                            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / total:.3f}, '
                                  f'accuracy: {100 * correct / total:.2f}%')
                
                epoch_loss = running_loss / total
                epoch_acc = correct / total
                print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc*100:.2f}%')
                
                # Update scheduler
                scheduler.step(epoch_loss)
                
                # Save best model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_state = model.state_dict().copy()
                    print(f"New best model saved! Loss: {epoch_loss:.4f}")
            
                print("Training completed successfully!")
                
                # Load the best model state
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("Loaded best model based on loss")
                
                # Convert final model weights to numpy
                numpy_dict = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

                # Evaluate the model after training
                self.logger.info("Evaluating model after training")
                post_train_loss, post_train_accuracy = self.evaluate_model(model=model, architecture=architecture)
                
                # Log improvement
                if pre_train_accuracy is not None and post_train_accuracy is not None:
                    improvement = post_train_accuracy - pre_train_accuracy
                    self.logger.info(f"Training improved accuracy by {improvement:.4f} ({improvement*100:.2f}%)")
                
                formatted_loss = float(post_train_loss) if post_train_loss is not None else None
                formatted_accuracy = float(post_train_accuracy) if post_train_accuracy is not None else None

                if return_weights:
                     # Return weights directly without saving or sending update
                     self.logger.info("Returning trained weights in memory (cluster simulation).")
                     # Return weights, loss, accuracy, AND the version used
                     return numpy_dict, formatted_loss, formatted_accuracy, global_model_version_used
                else:
                     # Save the trained model to disk
                     npz_dir = "models/npz"
                     os.makedirs(npz_dir, exist_ok=True)
                     timestamp = int(time.time())
                     npz_path = f"{npz_dir}/model_{self.client_id}_v{global_model_version_used}_{timestamp}.npz"
                     np.savez(npz_path, **numpy_dict)
                     self.logger.info(f"Model saved as NPZ: {npz_path} (based on global model v{global_model_version_used})")

                     # Conditionally send the trained model update request
                     if send_update:
                         if formatted_loss is not None and formatted_accuracy is not None:
                             self.send_trained_model(npz_path, formatted_loss, formatted_accuracy, global_model_version_used)
                         else:
                             self.logger.error("Cannot send model update due to evaluation failure.")
                     else:
                         # This case (send_update=False, return_weights=False) shouldn't normally be hit with the current run_pbft_network logic
                         self.logger.info("Saving model locally but not sending update.")

                     # Return path, loss, accuracy, AND the version used
                     return npz_path, post_train_loss, post_train_accuracy, global_model_version_used

            except Exception as e:
                print(f"Error during training: {str(e)}")
                traceback.print_exc()
                # Ensure we return None if training fails before saving/sending
                return None, None, None, None # Modified return on exception

            # Check if training completed successfully before proceeding
            if best_model_state is None:
                 print("Training failed, best model state not found.")
                 # Return version as None too
                 return None, None, None, None 

        except Exception as e:
            print(f"Error in train method: {str(e)}")
            traceback.print_exc()
            # Return Nones including for the version
            return None, None, None, None

    def send_trained_model(self, model_path, training_loss, training_accuracy, global_model_version=1):
        """Send the trained model back to the PBFT network as a model update
        
        Args:
            model_path: Path to the trained model file
            training_loss: Loss value from training
            training_accuracy: Accuracy value from training
            global_model_version: Version of the global model this update is based on
        """
        self.logger.info(f"Sending trained model {model_path} back to the network as an update (based on global model v{global_model_version})")
        
        # Create a unique request ID
        timestamp = int(time.time() * 1000)
        request_id = f"{self.client_id}:update_model_v{global_model_version}:{timestamp}"
        
        # Calculate model hash
        with open(model_path, 'rb') as f:
            file_content = f.read()
            model_hash = hashlib.sha256(file_content).hexdigest()
        
        # Format loss and accuracy values to avoid parsing issues
        formatted_loss = float(training_loss)
        formatted_accuracy = float(training_accuracy)
        
        # Create the request with UPDATE_MODEL operation
        # Format: UPDATE_MODEL model_path model_hash loss accuracy v{version}
        operation = f"UPDATE_MODEL {model_path} {model_hash} {formatted_loss} {formatted_accuracy} v{global_model_version}"
        
        # Log the operation string for debugging
        self.logger.info(f"Operation string: {operation}")
        
        request = {
            'type': 'request',
            'client_id': self.client_id,
            'timestamp': timestamp,
            'operation': operation,
            'request_id': request_id,
            'model_path': model_path,
            'model_hash': model_hash,
            'training_loss': formatted_loss,
            'training_accuracy': formatted_accuracy,
            'global_model_version': global_model_version  # Include version information
        }
        
        # Send request to all nodes
        success_count = 0
        for node in self.nodes:
            try:
                self.logger.info(f"Sending model update v{global_model_version} to node {node['id']}")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.0)  # Longer timeout for sending model info
                s.connect((node['host'], node['port']))
                s.sendall(json.dumps(request).encode('utf-8'))
                s.close()
                success_count += 1
            except Exception as e:
                self.logger.error(f"Error sending model update to node {node['id']}: {e}")
                continue
        
        if success_count > 0:
            self.logger.info(f"Successfully sent model update v{global_model_version} to {success_count} nodes")
            return True
        else:
            self.logger.error(f"Failed to send model update v{global_model_version} to any node")
            return False

    def evaluate_model(self, model_path=None, model=None, architecture=None):
        """Evaluate the model on the test dataset
        
        Args:
            model_path: Path to the model file (.pt or .npz)
            model: PyTorch model object (if already loaded)
            architecture: String identifier for the model architecture (e.g., 'simple_mnist_cnn')
        """
        self.logger.info("Evaluating model performance")
        
        try:            
            # Use CPU for evaluation to avoid MPS memory issues
            device = torch.device("cpu")
            self.logger.info("Using CPU device for evaluation (more stable)")
            
            # If model is not provided, load it from path
            if model is None and model_path is not None:
                # Determine architecture: use parameter, then check checkpoint, then default
                if architecture is None and model_path.endswith('.pt'):
                    try:
                        checkpoint = torch.load(model_path, map_location=device)
                        architecture = checkpoint.get('architecture')
                    except Exception:
                        pass # Ignore error if checkpoint doesn't contain architecture

                if architecture is None:
                    architecture = 'simple_mnist_cnn' # <<< CHANGED default if not provided/found
                    self.logger.info(f"Architecture not provided, defaulting to {architecture}")
                else:
                    self.logger.info(f"Using architecture: {architecture}")

                # --- Instantiate model based on architecture string ---
                if architecture == 'simple_mnist_cnn':
                     model_to_eval = SimpleMNISTNet(num_classes=10).to(device) # <<< USE NEW MODEL CLASS
                elif architecture == 'mobilenet_v2': # Keep old logic if needed
                     model_to_eval = Net(num_classes=10, arch=architecture).to(device)
                else:
                     self.logger.error(f"Unknown architecture '{architecture}' during evaluation init.")
                     return None, None # Fail evaluation
                # --- End Instantiate ---

                # Load weights into the correctly instantiated model
                if model_path.endswith('.npz'):
                    # Load NPZ file
                    npz_data = np.load(model_path, allow_pickle=True)
                    
                    # Convert numpy arrays to PyTorch tensors and load into model_to_eval
                    state_dict = {}
                    for key in npz_data.files:
                        # Skip any non-tensor metadata that might be in the file
                        try:
                            tensor = torch.from_numpy(npz_data[key])
                            state_dict[key] = tensor
                        except TypeError as e:
                            self.logger.warning(f"Skipping non-tensor key in NPZ file: {key} - {e}")
                    
                    # Check if we have a valid state dict
                    if state_dict:
                        try:
                            model_to_eval.load_state_dict(state_dict)
                            self.logger.info("Successfully loaded model weights from NPZ")
                        except Exception as e:
                            self.logger.error(f"Error loading state dict: {e}")
                            return None, None
                    else:
                        self.logger.error("No valid tensors found in NPZ file")
                        return None, None
                
                # Check if it's a PT file
                elif model_path.endswith('.pt'):
                    self.logger.info(f"Loading PT model from {model_path}")
                    # Load the PyTorch model
                    checkpoint = torch.load(model_path, map_location=device)
                    
                    if 'architecture' in checkpoint:
                        architecture = checkpoint['architecture']
                    
                    # Initialize model
                    model_to_eval = SimpleMNISTNet(num_classes=10).to(device)
                    
                    # Load weights
                    if 'model_state_dict' in checkpoint:
                        model_to_eval.load_state_dict(checkpoint['model_state_dict'])
                        self.logger.info("Successfully loaded model weights from PT")
                    else:
                        self.logger.warning("No model_state_dict found in checkpoint")
                        return None, None
                else:
                    self.logger.error(f"Unsupported model format: {model_path}")
                    return None, None
            
            # Set model to evaluation mode
            model_to_eval.eval()
            
            # Define loss function
            criterion = nn.CrossEntropyLoss()
            
            # Get test data from flower client and move to CPU
            x_test = self.flower_client.test_loader.dataset.tensors[0].to(device)
            y_test = self.flower_client.test_loader.dataset.tensors[1].to(device)
            
            # Create a proper test data loader with batching
            batch_size = 16  # Smaller batch size to avoid memory issues
            test_dataset = TensorDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Perform evaluation in batches
            with torch.no_grad():
                total_loss = 0.0
                correct = 0
                total = 0
                
                for inputs, labels in test_loader:
                    outputs = model_to_eval(inputs)
                    loss = criterion(outputs, labels)
                    
                    total_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                avg_loss = total_loss / total
                accuracy = correct / total
                
                # Display accuracy as percentage for clarity in logs
                self.logger.info(f"Evaluation results - Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
                
                # Calculate per-class accuracy
                class_correct = [0] * 10
                class_total = [0] * 10
                
                for inputs, labels in test_loader:
                    outputs = model_to_eval(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    for i in range(len(labels)):
                        label = labels[i].item()
                        class_total[label] += 1
                        if predicted[i] == labels[i]:
                            class_correct[label] += 1
                
                for i in range(10):
                    if class_total[i] > 0:
                        class_acc = 100 * class_correct[i] / class_total[i]
                        self.logger.info(f'Accuracy of class {i}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')
                
                # Return raw values (not multiplied by 100) for consistency with the rest of the code
                return avg_loss, accuracy
                
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            traceback.print_exc()
            return None, None
