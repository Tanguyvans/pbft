import time
import socket
import json
import logging
import hashlib
from sklearn.model_selection import train_test_split

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
        
        return True 

    def get_global_model(self):
        """Request the current global model from the network"""
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
            model_params = None
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
                    
                    if response_data and not got_response:
                        response = json.loads(response_data.decode('utf-8'))
                        if response.get('status') == 'success':
                            self.logger.info(f"Received global model from {addr}")
                            model_params = response.get('model_params')
                            got_response = True
                            # Don't break here - keep accepting connections to avoid refused connections
                        else:
                            self.logger.error(f"Error from node: {response.get('message')}")
                except socket.timeout:
                    # If we already got a response, we can stop waiting
                    if got_response:
                        break
                    continue
                except Exception as e:
                    self.logger.error(f"Error receiving response: {e}")
                    continue
            
            return model_params
        
        finally:
            # Always close the response socket
            time.sleep(0.5)  # Give other nodes a chance to connect before closing
            response_socket.close()

    def train(self):       
        global_model_params = self.get_global_model()

        print("the glboal model params are: ", len(global_model_params))

        # res, metrics = self.flower_client.fit(self.global_model_weights, self.id, {})
        # test_metrics = self.flower_client.evaluate(res, {'name': f'Client {self.id}'})

        # with open(self.save_results + "output.txt", "a") as f:
        #     f.write(f"client {self.id}: "
        #             f"data:{metrics['len_train']} "
        #             f"train: {metrics['len_train']} "
        #             f"train: {metrics['train_loss']} {metrics['train_acc']} "
        #             f"val: {metrics['val_loss']} {metrics['val_acc']} "
        #             f"test: {test_metrics['test_loss']} {test_metrics['test_acc']}\n")
 
        # return res, metrics, test_metrics