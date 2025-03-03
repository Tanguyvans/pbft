import socket
import threading
import json
import time
import hashlib
import random
from typing import Dict, List, Set, Tuple, Optional
import logging
import queue
from blockchain import Blockchain
from block import Block
from pbft import PBFT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PBFTNode:
    def __init__(self, node_id: int, host: str, port: int, nodes_config: List[Dict]):
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
        
        # Initialize blockchain with a deterministic genesis block
        self.blockchain = Blockchain()
        # Remove the default genesis block
        self.blockchain.blocks = []
        # Create a deterministic genesis block with fixed timestamp and nonce
        genesis_block = Block(
            index=0,
            data={"message": "Genesis Block"},
            model_type="genesis",
            storage_reference="",
            calculated_hash="",
            participants=["system"],
            previous_hash=""
        )
        # Set fixed values to ensure identical hash across all nodes
        genesis_block.timestamp = 1000000000  # Fixed timestamp
        genesis_block.nonce = 0
        self.blockchain.add_block(genesis_block)
        
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
        self.logger.info(f"Primary status: {self.pbft.is_primary}")
        self.logger.info(f"Genesis block hash: {genesis_block.current_hash}")
    
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
            # Pass PBFT protocol messages to the PBFT module
            self.pbft.process_message(message)
        elif msg_type == 'block-sync':
            self.handle_block_sync(message)
        elif msg_type == 'ping':
            self.handle_ping(message)
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
        
        # Calculate request digest
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
        if self.pbft.is_primary:
            self.pbft.start_consensus(request_id)
    
    def execute_operation(self, sequence: int, request: Dict):
        """Execute the operation and update the state"""
        operation = request.get('operation')
        request_id = request.get('request_id', '')
        view = request.get('view', 0)
        self.logger.info(f"Executing operation for seq {sequence}, view {view}: {operation}")
        
        result = None
        state_changed = False
        
        # Parse and execute the operation (simple key-value operations)
        try:
            parts = operation.split(' ', 2)  # Split only on the first two spaces
            
            if parts[0] == 'SET' and len(parts) == 3:
                key, value = parts[1], parts[2]
                with self.state_lock:
                    self.state[key] = value
                    self.logger.info(f"SET {key} = {value}")
                    result = f"SET {key} = {value}"
                    state_changed = True
                    self.logger.info(f"State changed: {state_changed}")
            
            elif parts[0] == 'GET' and len(parts) == 2:
                key = parts[1]
                with self.state_lock:
                    value = self.state.get(key, "NULL")
                    self.logger.info(f"GET {key} = {value}")
                    result = f"GET {key} = {value}"
                # GET operations don't modify state, so no new block needed
            
            elif parts[0] == 'DELETE' and len(parts) == 2:
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
            else:
                self.logger.warning(f"Unknown operation format: {operation}")
                result = f"UNKNOWN_OPERATION: {operation}"
        except Exception as e:
            self.logger.error(f"Error executing operation: {e}")
            result = f"ERROR: {str(e)}"
        
        # Create a new block for state-changing operations
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
                    if self.pbft.is_primary:
                        self.logger.info(f"Primary node creating block for {block_id}")
                    
                    new_block = self.blockchain.create_block(
                        data={
                            'operation': operation,
                            'sequence': sequence,
                            'view': view,
                            'block_id': block_id,
                            'request_id': request_id,
                            'result': result,
                            'state_snapshot': self.state.copy()
                        },
                        model_type="key-value-operation",
                        storage_reference=f"op-{block_id}",
                        calculated_hash=hashlib.sha256(str(self.state).encode()).hexdigest(),
                        participants=[str(self.node_id)]
                    )
                    self.blockchain.add_block(new_block)
                    self.logger.info(f"Created block #{new_block.index} for {block_id}, hash: {new_block.current_hash[:10]}...")
                    
                    # If this node is the primary, broadcast the new block to all nodes
                    if self.pbft.is_primary:
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
                
                # Check if we've already executed this request
                request_id = request.get('request_id', '')
                if request_id and request_id in self.executed_requests:
                    self.logger.info(f"Request {request_id} already executed, skipping")
                    continue
                
                # Check if we've already executed this sequence
                if seq <= self.last_executed_seq:
                    self.logger.info(f"Sequence {seq} already executed, skipping")
                    continue
                
                # Execute the operation
                operation = request.get('operation', '')
                result = self.execute_operation(seq, {
                    'operation': operation,
                    'request_id': request_id,
                    'view': self.pbft.view  # Include the view
                })
                
                # Mark as executed
                self.last_executed_seq = seq
                if request_id:  # Only add to executed_requests if request_id is not empty
                    self.executed_requests.add(request_id)
                
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
        if sender != primary_id:
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
    
    # Callback methods for PBFT
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
    
    def on_view_change(self, old_view, new_view):
        """Called by PBFT when a view change occurs"""
        self.logger.info(f"View changed from {old_view} to {new_view}")
        
        # Create a mapping of executed sequences for the old view
        with self.state_lock:
            # Store the highest sequence number executed in the old view
            if not hasattr(self, 'view_executed_seqs'):
                self.view_executed_seqs = {}
            
            self.view_executed_seqs[old_view] = self.last_executed_seq
            
            # Reset the last executed sequence for the new view
            # This is critical - we need to track execution separately for each view
            self.last_executed_seq = 0
            
            self.logger.info(f"Reset last executed sequence to 0 for view {new_view}")
            self.logger.info(f"View execution history: {self.view_executed_seqs}")
        
        # Clear the execution queue
        while not self.execution_queue.empty():
            try:
                self.execution_queue.get_nowait()
            except queue.Empty:
                break
        
        # Don't clear executed_requests - we still want to avoid duplicate executions
    
    def handle_ping(self, message: Dict):
        """Handle ping messages from clients"""
        client_id = message.get('client_id')
        timestamp = message.get('timestamp')
        
        # Send response with primary status
        response = {
            'type': 'pong',
            'is_primary': self.pbft.is_primary,
            'view': self.pbft.view,
            'timestamp': timestamp
        }
        
        # Since we don't have the client socket directly, we'll need to
        # modify the handle_client method to keep the socket open for response 