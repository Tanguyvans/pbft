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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PBFTNode:
    def __init__(self, node_id: int, host: str, port: int, nodes_config: List[Dict]):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.nodes = nodes_config
        self.logger = logging.getLogger(f"Node-{self.node_id}")
        
        # PBFT state
        self.view = 0  # Current view number
        self.sequence_number = 0  # Sequence number for requests
        self.is_primary = (self.node_id == self.view % len(self.nodes))
        
        # Message logs
        self.request_log = {}  # Store client requests
        self.pre_prepare_log = {}  # Store pre-prepare messages
        self.prepare_log = {}  # Store prepare messages
        self.commit_log = {}  # Store commit messages
        
        # Execution state
        self.last_executed_seq = 0
        self.state = {}  # Simple key-value store as the state
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
        self.sequence_lock = threading.Lock()
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
        self.execution_thread = threading.Thread(target=self.execution_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        
        self.logger.info(f"Node {self.node_id} started on {self.host}:{self.port}")
        self.logger.info(f"Primary status: {self.is_primary}")
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
        """Send a message to another node"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((target_node['host'], target_node['port']))
            s.sendall(json.dumps(message).encode('utf-8'))
            s.close()
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
        elif msg_type == 'pre-prepare':
            self.handle_pre_prepare(message)
        elif msg_type == 'prepare':
            self.handle_prepare(message)
        elif msg_type == 'commit':
            self.handle_commit(message)
        elif msg_type == 'block-sync':
            self.handle_block_sync(message)
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
        
        self.request_log[request_id] = {
            'client_id': client_id,
            'timestamp': timestamp,
            'operation': operation,
            'digest': digest
        }
        
        # If this node is the primary, initiate the PBFT protocol
        if self.is_primary:
            with self.sequence_lock:
                self.sequence_number += 1
                seq_num = self.sequence_number
            
            # Create pre-prepare message
            pre_prepare = {
                'type': 'pre-prepare',
                'view': self.view,
                'sequence': seq_num,
                'digest': digest,
                'request_id': request_id,
                'sender': self.node_id
            }
            
            # Store pre-prepare message
            key = f"{self.view}:{seq_num}"
            self.pre_prepare_log[key] = pre_prepare
            
            # Broadcast pre-prepare message
            self.broadcast(pre_prepare)
            self.logger.info(f"Sent pre-prepare for request {request_id}, seq {seq_num}")
            
            # Primary also sends prepare message
            self.handle_pre_prepare(pre_prepare)
    
    def handle_pre_prepare(self, message: Dict):
        """Handle pre-prepare message"""
        view = message.get('view')
        sequence = message.get('sequence')
        digest = message.get('digest')
        request_id = message.get('request_id')
        sender = message.get('sender')
        
        self.logger.info(f"Received pre-prepare from {sender} for seq {sequence}")
        
        # Verify the message
        if view != self.view:
            self.logger.warning(f"View mismatch: {view} != {self.view}")
            return
        
        # Store pre-prepare message
        key = f"{view}:{sequence}"
        self.pre_prepare_log[key] = message
        
        # Send prepare message
        prepare = {
            'type': 'prepare',
            'view': view,
            'sequence': sequence,
            'digest': digest,
            'request_id': request_id,
            'sender': self.node_id
        }
        
        # Store and broadcast prepare message
        if key not in self.prepare_log:
            self.prepare_log[key] = {}
        self.prepare_log[key][self.node_id] = prepare
        
        self.broadcast(prepare)
        self.logger.info(f"Sent prepare for seq {sequence}")
    
    def handle_prepare(self, message: Dict):
        """Handle prepare message"""
        view = message.get('view')
        sequence = message.get('sequence')
        digest = message.get('digest')
        sender = message.get('sender')
        request_id = message.get('request_id')
        
        self.logger.info(f"Received prepare from {sender} for seq {sequence}")
        
        # Verify the message
        if view != self.view:
            self.logger.warning(f"View mismatch: {view} != {self.view}")
            return
        
        # Store prepare message
        key = f"{view}:{sequence}"
        if key not in self.prepare_log:
            self.prepare_log[key] = {}
        self.prepare_log[key][sender] = message
        
        # Check if we have enough prepare messages (2f+1 including our own)
        f = (len(self.nodes) - 1) // 3  # Max number of faulty nodes
        if key in self.pre_prepare_log and len(self.prepare_log[key]) >= 2*f + 1:
            # We have enough prepare messages, send commit
            if key not in self.commit_log or self.node_id not in self.commit_log[key]:
                commit = {
                    'type': 'commit',
                    'view': view,
                    'sequence': sequence,
                    'digest': digest,
                    'request_id': request_id,
                    'sender': self.node_id
                }
                
                # Store and broadcast commit message
                if key not in self.commit_log:
                    self.commit_log[key] = {}
                self.commit_log[key][self.node_id] = commit
                
                self.broadcast(commit)
                self.logger.info(f"Sent commit for seq {sequence}")
    
    def handle_commit(self, message: Dict):
        """Handle commit messages"""
        sender = message.get('sender')
        view = message.get('view')
        sequence = message.get('sequence')
        digest = message.get('digest')
        request_id = message.get('request_id')
        
        self.logger.info(f"Received commit from {sender} for seq {sequence}")
        
        # Validate the message
        if view != self.view:
            self.logger.warning(f"Ignoring commit from different view: {view}")
            return
        
        # Create a key for this message
        key = f"{view}:{sequence}"
        
        # Store the commit message
        if key not in self.commit_log:
            self.commit_log[key] = {}
        self.commit_log[key][sender] = message
        
        # Check if we have enough commit messages (2f+1)
        f = (len(self.nodes) - 1) // 3  # Max number of faulty nodes
        if (key in self.pre_prepare_log and 
            len(self.commit_log[key]) >= 2*f + 1):
            
            # Find the request
            pre_prepare = self.pre_prepare_log[key]
            request_id = pre_prepare.get('request_id')
            
            if request_id in self.request_log and request_id not in self.executed_requests:
                request = self.request_log[request_id]
                
                # CRITICAL: Execute the operation immediately after consensus
                self.logger.info(f"CONSENSUS REACHED for operation: {request.get('operation')} (seq: {sequence})")
                
                # Execute the operation
                with self.state_lock:
                    operation = request.get('operation')
                    parts = operation.split(' ', 2)
                    
                    if parts[0] == 'SET' and len(parts) == 3:
                        key, value = parts[1], parts[2]
                        self.state[key] = value
                        self.logger.info(f"SET {key} = {value}")
                        result = f"SET {key} = {value}"
                        
                    elif parts[0] == 'GET' and len(parts) == 2:
                        key = parts[1]
                        value = self.state.get(key, "NULL")
                        self.logger.info(f"GET {key} = {value}")
                        result = f"GET {key} = {value}"
                        
                    elif parts[0] == 'DELETE' and len(parts) == 2:
                        key = parts[1]
                        if key in self.state:
                            del self.state[key]
                            self.logger.info(f"DELETE {key}")
                            result = f"DELETE {key} = SUCCESS"
                        else:
                            result = f"DELETE {key} = KEY_NOT_FOUND"
                    else:
                        self.logger.warning(f"Unknown operation format: {operation}")
                        result = f"UNKNOWN_OPERATION: {operation}"
                    
                    self.executed_requests.add(request_id)
                    self.last_executed_seq = sequence
                    
                    # Only the primary node creates blocks for state-changing operations
                    if self.is_primary and parts[0] in ['SET', 'DELETE']:
                        self.logger.info(f"PRIMARY NODE: Creating block for operation: {operation}")
                        with self.blockchain_lock:
                            new_block = self.blockchain.create_block(
                                data={
                                    'operation': operation,
                                    'sequence': sequence,
                                    'request_id': request_id,
                                    'result': result,
                                    'state_snapshot': self.state.copy()
                                },
                                model_type="key-value-operation",
                                storage_reference=f"op-{sequence}",
                                calculated_hash=hashlib.sha256(str(self.state).encode()).hexdigest(),
                                participants=[str(self.node_id)]
                            )
                            self.blockchain.add_block(new_block)
                            
                            # Broadcast the new block to all nodes
                            block_message = {
                                'type': 'block-sync',
                                'sender': self.node_id,
                                'block': new_block.to_dict(),
                                'sequence': sequence
                            }
                            self.broadcast(block_message)
                            self.logger.info(f"PRIMARY NODE: Created and broadcast block #{new_block.index} for sequence {sequence}")
                            self.logger.info(f"PRIMARY NODE: Current blockchain length: {len(self.blockchain.blocks)}")
    
    def execution_loop(self):
        """Process operations in sequence order"""
        while self.running:
            try:
                # Get the next operation with the lowest sequence number
                sequence, request_id, request = self.execution_queue.get(timeout=0.1)
                
                # Check if this request has already been executed
                if request_id in self.executed_requests:
                    self.logger.info(f"Request {request_id} already executed, skipping")
                    continue
                
                # Check if this is the next sequence to execute
                if sequence > self.last_executed_seq + 1:
                    # Put it back in the queue and try again later
                    self.logger.info(f"Sequence {sequence} not ready yet, current last executed: {self.last_executed_seq}")
                    self.execution_queue.put((sequence, request_id, request))
                    time.sleep(0.05)  # Small delay to prevent CPU spinning
                    continue
                
                # Execute the operation
                self.logger.info(f"Executing operation for sequence {sequence}: {request.get('operation')}")
                with self.state_lock:
                    result = self.execute_operation(sequence, request)
                    self.executed_requests.add(request_id)
                    self.last_executed_seq = sequence
                
                # Only the primary node creates blocks
                if self.is_primary:
                    self.logger.info(f"Primary node creating block for operation: {request.get('operation')}")
                    with self.blockchain_lock:
                        new_block = self.blockchain.create_block(
                            data={
                                'operation': request.get('operation'),
                                'sequence': sequence,
                                'request_id': request_id,
                                'result': result,
                                'state_snapshot': self.state.copy()
                            },
                            model_type="key-value-operation",
                            storage_reference=f"op-{sequence}",
                            calculated_hash=hashlib.sha256(str(self.state).encode()).hexdigest(),
                            participants=[str(self.node_id)]
                        )
                        self.blockchain.add_block(new_block)
                        
                        # Broadcast the new block to all nodes
                        block_message = {
                            'type': 'block-sync',
                            'sender': self.node_id,
                            'block': new_block.to_dict(),
                            'sequence': sequence
                        }
                        self.broadcast(block_message)
                        self.logger.info(f"Created and broadcast block #{new_block.index} for sequence {sequence}")
            
            except queue.Empty:
                # No operations to execute, just wait
                time.sleep(0.05)
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
    
    def execute_operation(self, sequence: int, request: Dict):
        """Execute the operation and update the state"""
        operation = request.get('operation')
        self.logger.info(f"Executing operation for seq {sequence}: {operation}")
        
        result = None
        
        # Parse and execute the operation (simple key-value operations)
        try:
            parts = operation.split(' ', 2)  # Split only on the first two spaces
            
            if parts[0] == 'SET' and len(parts) == 3:
                key, value = parts[1], parts[2]
                with self.state_lock:
                    self.state[key] = value
                    self.logger.info(f"SET {key} = {value}")
                    result = f"SET {key} = {value}"
                
                # Only the primary node should create blocks
                if self.is_primary:
                    with self.blockchain_lock:
                        new_block = self.blockchain.create_block(
                            data={
                                'operation': operation,
                                'sequence': sequence,
                                'request_id': request.get('request_id', ''),
                                'result': result,
                                'state_snapshot': self.state.copy()
                            },
                            model_type="key-value-operation",
                            storage_reference=f"op-{sequence}",
                            calculated_hash=hashlib.sha256(str(self.state).encode()).hexdigest(),
                            participants=[str(self.node_id)]
                        )
                        self.blockchain.add_block(new_block)
                        self.logger.info(f"Created and broadcast block #{new_block.index} for sequence {sequence}")
                        
                        # Broadcast the new block to all nodes (including self)
                        block_sync = {
                            'type': 'block-sync',
                            'sender': self.node_id,
                            'block': new_block.to_dict(),
                            'sequence': sequence
                        }
                        self.broadcast(block_sync)
            
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
                        
                        # Only the primary node should create blocks
                        if self.is_primary:
                            with self.blockchain_lock:
                                new_block = self.blockchain.create_block(
                                    data={
                                        'operation': operation,
                                        'sequence': sequence,
                                        'request_id': request.get('request_id', ''),
                                        'result': result,
                                        'state_snapshot': self.state.copy()
                                    },
                                    model_type="key-value-operation",
                                    storage_reference=f"op-{sequence}",
                                    calculated_hash=hashlib.sha256(str(self.state).encode()).hexdigest(),
                                    participants=[str(self.node_id)]
                                )
                                self.blockchain.add_block(new_block)
                                self.logger.info(f"Created and broadcast block #{new_block.index} for sequence {sequence}")
                                
                                # Broadcast the new block to all nodes (including self)
                                block_sync = {
                                    'type': 'block-sync',
                                    'sender': self.node_id,
                                    'block': new_block.to_dict(),
                                    'sequence': sequence
                                }
                                self.broadcast(block_sync)
                    else:
                        result = f"DELETE {key} = KEY_NOT_FOUND"
            else:
                self.logger.warning(f"Unknown operation format: {operation}")
                result = f"UNKNOWN_OPERATION: {operation}"
        except Exception as e:
            self.logger.error(f"Error executing operation: {e}")
            result = f"ERROR: {str(e)}"
        
        self.logger.info(f"Current state: {self.state}")
        return result
    
    def create_and_propose_block(self):
        """Create a new block and propose it to the network"""
        # Create a new block with pending operations
        participants = [str(self.node_id)]  # Start with this node as a participant
        
        new_block = self.blockchain.create_block_from_pending(
            model_type="key-value-store",
            storage_reference=f"state-{int(time.time())}",
            calculated_hash=hashlib.sha256(str(self.state).encode()).hexdigest(),
            participants=participants
        )
        
        if new_block:
            # Reset counter
            self.operations_since_last_block = 0
            
            # Propose the block to other nodes
            block_proposal = {
                'type': 'block-proposal',
                'sender': self.node_id,
                'block': new_block.to_dict(),
                'timestamp': time.time()
            }
            
            self.broadcast(block_proposal)
            self.logger.info(f"Proposed new block #{new_block.index} with {len(new_block.data)} operations")
    
    def handle_block_sync(self, message: Dict):
        """Handle block synchronization from the primary node"""
        sender = message.get('sender')
        block_data = message.get('block')
        sequence = message.get('sequence')
        
        # Only accept blocks from the primary node
        if sender != self.view % len(self.nodes):
            self.logger.warning(f"Ignoring block from non-primary node {sender}")
            return
        
        self.logger.info(f"Received block sync from primary node {sender} for sequence {sequence}")
        
        # Recreate the block exactly as sent by the primary
        new_block = Block(
            index=block_data.get('index'),
            data=block_data.get('data'),
            model_type=block_data.get('model_type'),
            storage_reference=block_data.get('storage_reference'),
            calculated_hash=block_data.get('calculated_hash'),
            participants=block_data.get('participants'),
            previous_hash=block_data.get('previous_hash')
        )
        # Important: preserve exact timestamp and nonce to ensure identical hash
        new_block.timestamp = block_data.get('timestamp')
        new_block.nonce = block_data.get('nonce')
        
        # Add or replace the block in our blockchain
        with self.blockchain_lock:
            if new_block.index < len(self.blockchain.blocks):
                # Replace existing block
                self.blockchain.blocks[new_block.index] = new_block
                self.logger.info(f"Replaced block #{new_block.index} with block from primary")
            else:
                # Add new block
                self.blockchain.add_block(new_block)
                self.logger.info(f"Added block #{new_block.index} from primary node {sender}")
    
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
        self.server_socket.close()
        self.logger.info(f"Node {self.node_id} stopped") 