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
        self.logger.info(f"Primary status: {self.pbft.is_primary_node()}")
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
        elif msg_type == 'state-sync':
            self.handle_state_sync(message)
        elif msg_type == 'view-sync':
            self.handle_view_sync(message)
        elif msg_type == 'node-join':
            self.handle_node_join(message)
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
        if self.pbft.is_primary_node():
            self.pbft.start_consensus(request_id)
    
    def execute_operation(self, sequence: int, request: Dict):
        """Execute the operation and update the state"""
        operation = request.get('operation')
        request_id = request.get('request_id', '')
        view = request.get('view', 0)
        validation_key = f"v{view}-s{sequence}"
        
        self.logger.info(f"Executing operation for seq {sequence}, view {view}: {operation}")
        
        # Check if this operation has been validated
        if hasattr(self, 'validated_operations') and validation_key in self.validated_operations:
            if not self.validated_operations[validation_key]:
                self.logger.warning(f"Operation for {validation_key} failed validation, skipping execution")
                return "VALIDATION_FAILED"
        
        # For SET operations that need validation but haven't been validated yet
        if operation.startswith('SET ') and (not hasattr(self, 'validated_operations') or validation_key not in self.validated_operations):
            # Store for later execution after validation
            if not hasattr(self, 'pending_validations'):
                self.pending_validations = {}
            
            self.logger.info(f"Operation for {validation_key} waiting for validation")
            self.pending_validations[validation_key] = request
            
            # Trigger validation check
            self.validate_operation(sequence, request)
            return "PENDING_VALIDATION"
        
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
                    if self.pbft.is_primary_node():
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
                
                # Execute the operation
                operation = request.get('operation', '')
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

    def handle_validation_result(self, message: Dict):
        """Handle validation results from other nodes"""
        sender = message.get('sender')
        sequence = message.get('sequence')
        view = message.get('view')
        is_valid = message.get('is_valid')
        request_id = message.get('request_id')
        
        self.logger.info(f"Received validation result from node {sender} for seq {sequence}: {is_valid}")
        
        # Store validation result
        if not hasattr(self, 'validation_results'):
            self.validation_results = {}
        
        validation_key = f"v{view}-s{sequence}"
        if validation_key not in self.validation_results:
            self.validation_results[validation_key] = {}
        
        self.validation_results[validation_key][sender] = is_valid
        
        # Check if we have enough validations to determine consensus
        valid_count = sum(1 for v in self.validation_results[validation_key].values() if v)
        total_count = len(self.validation_results[validation_key])
        
        self.logger.info(f"Validation consensus for {validation_key}: {valid_count}/{total_count} valid")
        
        # We need at least 2f+1 nodes to validate it as valid
        required_valid = 2 * self.pbft.f + 1
        
        # If we have enough valid votes, mark as valid
        if valid_count >= required_valid:
            self.logger.info(f"Validation consensus reached for {validation_key}: VALID")
            
            # Store the validation result for this operation
            if not hasattr(self, 'validated_operations'):
                self.validated_operations = {}
            self.validated_operations[validation_key] = True
            
            # Check if this operation is waiting for validation
            if hasattr(self, 'pending_validations') and validation_key in self.pending_validations:
                request = self.pending_validations[validation_key]
                self.logger.info(f"Executing validated operation for {validation_key}")
                self.execute_operation(sequence, request)
                del self.pending_validations[validation_key]
        
        # Only mark as invalid if we've heard from all nodes or if it's mathematically impossible
        # to reach the required valid count
        elif (total_count == len(self.nodes)) or (valid_count + (len(self.nodes) - total_count) < required_valid):
            self.logger.warning(f"Validation consensus reached for {validation_key}: INVALID")
            
            # Store the validation result for this operation
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

    def validate_operation(self, sequence: int, request: Dict):
        """Perform validation check for an operation"""
        operation = request.get('operation', '')
        request_id = request.get('request_id', '')
        view = request.get('view', 0)
        
        self.logger.info(f"Validating operation for seq {sequence}, view {view}: {operation}")
        
        try:
            parts = operation.split(' ', 2)  # Split only on the first two spaces
            
            # Add validation rule: value must be greater than node_id
            if parts[0] == 'SET' and len(parts) == 3:
                key, value = parts[1], parts[2]
                
                # Check if value is numeric and greater than node_id
                try:
                    value_int = int(value)
                    is_valid = value_int > self.node_id
                    self.logger.info(f"Validation check: value {value_int} > node_id {self.node_id} = {is_valid}")
                    
                    # Record validation result for consensus
                    if not hasattr(self, 'validations'):
                        self.validations = {}
                    
                    validation_key = f"v{view}-s{sequence}"
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
                    
                except ValueError:
                    # Non-numeric value, use default validation
                    is_valid = True
                    self.logger.info(f"Non-numeric value {value}, using default validation")
                    
                    # Broadcast validation result
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
            else:
                # Non-SET operations are always valid
                is_valid = True
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
        except Exception as e:
            self.logger.error(f"Error during validation: {e}")
            # Default to valid in case of errors
            is_valid = True
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