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

from components.message_handler import MessageHandler
from components.model_manager import ModelManager
from components.message_types import MessageType

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

        self.message_handler = MessageHandler(self, self.server_socket)
        self.model_manager = ModelManager(self, test_set, self.flower_client)
        
        self.running = True
        self.server_thread = threading.Thread(target=self.message_handler.start_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Start execution thread
        self.execution_thread = threading.Thread(target=self.process_execution_queue)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        
        self.logger.info(f"Node {self.node_id} started on {self.host}:{self.port}")

        self.is_primary = self.pbft.is_primary_node()

        # Add a longer delay before the primary node creates the initial global model
        if self.pbft.is_primary_node():
            # Schedule the creation of the initial model after a longer delay
            threading.Timer(3.0, self.create_initial_global_model_request).start()
        
        # Model aggregation parameters
        self.update_threshold = 3  # Number of updates before aggregation
        
        # Track model updates
        self.pending_updates = []  # List of validated updates since last aggregation
        self.global_model_version = 1  # Current global model version
    
    def create_initial_global_model_request(self):
        self.logger.info(f"Primary node {self.node_id} creating initial global model")
        try: 
            request_id = "initial_global_model"
            operation = "CREATE_GLOBAL_MODEL"
            digest = hashlib.sha256(f"{request_id}:{operation}".encode()).hexdigest()
            
            request = {
                'type': MessageType.REQUEST,
                'client_id': "system",
                'timestamp': int(time.time() * 1000),
                'operation': operation,
                'digest': digest,
                'request_id': request_id
            }
            
            # First make sure all nodes are aware of this request
            self.message_handler.broadcast(request)
            self.pbft.start_consensus(request_id)
            
            self.logger.info(f"Primary status: {self.node_id} is the primary node")

        except Exception as e:
            self.logger.error(f"Error creating initial global model: {e}")
            traceback.print_exc()
            return None

    def process_message(self, message: Dict):
        """Process incoming messages based on their type"""
        msg_type = message.get('type')
        
        if msg_type == MessageType.REQUEST:
            self.handle_request(message)
        elif msg_type in [MessageType.PRE_PREPARE, MessageType.PREPARE, MessageType.COMMIT, MessageType.VIEW_CHANGE, MessageType.NEW_VIEW, MessageType.HEARTBEAT]:
            self.pbft.process_message(message)
        elif msg_type == MessageType.BLOCK_SYNC:
            self.handle_block_sync(message)
        elif msg_type == MessageType.STATE_SYNC:
            self.handle_state_sync(message)
        elif msg_type == MessageType.VIEW_SYNC:
            self.handle_view_sync(message)
        elif msg_type == MessageType.NODE_JOIN:
            self.handle_node_join(message)
        elif msg_type == MessageType.MODEL_REQUEST:
            self.model_manager.handle_model_request(message)
        elif msg_type == MessageType.VALIDATION_RESULT:
            self.handle_validation_result(message)
        elif msg_type == MessageType.VALIDATION_FAILED:
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
            if any(operation.startswith(op) for op in ['UPDATE_MODEL', 'CREATE_GLOBAL_MODEL', 'UPDATE_GLOBAL_MODEL']):
                result, state_changed = self.model_manager.handle_operation(operation, request, sequence)
                
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
                        self.message_handler.broadcast(block_sync)
        
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
                'type': MessageType.VIEW_SYNC,
                'sender': self.node_id,
                'view': self.pbft.view,
                'primary': self.pbft.view % len(self.nodes)
            }
            self.message_handler.send_message(new_node, view_sync)
            
            # Then sync all blocks
            for block in self.blockchain.blocks:
                block_sync = {
                    'type': MessageType.BLOCK_SYNC,
                    'sender': self.node_id,
                    'block': block.to_dict(),
                    'sequence': block.index,
                    'view': self.pbft.view  # Include current view
                }
                self.message_handler.send_message(new_node, block_sync)
            
            # Sync state
            state_sync = {
                'type': MessageType.STATE_SYNC,
                'sender': self.node_id,
                'state': self.state,
                'last_executed_seq': self.last_executed_seq
            }
            self.message_handler.send_message(new_node, state_sync)
            
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
                'type': MessageType.NODE_JOIN,
                'sender': self.node_id,
                'view': view
            }
            self.message_handler.broadcast(join_msg)

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
                    'type': MessageType.STATE_SYNC,
                    'sender': self.node_id,
                    'state': self.state,
                    'last_executed_seq': self.last_executed_seq,
                    'view': self.pbft.view
                }
                self.message_handler.send_message(target_node, state_sync)
                
                # Send all blocks
                for block in self.blockchain.blocks:
                    block_sync = {
                        'type': MessageType.BLOCK_SYNC,
                        'sender': self.node_id,
                        'block': block.to_dict(),
                        'sequence': block.index,
                        'view': self.pbft.view
                    }
                    self.message_handler.send_message(target_node, block_sync)
                
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
                    'type': MessageType.VIEW_CHANGE,
                    'new_view': new_view,
                    'last_seq': self.pbft.sequence_number,
                    'sender': self.node_id,
                    'prepared': {},
                    'reason': 'selective_censorship',
                    'censored_requests': unreported
                }
                
                self.pbft.in_view_change = True
                self.pbft.process_message(view_change_msg)
                self.message_handler.broadcast(view_change_msg)
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
                'type': MessageType.BLOCK_SYNC,
                'sender': self.node_id,
                'block': new_block.to_dict(),
                'sequence': 0,  # Not part of consensus
                'view': self.pbft.view
            }
            self.message_handler.broadcast(block_sync, exclude_self=True)
            
            return new_block

    def validate_operation(self, sequence: int, request: Dict):
        """Perform validation check for an operation"""
        operation = request.get('operation', '')
        request_id = request.get('request_id', '')
        view = request.get('view', 0)
        validation_key = f"v{view}-s{sequence}"
        
        self.logger.info(f"Validating operation for seq {sequence}, view {view}: {operation}")
        
        try:
            # Handle model updates
            if operation.startswith('UPDATE_MODEL '):
                result, is_valid = self.model_manager.validate_model_update(operation, request)
                self._track_model_validation(validation_key, operation, request.get('client_id', 'unknown'), is_valid)
                return is_valid
            
            # Handle SET operations
            elif operation.startswith('SET '):
                parts = operation.split(' ', 2)
                if len(parts) == 3:
                    _, key, value = parts
                    try:
                        value_int = int(value)
                        is_valid = value_int > self.node_id
                    except ValueError:
                        is_valid = True  # Non-numeric values are valid
                else:
                    is_valid = False
            
            # All other operations are valid by default
            else:
                is_valid = True
            
            # Record and broadcast validation result
            self._record_validation_result(validation_key, is_valid, sequence, view, request_id)
            return is_valid
        
        except Exception as e:
            self.logger.error(f"Error during validation: {e}")
            return True  # Default to valid in case of errors

    def _track_model_validation(self, validation_key: str, model_path: str, 
                              client_id: str, is_valid: bool):
        """Track model validation results"""
        if not hasattr(self, 'model_validation_log'):
            self.model_validation_log = {}
        
        if validation_key not in self.model_validation_log:
            self.model_validation_log[validation_key] = {
                'client_id': client_id,
                'model_path': model_path,
                'votes': {},
                'result': None
            }
        
        # Record vote
        self.model_validation_log[validation_key]['votes'][self.node_id] = is_valid
        
        # Log voting status
        votes = self.model_validation_log[validation_key]['votes']
        votes_for = sum(1 for v in votes.values() if v)
        votes_against = len(votes) - votes_for
        
        self.logger.info(f"Model validation votes for {client_id}: {votes_for} FOR, {votes_against} AGAINST")

    def _record_validation_result(self, validation_key: str, is_valid: bool, 
                                sequence: int, view: int, request_id: str):
        """Record and broadcast validation result"""
        # Store validation
        if not hasattr(self, 'validations'):
            self.validations = {}
        self.validations[validation_key] = is_valid
        
        # Create validation message
        validation_msg = {
            'type': MessageType.VALIDATION_RESULT,
            'sender': self.node_id,
            'sequence': sequence,
            'view': view,
            'request_id': request_id,
            'is_valid': is_valid
        }
        
        # Broadcast and process
        self.message_handler.broadcast(validation_msg)
        self.handle_validation_result(validation_msg)

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
                'type': MessageType.VALIDATION_FAILED,
                'sender': self.node_id,
                'sequence': sequence,
                'view': view,
                'request_id': request_id
            }
            self.message_handler.broadcast(validation_failed_msg)

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