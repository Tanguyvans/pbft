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

        # --- Validation Check ---
        # We need validation results *before* potentially modifying state
        # Check if validation is required for this operation type
        needs_validation = operation.startswith('UPDATE_MODEL ') or \
                           operation.startswith('CLUSTER_TRAIN_VALIDATE ') or \
                           operation.startswith('SET ') # Add other types needing validation

        if needs_validation:
            # Check if validation has already completed and failed
            if hasattr(self, 'validated_operations') and validation_key in self.validated_operations:
                if not self.validated_operations[validation_key]:
                    self.logger.warning(f"Operation for {validation_key} failed validation consensus, skipping execution.")
                    return "VALIDATION_FAILED"
                else:
                    # Validation completed and succeeded, proceed with execution
                    self.logger.info(f"Operation {validation_key} previously passed validation. Proceeding.")
            else:
                # Validation is needed, but result isn't available yet.
                # Defer execution and trigger validation if not already pending.
                if not hasattr(self, 'pending_validations'): self.pending_validations = {}

                if validation_key not in self.pending_validations: # Avoid redundant logging/triggering
                    self.logger.info(f"Operation {validation_key} requires validation, result not ready. Triggering validation and deferring execution.")
                    self.pending_validations[validation_key] = request
                    self.validate_operation(sequence, request) # Trigger validation now
                else:
                    # Already pending, just log it
                    self.logger.info(f"Operation {validation_key} still pending validation.")
                return "PENDING_VALIDATION" # Defer execution
        # --- End Validation Check ---


        # --- Duplicate Global Model Update Check ---
        if operation.startswith('UPDATE_GLOBAL_MODEL '):
            parts = operation.split(' ', 3)
            if len(parts) >= 4:
                version = int(parts[3])
                with self.state_lock:
                    try:
                        current_model_info = json.loads(self.state.get('global_model', '{}'))
                        current_version = current_model_info.get('version', 0)
                        if current_version >= version:
                            self.logger.warning(f"Skipping duplicate global model update for version {version}, current version is {current_version}")
                            return f"SKIPPED_DUPLICATE: Global model v{version} already processed"
                    except Exception as e:
                         self.logger.warning(f"Could not parse current global model state for duplicate check: {e}")
        # --- End Duplicate Check ---


        result = None
        state_changed = False
        block_model_type = "unknown" # Default block type
        parsed_client_ids = [] # Store parsed client IDs for block data if needed

        try:
            # --- Delegate Model Operations to ModelManager ---
            # Ensure CLUSTER_TRAIN_VALIDATE is included here for delegation
            if any(op_type in operation for op_type in ['UPDATE_MODEL', 'CREATE_GLOBAL_MODEL', 'UPDATE_GLOBAL_MODEL', 'CLUSTER_TRAIN_VALIDATE']):
                # Parse client_ids specifically for CLUSTER_TRAIN_VALIDATE *before* calling handle_operation
                # so ModelManager might use it if needed, and we can put it in the block later.
                if operation.startswith('CLUSTER_TRAIN_VALIDATE '):
                     match = re.match(
                         r"CLUSTER_TRAIN_VALIDATE.*?client_ids='(\[.*\])'.*", # Simplified regex to get client_ids
                         operation
                     )
                     if match:
                         client_ids_json = match.group(1)
                         try:
                             parsed_client_ids = json.loads(client_ids_json)
                             self.logger.debug(f"Parsed client IDs for {validation_key}: {parsed_client_ids}")
                         except json.JSONDecodeError:
                             self.logger.error(f"Failed to parse client_ids JSON in execute_operation: {client_ids_json}")
                             # Decide how to handle - maybe return error? For now, log and continue.
                     else:
                         self.logger.error(f"Could not parse client_ids from CLUSTER_TRAIN_VALIDATE operation: {operation}")

                # Let ModelManager handle the state update and determine block type
                # Ensure ModelManager.handle_operation actually handles 'CLUSTER_TRAIN_VALIDATE'
                result, state_changed, block_model_type = self.model_manager.handle_operation(operation, request, sequence)

            # --- Handle Key-Value Operations ---
            elif operation.startswith('SET '):
                # ... (SET logic remains the same) ...
                parts = operation.split(' ', 2)
                if len(parts) == 3:
                    key, value = parts[1], parts[2]
                    with self.state_lock:
                        self.state[key] = value
                    self.logger.info(f"SET {key} = {value}")
                    result = f"SET {key} = {value}"
                    state_changed = True
                    block_model_type = "key-value-operation" # Corrected type assignment location
                else:
                     result = "ERROR: Invalid SET format" # Handle error

            elif operation.startswith('GET '):
                # ... (GET logic remains the same) ...
                parts = operation.split(' ', 2)
                if len(parts) == 2:
                    key = parts[1]
                    with self.state_lock:
                        value = self.state.get(key, "NULL")
                    self.logger.info(f"GET {key} = {value}")
                    result = f"GET {key} = {value}"
                    # No state change, no block
                else:
                    result = "ERROR: Invalid GET format" # Handle error

            elif operation.startswith('DELETE '):
                # ... (DELETE logic remains the same) ...
                parts = operation.split(' ', 2)
                if len(parts) == 2:
                    key = parts[1]
                    with self.state_lock:
                        if key in self.state:
                            del self.state[key]
                            self.logger.info(f"DELETE {key}")
                            result = f"DELETE {key} = SUCCESS"
                            state_changed = True
                            block_model_type = "key-value-operation" # Corrected type assignment location
                        else:
                            result = f"DELETE {key} = KEY_NOT_FOUND"
                else:
                    result = "ERROR: Invalid DELETE format" # Handle error

            # --- Handle Unknown Operations ---
            else:
                self.logger.warning(f"Unknown operation format during execution: {operation}")
                result = f"UNKNOWN_OPERATION: {operation}"
                # No state change, no block for unknown operations

        except Exception as e:
            self.logger.error(f"Error executing operation {validation_key}: {e}", exc_info=True)
            result = f"EXECUTION_ERROR: {str(e)}"
            state_changed = False # Ensure no block is created if execution fails

        # --- Block Creation (only if state changed) ---
        if state_changed:
            # Assign block_id here, making it available to both primary and backup logs/checks
            block_id = validation_key 
            self.logger.debug(f"State changed for {block_id}. Preparing block data.")
            
            # (Logging state before block creation)
            try:
                current_state_str = json.dumps(self.state.copy(), sort_keys=True, indent=2)
                self.logger.debug(f"State BEFORE potential block creation for {block_id}:\n{current_state_str}")
            except Exception as e:
                self.logger.error(f"Error logging state before block creation: {e}")

            # Only the PRIMARY node creates, adds, and broadcasts the definitive block
            if self.pbft.is_primary_node():
                with self.blockchain_lock:
                    # Check if block already exists (should be less likely now, but good check)
                    if any(b.data.get('block_id') == block_id for b in self.blockchain.blocks):
                        self.logger.info(f"Primary: Block {block_id} already exists, skipping creation.")
                    else:
                        self.logger.info(f"Primary node creating and adding block for {block_id}")
                        # --- Prepare data for the block (moved inside primary check) ---
                        block_data = {
                            'operation': operation,
                            'sequence': sequence,
                            'view': view,
                            'block_id': block_id,
                            'request_id': request_id,
                            'result': result,
                        }
                        # Add snapshot/specific data based on the block_model_type determined earlier
                        if block_model_type == "cluster-validation-request":
                            block_data['cluster_client_ids'] = parsed_client_ids # Use the parsed list
                            block_data['state_snapshot'] = { 'cluster_validations': self.state.get('cluster_validations', {}).copy() }
                        elif block_model_type == "model-update":
                            block_data['state_snapshot'] = self.state.copy() # Snapshot happens here
                        elif block_model_type == "global-model-creation-or-update":
                            block_data['state_snapshot'] = {'global_model': self.state.get('global_model')}
                        elif block_model_type == "key-value-operation":
                             block_data['state_snapshot'] = self.state.copy() # Full state snapshot
                        # Add more cases if ModelManager introduces new block types

                        # ---> Log SNAPSHOT before creating block <---
                        try:
                            snapshot_str = json.dumps(block_data.get('state_snapshot', {}), sort_keys=True, indent=2)
                            self.logger.info(f"State SNAPSHOT being added to block {block_id}:\n{snapshot_str}")
                        except Exception as e:
                            self.logger.error(f"Error logging block snapshot: {e}")
                        # ---> END LOGGING <---

                        # --- Create the block ---
                        new_block = self.blockchain.create_block(
                            data=block_data,
                            model_type=block_model_type if block_model_type != "unknown" else "generic-state-change", # Use determined type
                            storage_reference=f"op-{block_id}",
                            calculated_hash=hashlib.sha256(json.dumps(block_data, sort_keys=True).encode()).hexdigest(),
                            participants=[str(self.node_id)]
                        )
                        self.blockchain.add_block(new_block)
                        self.logger.info(f"Primary added block #{new_block.index} for {block_id} (type: {block_model_type}), hash: {new_block.current_hash[:10]}...")

                        # Broadcast the new block
                        self.logger.info(f"Primary broadcasting block #{new_block.index} to all nodes")
                        block_sync = {
                            'type': MessageType.BLOCK_SYNC, # Use constant
                            'sender': self.node_id,
                            'block': new_block.to_dict(),
                            'sequence': sequence,
                            'view': view
                        }
                        self.message_handler.broadcast(block_sync)
            else:
                # Backup nodes just log that they executed the state change
                # They will add the block when they receive the BLOCK_SYNC from the primary
                # Now block_id is accessible here
                self.logger.info(f"Backup node executed state change for {block_id}. Waiting for block sync from primary.")
                # Ensure the state snapshot reflects the change made *before* block sync handling
                try:
                    current_state_str = json.dumps(self.state.copy(), sort_keys=True, indent=2)
                    self.logger.debug(f"Backup node state AFTER execution of {block_id}:\n{current_state_str}")
                except Exception as e:
                    self.logger.error(f"Error logging state after backup execution: {e}")


        # This logging might be confusing now for backup nodes, as their state
        # might change *again* when the block sync is applied. Consider adjusting.
        self.logger.debug(f"Finished execute_operation for {validation_key}. Current state values: {len(self.state.keys())} keys.")
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
                request_id = request.get('request_id', '') # Get request_id early for logging
                view_seq_key = f"v{view}-s{seq}" # Get view_seq_key early for logging
                
                # Check if we've already successfully executed this sequence in this view
                if hasattr(self, 'executed_view_seqs') and view_seq_key in self.executed_view_seqs:
                    self.logger.info(f"Sequence {view_seq_key} already successfully executed, skipping")
                    continue

                # Check if we've already executed this specific request ID (less reliable across views)
                # if request_id and request_id in self.executed_requests:
                #     self.logger.info(f"Request {request_id} already executed, skipping")
                #     continue

                # --- Duplicate Global Model Version Check (remains the same) ---
                operation = request.get('operation', '')
                if operation.startswith('UPDATE_GLOBAL_MODEL '):
                    parts = operation.split(' ', 3)
                    if len(parts) >= 4:
                        version = int(parts[3])
                        with self.state_lock:
                            try:
                                current_model_info = json.loads(self.state.get('global_model', '{}'))
                                current_version = current_model_info.get('version', 0)
                                if current_version >= version:
                                    self.logger.warning(f"Skipping duplicate global model update v{version} (current is v{current_version}) for {view_seq_key}")
                                    # Mark as executed to avoid reprocessing THIS specific case
                                    self.last_executed_seq = max(self.last_executed_seq, seq)
                                    if request_id: self.executed_requests.add(request_id)
                                    if not hasattr(self, 'executed_view_seqs'): self.executed_view_seqs = set()
                                    self.executed_view_seqs.add(view_seq_key)
                                    continue
                            except Exception as e:
                                self.logger.warning(f"Error checking duplicate global model version: {e}")
                                # Allow execution to proceed if check fails

                # --- Execute the operation ---
                execution_result = self.execute_operation(seq, {
                    'operation': operation,
                    'request_id': request_id,
                    'view': view
                })

                # --- Mark as executed ONLY IF execution wasn't deferred ---
                if execution_result != "PENDING_VALIDATION":
                    self.logger.debug(f"Marking sequence {view_seq_key} as successfully executed.")
                    self.last_executed_seq = max(self.last_executed_seq, seq) # Update last executed sequence number
                    if request_id:
                        self.executed_requests.add(request_id) # Track specific request ID

                    # Track successfully executed sequences per view
                    if not hasattr(self, 'executed_view_seqs'):
                        self.executed_view_seqs = set()
                    self.executed_view_seqs.add(view_seq_key)
                else:
                    # If pending validation, it will be re-queued later by _mark_operation_as_valid
                    self.logger.info(f"Execution for {view_seq_key} deferred due to pending validation. Not marking as executed yet.")

            except Exception as e:
                self.logger.error(f"Error processing execution queue: {e}", exc_info=True) # Add traceback
                time.sleep(0.1) # Avoid tight loop on error
    
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
            is_valid = True # Default to valid unless specific check fails

            # Handle model updates
            if operation.startswith('UPDATE_MODEL '):
                # Parse the operation string
                # Example: UPDATE_MODEL path hash loss accuracy v_version
                parts = operation.split(' ')
                if len(parts) >= 6: # UPDATE_MODEL + path + hash + loss + accuracy + version
                    try:
                        model_path = parts[1]
                        model_hash = parts[2]
                        reported_loss = float(parts[3])
                        reported_accuracy = float(parts[4])

                        # Call validate_model_update - it only returns a boolean
                        is_valid = self.model_manager.validate_model_update( # <--- Assign single return value
                            model_path=model_path,
                            model_hash=model_hash,
                            reported_loss=reported_loss,
                            reported_accuracy=reported_accuracy
                        )
                        self._track_model_validation(validation_key, model_path, request.get('client_id', 'unknown'), is_valid)
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"Failed to parse UPDATE_MODEL operation string '{operation}': {e}")
                        is_valid = False # Mark as invalid if parsing fails
                else:
                     self.logger.error(f"Invalid UPDATE_MODEL format: '{operation}'")
                     is_valid = False # Mark as invalid if format is wrong

            # Handle SET operations (example validation)
            elif operation.startswith('SET '):
                # ... (rest of SET validation logic remains the same) ...
                parts = operation.split(' ', 2)
                if len(parts) == 3:
                    _, key, value = parts
                    try:
                        # Example: Ensure value is greater than node ID
                        value_int = int(value)
                        is_valid = value_int > self.node_id
                        self.logger.debug(f"SET validation for {key}={value}: {'Valid' if is_valid else 'Invalid'} (value > {self.node_id}?)")
                    except ValueError:
                        is_valid = True # Non-numeric values are considered valid in this example
                else:
                    is_valid = False # Invalid format

            # CLUSTER_TRAIN_VALIDATE operations (assuming validation happens here)
            elif operation.startswith('CLUSTER_TRAIN_VALIDATE '):
                 # Parse the operation string for cluster validation details
                 match = re.match(
                     r"CLUSTER_TRAIN_VALIDATE cluster_id=(\d+) "
                     r"aggregated_model_path='([^']*)' "
                     r"aggregated_model_hash='([^']*)' "
                     r"client_ids='(\[.*\])' " # Capture JSON list
                     r"global_model_version=(\d+)",
                     operation
                 )
                 if match:
                     cluster_id = int(match.group(1))
                     agg_model_path = match.group(2)
                     agg_model_hash = match.group(3)
                     client_ids_json = match.group(4)
                     base_global_model_version = int(match.group(5))

                     try:
                         client_ids_list = json.loads(client_ids_json)
                     except json.JSONDecodeError:
                         self.logger.error(f"Failed to parse client_ids JSON in CLUSTER_TRAIN_VALIDATE: {client_ids_json}")
                         client_ids_list = []
                         is_valid = False # Mark as invalid if client_ids parsing fails

                     if is_valid: # Proceed only if client_ids parsed correctly
                         # Call a validation method in ModelManager specifically for cluster results
                         # This method might compare hash, check version, evaluate performance etc.
                         # Assume validate_cluster_aggregation also returns only a boolean
                         is_valid = self.model_manager.validate_cluster_aggregation( # <--- Assign single return value
                             cluster_id=cluster_id,
                             agg_model_path=agg_model_path,
                             expected_hash=agg_model_hash,
                             client_ids=client_ids_list,
                             base_global_model_version=base_global_model_version
                         )
                         # Potentially track cluster validation separately if needed
                         self.logger.info(f"Cluster {cluster_id} validation result: {'Valid' if is_valid else 'Invalid'}")

                 else:
                     self.logger.error(f"Invalid CLUSTER_TRAIN_VALIDATE format: '{operation}'")
                     is_valid = False # Mark as invalid if format is wrong


            # Record and broadcast the final validation result
            self._record_validation_result(validation_key, is_valid, sequence, view, request_id)
            return is_valid

        except Exception as e:
            self.logger.error(f"Error during validation for key {validation_key}: {e}", exc_info=True)
            # Ensure we still record an outcome, default to invalid on error
            self._record_validation_result(validation_key, False, sequence, view, request_id)
            return False # Return False on unexpected validation error

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
        """Mark an operation as valid and re-queue it if it was pending."""
        original_request = None
        # Retrieve the original request *before* potentially deleting it from pending
        if hasattr(self, 'pending_validations') and validation_key in self.pending_validations:
             original_request = self.pending_validations.get(validation_key) # Use .get for safety

        # Store the final validation consensus result definitively
        if not hasattr(self, 'validated_operations'):
            self.validated_operations = {}
        # Only log definitive marking if it wasn't already marked valid
        # and potentially re-queue
        needs_requeue = False
        if validation_key not in self.validated_operations or not self.validated_operations[validation_key]:
             self.validated_operations[validation_key] = True
             self.logger.info(f"Operation {validation_key} definitively marked as VALID by consensus.")
             needs_requeue = True # Mark for potential re-queue only on the first time it's marked valid
        else:
             # Already marked valid, likely by an earlier validation message processing
             self.logger.debug(f"Operation {validation_key} was already marked valid.")


        # If it was pending, try to remove it. Don't error if already removed.
        if hasattr(self, 'pending_validations') and validation_key in self.pending_validations:
            self.logger.info(f"Validation consensus for {validation_key} reached. Removing from pending if still present.")
            try:
                # Safely remove the key if it exists
                del self.pending_validations[validation_key]
                self.logger.debug(f"Removed {validation_key} from pending validations.")
            except KeyError:
                 # This is okay - it means it was likely already removed/processed.
                 self.logger.warning(f"{validation_key} was already removed from pending validations, likely executed or removed by another path.")

        # If the original request was retrieved AND we just marked it valid,
        # re-queue it for execution.
        if original_request and needs_requeue:
            self.logger.info(f"Re-queuing validated operation {validation_key} for execution.")
            # Ensure the request has necessary info (view might be missing if stored minimally)
            if 'view' not in original_request:
                 try:
                     # Extract view from validation_key (e.g., "v0-s2")
                     original_request['view'] = int(validation_key.split('-')[0][1:])
                 except:
                     self.logger.warning(f"Could not extract view from {validation_key}, using current view {self.pbft.view}")
                     original_request['view'] = self.pbft.view # Fallback
            self.execution_queue.put((sequence, original_request))
        elif needs_requeue:
             # Marked valid, but wasn't in pending_validations.
             # This might happen if validation finished *before* execute_operation
             # initially put it in pending. The execution queue should handle it.
             self.logger.debug(f"Operation {validation_key} marked valid, but not found in pending. Execution queue will handle.")
        # else: # No re-queue needed if not original_request or not needs_requeue
            # self.logger.debug(f"No re-queue needed for {validation_key}.")


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