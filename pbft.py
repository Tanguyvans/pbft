import logging
import threading
from typing import Dict
import time

class PBFT:
    def __init__(self, node_id: int, total_nodes: int, node):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.node = node  # Reference to the parent node
        self.logger = logging.getLogger(f"PBFT-{self.node_id}")
        
        # PBFT state
        self.view = 0  # Current view number
        self.sequence_number = 0  # Sequence number for requests
        self.sequence_lock = threading.Lock()
        self._is_primary = (self.node_id == self.view % self.total_nodes)  # Use an underscore for the internal attribute
        self.in_view_change = False
        
        # Message logs
        self.request_log = {}  # Store client requests
        self.pre_prepare_log = {}  # Store pre-prepare messages
        self.prepare_log = {}  # Store prepare messages
        self.commit_log = {}  # Store commit messages
        self.executed_log = {}  # Store executed requests
        
        # View change related
        self.view_change_log = {}  # Store view-change messages
        self.new_view_log = {}  # Store new-view messages
        self.view_change_timeout = 10.0  # Increased from 5.0 to 10.0 seconds
        self.view_change_timer = None
        self.changing_view = False
        self.last_activity_time = time.time()
        
        # Calculate the maximum number of faulty nodes
        self.f = (self.total_nodes - 1) // 3
        
        # Heartbeat related
        self.heartbeat_interval = 2.0  # Send heartbeat every 2 seconds
        self.heartbeat_timer = None
        
        # Start the view change timer
        self.reset_view_change_timer()
        
        # Start the heartbeat timer if this is the primary
        self.start_heartbeat_timer()
        
        self.logger.info(f"PBFT initialized with {total_nodes} nodes, f={self.f}")
    
    def is_primary_node(self) -> bool:
        """Check if this node is the primary for the current view"""
        primary = self._is_primary
        self.logger.debug(f"Checking if node {self.node_id} is primary: {primary} (view: {self.view}, total: {self.total_nodes})")
        return primary
    
    def set_primary_status(self, status):
        """Set the primary status"""
        self._is_primary = status
        self.logger.info(f"Primary status updated to: {self._is_primary}")
    
    def store_request(self, request_id: str, request: Dict):
        """Store a client request"""
        self.request_log[request_id] = request
    
    def start_consensus(self, request_id: str):
        """Start the consensus process for a request (primary only)"""
        if not self.is_primary_node():
            self.logger.warning("Non-primary node tried to start consensus")
            return
        
        if request_id not in self.request_log:
            self.logger.warning(f"Request {request_id} not found in request log")
            return
        
        request = self.request_log[request_id]
        
        with self.sequence_lock:
            self.sequence_number += 1
            seq_num = self.sequence_number
        
        # Create pre-prepare message
        pre_prepare = {
            'type': 'pre-prepare',
            'view': self.view,
            'sequence': seq_num,
            'digest': request['digest'],
            'request_id': request_id,
            'sender': self.node_id
        }
        
        # Store pre-prepare message
        key = f"{self.view}:{seq_num}"
        self.pre_prepare_log[key] = pre_prepare
        
        # Broadcast pre-prepare message
        self.node.broadcast(pre_prepare)
        self.logger.info(f"Sent pre-prepare for request {request_id}, seq {seq_num}")
        
        # Primary also processes its own pre-prepare message
        self.process_pre_prepare(pre_prepare)
    
    def process_message(self, message: Dict):
        """Process PBFT protocol messages"""
        msg_type = message.get('type')
        
        # Record activity for any message type
        self.record_activity()
        
        if msg_type == 'pre-prepare':
            self.process_pre_prepare(message)
        elif msg_type == 'prepare':
            self.process_prepare(message)
        elif msg_type == 'commit':
            self.process_commit(message)
        elif msg_type == 'view-change':
            self.process_view_change(message)
        elif msg_type == 'new-view':
            self.process_new_view(message)
        elif msg_type == 'heartbeat':
            # Just record activity, no further processing needed
            pass
    
    def process_pre_prepare(self, message: Dict):
        """Process a pre-prepare message"""
        view = message.get('view')
        seq = message.get('sequence')
        global_seq = message.get('global_sequence', seq)  # Use global sequence if available
        request_id = message.get('request_id')
        digest = message.get('digest')
        sender = message.get('sender')
        
        self.logger.info(f"Processing pre-prepare from {sender} for seq {seq}")
        
        # Verify the message
        if view != self.view:
            self.logger.warning(f"Ignoring pre-prepare for different view: {view}")
            return
        
        # Store the pre-prepare message
        key = f"{view}:{seq}"
        if key not in self.pre_prepare_log:
            self.pre_prepare_log[key] = {}
        self.pre_prepare_log[key] = message
        
        # Create and send prepare message
        prepare = {
            'type': 'prepare',
            'view': view,
            'sequence': seq,
            'global_sequence': global_seq,
            'digest': digest,
            'sender': self.node_id
        }
        
        # Store and broadcast prepare message
        if key not in self.prepare_log:
            self.prepare_log[key] = {}
        self.prepare_log[key][self.node_id] = prepare
        
        self.node.broadcast(prepare)
        self.logger.info(f"Sent prepare for seq {seq}")
        
        # Process the prepare message locally
        self.process_prepare(prepare)
    
    def process_prepare(self, message: Dict):
        """Process prepare message"""
        view = message.get('view')
        sequence = message.get('sequence')
        digest = message.get('digest')
        sender = message.get('sender')
        request_id = message.get('request_id')
        
        self.logger.info(f"Processing prepare from {sender} for seq {sequence}")
        
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
        if key in self.pre_prepare_log and len(self.prepare_log[key]) >= 2*self.f + 1:
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
                
                self.node.broadcast(commit)
                self.logger.info(f"Sent commit for seq {sequence}")
    
    def process_commit(self, message: Dict):
        """Process a commit message"""
        view = message.get('view')
        seq = message.get('sequence')
        digest = message.get('digest')
        sender = message.get('sender')
        
        self.logger.info(f"Processing commit from {sender} for seq {seq}")
        
        # Verify the message
        if view != self.view:
            self.logger.warning(f"Ignoring commit for different view: {view}")
            return
        
        # Store the commit message
        key = f"{view}:{seq}"
        if key not in self.commit_log:
            self.commit_log[key] = {}
        self.commit_log[key][sender] = message
        
        # Check if we have enough commits (2f+1 including our own)
        if len(self.commit_log[key]) >= 2 * self.f + 1:
            # Check if we've already executed this request
            if key in self.executed_log:
                return
            
            # Mark as executed
            self.executed_log[key] = True
            
            # Get the request from the pre-prepare message
            pre_prepare = self.pre_prepare_log.get(key)
            if not pre_prepare:
                self.logger.warning(f"No pre-prepare found for {key}")
                return
            
            request_id = pre_prepare.get('request_id')
            request = self.request_log.get(request_id)
            if not request:
                self.logger.warning(f"No request found for {request_id}")
                return
            
            # Notify the node that consensus has been reached
            self.logger.info(f"Consensus reached for request {request_id}, seq {seq}")
            self.node.on_consensus_reached(seq, request)

    def handle_request(self, request: Dict):
        """Handle a client request"""
        request_id = request.get('request_id')
        
        # Store the request
        self.request_log[request_id] = request
        
        # Track when we received this request (for censorship detection)
        if not hasattr(self, 'request_timestamps'):
            self.request_timestamps = {}
        self.request_timestamps[request_id] = time.time()
        
        # If this node is the primary, start the consensus process
        if self.is_primary_node():
            self.logger.info(f"Primary node initiating consensus for request {request_id}")
            
            # Get the next sequence number
            with self.sequence_lock:
                seq_num = self.sequence_number
                self.sequence_number += 1
            
            # Create pre-prepare message
            pre_prepare = {
                'type': 'pre-prepare',
                'view': self.view,
                'sequence': seq_num,
                'digest': request['digest'],
                'request_id': request_id,
                'sender': self.node_id
            }
            
            # Store pre-prepare message
            key = f"{self.view}:{seq_num}"
            self.pre_prepare_log[key] = pre_prepare
            
            # Broadcast pre-prepare message
            self.node.broadcast(pre_prepare)
            self.logger.info(f"Sent pre-prepare for request {request_id}, seq {seq_num}")
            
            # Primary also processes its own pre-prepare message
            self.process_pre_prepare(pre_prepare)

    def reset_view_change_timer(self):
        """Reset the view change timer"""
        if self.view_change_timer:
            self.view_change_timer.cancel()
        
        # Only non-primary nodes need to monitor for primary failure
        if not self.is_primary_node():
            self.view_change_timer = threading.Timer(self.view_change_timeout, self.initiate_view_change)
            self.view_change_timer.daemon = True
            self.view_change_timer.start()
        
        self.last_activity_time = time.time()

    def record_activity(self):
        """Record activity to prevent unnecessary view changes"""
        self.last_activity_time = time.time()
        # If we're not in the middle of a view change, reset the timer
        if not self.changing_view:
            self.reset_view_change_timer()

    def initiate_view_change(self):
        """Initiate a view change when the primary is suspected to be faulty"""
        # Check if there has been recent activity
        if time.time() - self.last_activity_time < self.view_change_timeout:
            self.reset_view_change_timer()
            return
        
        # Don't initiate if we're already changing view
        if self.changing_view:
            return
        
        self.changing_view = True
        new_view = self.view + 1
        self.logger.warning(f"Initiating view change to view {new_view}")
        
        # Create view-change message
        view_change_msg = {
            'type': 'view-change',
            'new_view': new_view,
            'last_seq': self.sequence_number,
            'sender': self.node_id,
            # Include information about prepared requests
            'prepared': self.get_prepared_requests()
        }
        
        # Store and broadcast the view-change message
        if new_view not in self.view_change_log:
            self.view_change_log[new_view] = {}
        self.view_change_log[new_view][self.node_id] = view_change_msg
        
        self.node.broadcast(view_change_msg)
        self.logger.info(f"Sent view-change message for view {new_view}")

    def get_prepared_requests(self):
        """Get information about prepared requests for view change"""
        prepared = []
        for key, prepares in self.prepare_log.items():
            view, seq = map(int, key.split(':'))
            # Check if this request has enough prepares (2f+1)
            if len(prepares) >= 2*self.f + 1:
                # Get the digest from any prepare message
                any_prepare = next(iter(prepares.values()))
                digest = any_prepare.get('digest')
                request_id = any_prepare.get('request_id')
                prepared.append({
                    'view': view,
                    'sequence': seq,
                    'digest': digest,
                    'request_id': request_id
                })
        return prepared

    def process_view_change(self, message):
        """Process a view-change message"""
        sender = message.get('sender')
        new_view = message.get('new_view')
        
        self.logger.info(f"Processing view-change from {sender} for view {new_view}")
        
        # Verify the message
        if new_view <= self.view:
            self.logger.warning(f"Ignoring view-change for old or current view: {new_view}")
            return
        
        # Store the view-change message
        if new_view not in self.view_change_log:
            self.view_change_log[new_view] = {}
        self.view_change_log[new_view][sender] = message
        
        # Check if we have enough view-change messages (2f+1)
        if len(self.view_change_log[new_view]) >= 2 * self.f + 1:
            self.logger.info(f"Received enough view-change messages for view {new_view}")
            
            # If this node is the new primary, send new-view message
            new_primary = new_view % self.total_nodes
            if self.node_id == new_primary:
                self.logger.info(f"I am the new primary for view {new_view}")
                
                # Create new-view message
                new_view_msg = {
                    'type': 'new-view',
                    'view': new_view,
                    'sender': self.node_id,
                    'view_changes': list(self.view_change_log[new_view].keys())
                }
                
                # Broadcast new-view message
                self.node.broadcast(new_view_msg)
                
                # Install the new view
                self.install_new_view(new_view)
            else:
                self.logger.info(f"Node {new_primary} should be the new primary for view {new_view}")

    def send_new_view(self, new_view: int):
        """Send new-view message (only called by the new primary)"""
        # Collect all view-change messages for this view
        view_changes = self.view_change_log.get(new_view, {})
        
        # Create new-view message
        new_view_msg = {
            'type': 'new-view',
            'view': new_view,
            'view_changes': list(view_changes.values()),  # Include all view-change messages
            'sender': self.node_id
        }
        
        # Store and broadcast the new-view message
        self.new_view_log[new_view] = new_view_msg
        self.node.broadcast(new_view_msg)
        self.logger.info(f"Sent new-view message for view {new_view}")
        
        # Update to the new view
        self.install_new_view(new_view)

    def process_new_view(self, message: Dict):
        """Process new-view message"""
        sender = message.get('sender')
        new_view = message.get('view')
        view_changes = message.get('view_changes', [])
        
        self.logger.info(f"Processing new-view from {sender} for view {new_view}")
        
        # Verify the message
        if sender != (new_view % self.total_nodes):
            self.logger.warning(f"New-view message from incorrect sender: {sender}")
            return
        
        # Verify that there are enough view-change messages
        if len(view_changes) < 2*self.f + 1:
            self.logger.warning(f"Not enough view-change messages: {len(view_changes)}")
            return
        
        # Store the new-view message
        self.new_view_log[new_view] = message
        
        # Update to the new view
        self.install_new_view(new_view)

    def install_new_view(self, new_view):
        """Install a new view"""
        self.logger.info(f"Installing new view: {new_view}")
        
        # Update the view number
        self.view = new_view
        
        # Calculate the new primary
        new_primary_id = new_view % self.total_nodes
        
        # Update primary status
        self._is_primary = (self.node_id == new_primary_id)
        
        self.logger.info(f"Primary status after view change: {self._is_primary}")
        
        # Reset sequence number for the new view
        # Don't reset sequence number, just track it per view
        # self.sequence_number = 0
        
        # Notify the node about the view change
        self.node.on_view_change(new_view)
        
        # Clear view change logs for this view
        if new_view in self.view_change_log:
            del self.view_change_log[new_view]

    def reissue_prepares(self):
        """Re-issue pre-prepares for prepared requests from previous view"""
        # This would re-issue pre-prepares for any requests that were prepared
        # but not committed in the previous view
        # Implementation depends on how you track prepared but uncommitted requests
        pass

    def start_heartbeat_timer(self):
        """Start or restart the heartbeat timer for the primary"""
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
        
        # Only primary nodes need to send heartbeats
        if self.is_primary_node():
            self.heartbeat_timer = threading.Timer(self.heartbeat_interval, self.send_heartbeat)
            self.heartbeat_timer.daemon = True
            self.heartbeat_timer.start()

    def send_heartbeat(self):
        """Send a heartbeat message to all nodes"""
        if not self.is_primary_node():
            return
        
        heartbeat_msg = {
            'type': 'heartbeat',
            'view': self.view,
            'sender': self.node_id,
            'timestamp': time.time()
        }
        
        self.node.broadcast(heartbeat_msg)
        self.logger.debug(f"Primary sent heartbeat for view {self.view}")
        
        # Schedule the next heartbeat
        self.start_heartbeat_timer()

    def cleanup(self):
        """Clean up timers when node is shutting down"""
        if self.view_change_timer:
            self.view_change_timer.cancel()
        
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()

    def update_total_nodes(self, total_nodes: int):
        """Update the total number of nodes in the network"""
        self.logger.info(f"Updating total nodes from {self.total_nodes} to {total_nodes}")
        self.total_nodes = total_nodes
        
        # Recalculate f (max number of faulty nodes)
        self.f = (self.total_nodes - 1) // 3
        self.logger.info(f"Updated f to {self.f}")
        
        # Check if we need to change view due to new node count
        primary_id = self.view % self.total_nodes
        self._is_primary = (self.node_id == primary_id)
        self.logger.info(f"Primary status after node update: {self._is_primary}") 