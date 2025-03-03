import logging
import threading
from typing import Dict

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
        
        # Message logs
        self.request_log = {}  # Store client requests
        self.pre_prepare_log = {}  # Store pre-prepare messages
        self.prepare_log = {}  # Store prepare messages
        self.commit_log = {}  # Store commit messages
        
        # Calculate the maximum number of faulty nodes
        self.f = (self.total_nodes - 1) // 3
        
        self.logger.info(f"PBFT initialized with {total_nodes} nodes, f={self.f}")
    
    @property
    def is_primary(self) -> bool:
        """Check if this node is the primary for the current view"""
        primary = self.node_id == (self.view % self.total_nodes)
        self.logger.debug(f"Checking if node {self.node_id} is primary: {primary} (view: {self.view}, total: {self.total_nodes})")
        return primary
    
    def store_request(self, request_id: str, request: Dict):
        """Store a client request"""
        self.request_log[request_id] = request
    
    def start_consensus(self, request_id: str):
        """Start the consensus process for a request (primary only)"""
        if not self.is_primary:
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
        
        if msg_type == 'pre-prepare':
            self.process_pre_prepare(message)
        elif msg_type == 'prepare':
            self.process_prepare(message)
        elif msg_type == 'commit':
            self.process_commit(message)
    
    def process_pre_prepare(self, message: Dict):
        """Process pre-prepare message"""
        view = message.get('view')
        sequence = message.get('sequence')
        digest = message.get('digest')
        request_id = message.get('request_id')
        sender = message.get('sender')
        
        self.logger.info(f"Processing pre-prepare from {sender} for seq {sequence}")
        
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
        
        self.node.broadcast(prepare)
        self.logger.info(f"Sent prepare for seq {sequence}")
    
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
        """Process commit message"""
        sender = message.get('sender')
        view = message.get('view')
        sequence = message.get('sequence')
        digest = message.get('digest')
        request_id = message.get('request_id')
        
        self.logger.info(f"Processing commit from {sender} for seq {sequence}")
        
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
        if (key in self.pre_prepare_log and 
            len(self.commit_log[key]) >= 2*self.f + 1):
            
            # Find the request
            pre_prepare = self.pre_prepare_log[key]
            request_id = pre_prepare.get('request_id')
            
            if request_id in self.request_log and request_id not in self.node.executed_requests:
                request = self.request_log[request_id]
                
                # Notify the node that consensus has been reached
                self.node.on_consensus_reached(sequence, request_id, request)

    def handle_request(self, request: Dict):
        """Handle a client request"""
        request_id = request.get('request_id')
        
        # Store the request
        self.request_log[request_id] = request
        
        # If this node is the primary, start the consensus process
        if self.is_primary:
            self.logger.info(f"Primary node initiating consensus for request {request_id}")
            
            # Get the next sequence number
            with self.sequence_lock:
                # Start from sequence 1 (not 0)
                if self.sequence_number == 0:
                    self.sequence_number = 1
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