import threading
import logging
import json
from typing import Dict, List
import socket
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MessageHandler:
    def __init__(self, node, server_socket):
        self.node = node
        self.server_socket = server_socket
        self.running = True
        self.logger = logging.getLogger(f"Node-{self.node.node_id}")
    
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
                self.stop()
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
                    self.node.process_message(message)
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
            # If connection succeeds, remove node from failed set if it was there
            if hasattr(self.node, 'failed_nodes') and target_node['id'] in self.node.failed_nodes:
                self.logger.info(f"Node {target_node['id']} is back online.")
                self.node.failed_nodes.remove(target_node['id'])
        except ConnectionRefusedError:
            # Initialize failed_nodes set if it doesn't exist
            if not hasattr(self.node, 'failed_nodes'):
                self.node.failed_nodes = set()
            
            # Add the node ID to the set *before* logging
            node_id = target_node['id']
            is_new_failure = node_id not in self.node.failed_nodes
            
            if is_new_failure:
                self.node.failed_nodes.add(node_id)
                self.logger.warning(f"Node {node_id} appears to be down. Added to failed set.")

                # --- Trigger View Change if Primary Fails ---
                # Check if the failed node is the current primary and we are not already in view change
                current_primary_id = self.node.pbft.view % len(self.node.nodes)
                if node_id == current_primary_id and not self.node.pbft.in_view_change:
                    self.logger.warning(f"Detected primary node {node_id} failed. Initiating view change.")
                    # Use the PBFT method to start the view change process
                    self.node.pbft.initiate_view_change(reason="primary_unreachable")
            # else: # Optional: Log subsequent failures differently or less verbosely
            #     self.logger.debug(f"Still unable to connect to failed node {node_id}.")

        except Exception as e:
            self.logger.error(f"Error sending message to {target_node['id']}: {e}")
    
    def broadcast(self, message: Dict, exclude_self=False):
        """Broadcast a message to all nodes"""
        for node in self.node.nodes:
            if exclude_self and node['id'] == self.node.node_id:
                continue
            self.send_message(node, message)
    
    def stop(self):
        """Stop the node"""
        self.running = False
        self.node.pbft.cleanup()  # Clean up PBFT timers
        self.server_socket.close()
        self.logger.info(f"Node {self.node.node_id} stopped")
       