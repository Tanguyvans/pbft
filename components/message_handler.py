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
        except ConnectionRefusedError:
            # More concise error for expected failures
            if not hasattr(self, 'failed_nodes'):
                self.node.failed_nodes = set()
            
            # Only log the first time we detect a node is down
            if target_node['id'] not in self.node.failed_nodes:
                self.logger.warning(f"Node {target_node['id']} appears to be down")
                self.node.failed_nodes.add(target_node['id'])
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
        self.pbft.cleanup()  # Clean up PBFT timers
        self.server_socket.close()
        self.logger.info(f"Node {self.node_id} stopped")
       