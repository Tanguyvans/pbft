import socket
import json
import time
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PBFTClient:
    def __init__(self, client_id: str, nodes_config: list):
        self.client_id = client_id
        self.nodes = nodes_config
        self.logger = logging.getLogger(f"Client-{self.client_id}")
        self.request_count = 0
    
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
        
        # Send to all nodes (in a real system, you might just send to the primary)
        success_count = 0
        for node in self.nodes:
            try:
                self.logger.info(f"Sending request to node {node['id']}: {operation}")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((node['host'], node['port']))
                s.sendall(json.dumps(request).encode('utf-8'))
                s.close()
                success_count += 1
            except Exception as e:
                self.logger.error(f"Error sending to node {node['id']}: {e}")
        
        return success_count > 0 