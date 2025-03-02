import hashlib
import json
import time

class Block:
    def __init__(self, index, data, model_type="", storage_reference="", calculated_hash="", participants=None, previous_hash=""):
        self.index = index
        self.timestamp = time.time()
        self.data = data  # This can store any data (like operations in our case)
        self.model_type = model_type
        self.storage_reference = storage_reference
        self.calculated_hash = calculated_hash
        self.participants = participants if participants else []
        self.previous_hash = previous_hash
        self.nonce = 0
    
    @property
    def current_hash(self):
        """
        Calculate and return the current hash on each access
        """
        block_string = (f"{self.index}{self.timestamp}{json.dumps(self.data, sort_keys=True)}"
                        f"{self.model_type}{self.storage_reference}{self.calculated_hash}"
                        f"{self.participants}{self.previous_hash}{self.nonce}")
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty=2):
        """Simple proof of work algorithm"""
        target = '0' * difficulty
        while self.current_hash[:difficulty] != target:
            self.nonce += 1
        return self.current_hash
    
    def __str__(self):
        return f"================\n" \
               f"Index:\t\t {self.index}\n" \
               f"Timestamp:\t {self.timestamp}\n" \
               f"Data:\t\t {self.data}\n" \
               f"Model Type:\t {self.model_type}\n" \
               f"Storage Ref:\t {self.storage_reference}\n" \
               f"Calc Hash:\t {self.calculated_hash}\n" \
               f"Participants:\t {self.participants}\n" \
               f"Previous Hash:\t {self.previous_hash}\n" \
               f"Current Hash:\t {self.current_hash}\n" \
               f"Nonce:\t\t {self.nonce}\n"
    
    def to_dict(self):
        """Convert the block to a dictionary for JSON serialization"""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "model_type": self.model_type,
            "storage_reference": self.storage_reference,
            "calculated_hash": self.calculated_hash,
            "participants": self.participants,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "current_hash": self.current_hash
        } 