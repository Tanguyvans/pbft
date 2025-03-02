from block import Block
import json
import hashlib
import time

class Blockchain:
    def __init__(self):
        self.blocks = []
        self.pending_operations = []
        self.add_genesis_block()
    
    def add_genesis_block(self):
        """Create the first block in the chain"""
        genesis_block = Block(
            index=0,
            data={"message": "Genesis Block"},
            model_type="genesis",
            storage_reference="",
            calculated_hash="",
            participants=["system"],
            previous_hash=""
        )
        self.blocks.append(genesis_block)
    
    def add_block(self, block):
        """Add a new block to the chain"""
        self.blocks.append(block)
    
    def create_block(self, data, model_type="", storage_reference="", calculated_hash="", participants=None):
        """Create a new block with the given data"""
        previous_block = self.blocks[-1]
        new_block = Block(
            index=len(self.blocks),
            data=data,
            model_type=model_type,
            storage_reference=storage_reference,
            calculated_hash=calculated_hash,
            participants=participants if participants else [],
            previous_hash=previous_block.current_hash
        )
        # Mine the block (optional)
        new_block.mine_block(difficulty=2)
        return new_block
    
    def add_operation(self, operation):
        """Add an operation to the pending operations list"""
        self.pending_operations.append(operation)
    
    def create_block_from_pending(self, model_type="", storage_reference="", calculated_hash="", participants=None):
        """Create a new block from pending operations"""
        if not self.pending_operations:
            return None
        
        block = self.create_block(
            data=self.pending_operations.copy(),
            model_type=model_type,
            storage_reference=storage_reference,
            calculated_hash=calculated_hash,
            participants=participants
        )
        self.pending_operations = []
        return block
    
    def is_valid_chain(self):
        """Verify the integrity of the blockchain"""
        for i in range(1, len(self.blocks)):
            current_block = self.blocks[i]
            previous_block = self.blocks[i-1]
            
            # Check if the current block points to the previous block's hash
            if current_block.previous_hash != previous_block.current_hash:
                return False
            
            # Verify the hash of the current block
            if current_block.current_hash != current_block.current_hash:
                return False
        
        return True
    
    def is_valid_block(self, block, hash_to_verify=None):
        """Verify if a block is valid"""
        # If no specific hash is provided, verify against the block's own hash
        if hash_to_verify is None:
            hash_to_verify = block.current_hash
        
        # Check if the block's calculated hash matches the provided hash
        if block.current_hash != hash_to_verify:
            return False
        
        # If this is not the genesis block, check if it points to the previous block
        if block.index > 0 and len(self.blocks) >= block.index:
            previous_block = self.blocks[block.index - 1]
            if block.previous_hash != previous_block.current_hash:
                return False
        
        return True
    
    @property
    def len_chain(self):
        """Return the length of the blockchain"""
        return len(self.blocks)
    
    def print_blockchain(self):
        """Print the contents of the blockchain"""
        for block in self.blocks:
            print(block)
    
    def save_chain_to_file(self, filename):
        """Save the blockchain to a file"""
        with open(filename, "w") as f:
            for block in self.blocks:
                f.write(str(block) + "\n")
    
    def to_json(self):
        """Convert the blockchain to JSON"""
        chain_dict = {
            "length": len(self.blocks),
            "blocks": [block.to_dict() for block in self.blocks]
        }
        return json.dumps(chain_dict, indent=4) 