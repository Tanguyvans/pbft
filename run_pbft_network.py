import time
import threading
import random
import logging
from pbft_node import PBFTNode
from pbft_client import PBFTClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PBFT-Network")

def send_concurrent_requests(clients, operations):
    """Send requests from multiple clients concurrently"""
    threads = []
    for i, (client, operation) in enumerate(zip(clients, operations)):
        thread = threading.Thread(target=client.send_request, args=(operation,))
        threads.append(thread)
        logger.info(f"Starting thread {i} for operation: {operation}")
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    logger.info("All concurrent requests sent")

def main():
    # Define node configurations
    num_nodes = 4  # 4 nodes can tolerate 1 Byzantine fault
    base_port = 10000
    
    nodes_config = []
    for i in range(num_nodes):
        nodes_config.append({
            'id': i,
            'host': 'localhost',
            'port': base_port + i
        })
    
    # Start nodes
    nodes = []
    for i in range(num_nodes):
        node = PBFTNode(
            node_id=i,
            host='localhost',
            port=base_port + i,
            nodes_config=nodes_config
        )
        nodes.append(node)
        logger.info(f"Started node {i} on port {base_port + i}")
    
    # Give nodes time to start
    time.sleep(2)
    
    # Create clients
    clients = []
    for i in range(num_nodes):
        client = PBFTClient(client_id=f"client{i}", nodes_config=nodes_config)
        clients.append(client)
    
    # Initial operations to demonstrate functionality
    operations = [
        "SET key1 value1",
        "SET key2 value2",
    ]
    
    for op in operations:
        clients[0].send_request(op)
        time.sleep(2)  # Wait between requests
    
    # Interactive mode
    try:
        while True:
            print("\nPBFT Blockchain Network Menu:")
            print("1. Add a new element to the chain")
            print("2. Check if all nodes have the same state")
            print("3. View state of a specific node")
            print("4. Send concurrent requests from multiple clients")
            print("5. View blockchain of a specific node")
            print("6. Compare blockchains across nodes")
            print("7. Save blockchain to file")
            print("8. Exit")
            
            choice = input("Enter your choice (1-8): ")
            
            if choice == '1':
                operation_type = input("Enter operation type (SET/GET/DELETE): ").upper()
                if operation_type in ["SET", "GET", "DELETE"]:
                    if operation_type == "SET":
                        key = input("Enter key: ")
                        value = input("Enter value: ")
                        operation = f"SET {key} {value}"
                    else:
                        key = input("Enter key: ")
                        operation = f"{operation_type} {key}"
                    
                    clients[0].send_request(operation)
                    print(f"Request sent: {operation}")
                    time.sleep(2)  # Wait for propagation
                else:
                    print("Invalid operation type. Please use SET, GET, or DELETE.")
            
            elif choice == '2':
                # Check if all nodes have the same state
                states = []
                for i, node in enumerate(nodes):
                    states.append((i, node.get_state()))
                
                print("\nState comparison across nodes:")
                reference_state = states[0][1]
                all_same = True
                
                for node_id, state in states:
                    is_same = (state == reference_state)
                    if not is_same:
                        all_same = False
                    print(f"Node {node_id}: {'SAME' if is_same else 'DIFFERENT'}")
                
                if all_same:
                    print("\n✅ All nodes have the same state - Consensus achieved!")
                else:
                    print("\n❌ Nodes have different states - Consensus failed!")
                
                print("\nDetailed state for each node:")
                for node_id, state in states:
                    print(f"Node {node_id}: {state}")
            
            elif choice == '3':
                node_id = input("Enter node ID (0-3): ")
                try:
                    node_id = int(node_id)
                    if 0 <= node_id < len(nodes):
                        print(f"\nState of Node {node_id}:")
                        print(nodes[node_id].get_state())
                        print(f"Last executed sequence: {nodes[node_id].last_executed_seq}")
                        print(f"Is primary: {nodes[node_id].is_primary}")
                    else:
                        print(f"Invalid node ID. Please enter a number between 0 and {len(nodes)-1}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '4':
                # Send concurrent requests from multiple clients
                num_requests = input("Enter number of concurrent requests (2-4): ")
                try:
                    num_requests = int(num_requests)
                    if 2 <= num_requests <= 4:
                        concurrent_operations = []
                        for i in range(num_requests):
                            print(f"\nRequest {i+1}:")
                            operation_type = input("Enter operation type (SET/GET/DELETE): ").upper()
                            if operation_type in ["SET", "GET", "DELETE"]:
                                if operation_type == "SET":
                                    key = input("Enter key: ")
                                    value = input("Enter value: ")
                                    operation = f"SET {key} {value}"
                                else:
                                    key = input("Enter key: ")
                                    operation = f"{operation_type} {key}"
                                concurrent_operations.append(operation)
                            else:
                                print("Invalid operation type. Skipping this request.")
                        
                        if concurrent_operations:
                            print("\nSending concurrent requests...")
                            send_concurrent_requests(clients[:len(concurrent_operations)], concurrent_operations)
                            print("Concurrent requests sent. Waiting for processing...")
                            time.sleep(3)  # Wait for processing
                    else:
                        print("Please enter a number between 2 and 4.")
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '5':
                # View blockchain of a specific node
                node_id = input("Enter node ID (0-3): ")
                try:
                    node_id = int(node_id)
                    if 0 <= node_id < len(nodes):
                        blockchain = nodes[node_id].get_blockchain()
                        print(f"\nBlockchain of Node {node_id}:")
                        print(f"Chain length: {blockchain.len_chain}")
                        
                        view_details = input("View detailed block information? (y/n): ").lower()
                        if view_details == 'y':
                            blockchain.print_blockchain()
                        else:
                            # Print summary
                            for i, block in enumerate(blockchain.blocks):
                                operations_count = len(block.data) if isinstance(block.data, list) else 1
                                print(f"Block #{i}: {operations_count} operations, hash: {block.current_hash[:10]}...")
                    else:
                        print(f"Invalid node ID. Please enter a number between 0 and {len(nodes)-1}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '6':
                # Compare blockchains across nodes
                blockchains = []
                for i, node in enumerate(nodes):
                    blockchains.append((i, node.get_blockchain()))
                
                print("\nBlockchain comparison across nodes:")
                reference_chain = blockchains[0][1]
                all_same = True
                
                for node_id, blockchain in blockchains:
                    # Compare chain length
                    length_same = (blockchain.len_chain == reference_chain.len_chain)
                    
                    # Compare last block hash
                    last_block_same = False
                    if length_same and blockchain.len_chain > 0:
                        last_block_same = (blockchain.blocks[-1].current_hash == 
                                          reference_chain.blocks[-1].current_hash)
                    
                    is_same = length_same and last_block_same
                    if not is_same:
                        all_same = False
                    
                    print(f"Node {node_id}: {'SAME' if is_same else 'DIFFERENT'} " +
                          f"(Length: {blockchain.len_chain}, " +
                          f"Last block hash: {blockchain.blocks[-1].current_hash[:10]}...)")
                
                if all_same:
                    print("\n✅ All nodes have the same blockchain - Consensus achieved!")
                else:
                    print("\n❌ Nodes have different blockchains - Consensus failed!")
            
            elif choice == '7':
                # Save blockchain to file
                node_id = input("Enter node ID to save blockchain from (0-3): ")
                try:
                    node_id = int(node_id)
                    if 0 <= node_id < len(nodes):
                        filename = input("Enter filename to save to: ")
                        if not filename:
                            filename = f"blockchain_node_{node_id}.txt"
                        
                        blockchain = nodes[node_id].get_blockchain()
                        blockchain.save_chain_to_file(filename)
                        print(f"Blockchain saved to {filename}")
                        
                        # Also save as JSON
                        json_filename = f"{filename.split('.')[0]}.json"
                        with open(json_filename, "w") as f:
                            f.write(blockchain.to_json())
                        print(f"Blockchain also saved as JSON to {json_filename}")
                    else:
                        print(f"Invalid node ID. Please enter a number between 0 and {len(nodes)-1}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '8':
                break
            
            else:
                print("Invalid choice. Please enter a number between 1 and 8.")
    
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down PBFT network...")
        for node in nodes:
            node.stop()

if __name__ == "__main__":
    main() 