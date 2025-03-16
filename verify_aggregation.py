import numpy as np
import os
import hashlib
import time
import json
import logging
import glob
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AggregationVerifier")

def load_model_weights(model_path):
    """Load model weights from NPZ file with proper error handling"""
    try:
        weights = {}
        with np.load(model_path, allow_pickle=True) as npz_data:
            for key in npz_data.files:
                try:
                    # Make a copy to avoid issues after file is closed
                    weights[key] = npz_data[key].copy()
                except Exception as e:
                    logger.warning(f"Error loading key {key} from model: {e}")
        return weights
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None

def find_global_model(version=None, models_dir="models/npz"):
    """Find a global model, with or without version number"""
    if version is not None:
        # Try to find versioned global model first
        pattern = os.path.join(models_dir, f"global_model_v{version}_*.npz")
        files = glob.glob(pattern)
        
        if files:
            # Sort by timestamp (assuming the timestamp is the last part before .npz)
            latest_file = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)[0]
            logger.info(f"Found latest global model v{version}: {latest_file}")
            return latest_file, version
        
        logger.warning(f"No global model with version {version} found, looking for any global model")
    
    # Look for any global model
    pattern = os.path.join(models_dir, "global_model*.npz")
    files = glob.glob(pattern)
    
    if not files:
        logger.warning("No global model found")
        return None, None
    
    # Sort by timestamp
    latest_file = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0, reverse=True)[0]
    
    # Extract version from filename
    version_match = re.search(r'global_model_v(\d+)_', latest_file)
    found_version = int(version_match.group(1)) if version_match else 1
    
    logger.info(f"Found global model v{found_version}: {latest_file}")
    return latest_file, found_version

def extract_timestamp(filename):
    """Extract timestamp from filename"""
    try:
        # Try to extract timestamp from the end of the filename
        return int(filename.split('_')[-1].split('.')[0])
    except (ValueError, IndexError):
        logger.warning(f"Could not extract timestamp from {filename}")
        return 0

def extract_version_from_filename(filename):
    """Extract version from filename"""
    try:
        # Look for v{number} pattern in the filename
        version_match = re.search(r'_v(\d+)_', filename)
        if version_match:
            return int(version_match.group(1))
        return None
    except (ValueError, IndexError):
        logger.warning(f"Could not extract version from {filename}")
        return None

def find_client_models_for_verification(global_model_path, global_version, models_dir="models/npz"):
    """Find client models that were used to create the given global model
    
    For global model vX, we need to find client models with version v(X-1)
    because client models vN are based on global model vN and are aggregated
    to create global model v(N+1)
    """
    if not global_model_path:
        logger.error("No global model provided")
        return []
    
    # The client models we want have the PREVIOUS version compared to the global model
    previous_version = global_version - 1
    
    if previous_version < 1:
        logger.warning(f"Cannot find client models for global model v{global_version} as it appears to be the initial model")
        return []
    
    logger.info(f"Looking for client models with version v{previous_version} (used to create global model v{global_version})")
    
    # Find all client models with the previous version
    pattern = os.path.join(models_dir, f"model_client*_v{previous_version}_*.npz")
    client_models = glob.glob(pattern)
    
    if not client_models:
        logger.warning(f"No client models found with version v{previous_version}")
        
        # As a fallback, try to find any client models
        fallback_pattern = os.path.join(models_dir, "model_client*_*.npz")
        all_client_files = glob.glob(fallback_pattern)
        
        # Extract timestamp from global model
        global_timestamp = extract_timestamp(global_model_path)
        
        # Filter client models created before global model
        client_models = []
        for file in all_client_files:
            timestamp = extract_timestamp(file)
            if timestamp < global_timestamp:
                client_models.append(file)
        
        if client_models:
            logger.warning(f"Using {len(client_models)} client models based on timestamp instead of version")
    
    if not client_models:
        logger.warning("No client models found for verification")
        return []
    
    logger.info(f"Found {len(client_models)} client models for verification")
    for file in client_models:
        version = extract_version_from_filename(file)
        logger.info(f"  - {os.path.basename(file)} (version: {version})")
    
    return client_models

def aggregate_models(client_model_paths, global_model_path=None):
    """Aggregate client models using the same algorithm as in pbft_node.py"""
    logger.info(f"Starting model aggregation verification with {len(client_model_paths)} client models")
    
    # Load all client models
    client_weights = []
    for path in client_model_paths:
        if not os.path.exists(path):
            logger.warning(f"Client model file not found: {path}")
            continue
            
        logger.info(f"Loading client model from {path}")
        weights = load_model_weights(path)
        if weights:
            client_weights.append(weights)
        else:
            logger.warning(f"No valid weights loaded from {path}")
    
    if not client_weights:
        logger.error("No valid client models found, cannot perform aggregation")
        return None
    
    # Load global model if provided (for reference or initialization)
    global_weights = None
    if global_model_path and os.path.exists(global_model_path):
        logger.info(f"Loading global model from {global_model_path}")
        global_weights = load_model_weights(global_model_path)
    
    # Use the first client model to determine the structure if global model not available
    reference_weights = global_weights if global_weights else client_weights[0]
    
    # Initialize aggregated weights with zeros of the same shape
    aggregated_weights = {}
    for key in reference_weights:
        # Convert to float64 to avoid casting issues
        aggregated_weights[key] = np.zeros_like(reference_weights[key], dtype=np.float64)
    
    # Average all client models (equal weights)
    logger.info(f"Performing simple averaging of {len(client_weights)} models")
    for model_weights in client_weights:
        for key in aggregated_weights:
            if key in model_weights:
                try:
                    # Convert to float64 before adding to avoid casting issues
                    weight_array = model_weights[key].astype(np.float64)
                    aggregated_weights[key] += weight_array / len(client_weights)
                except Exception as e:
                    logger.warning(f"Error adding weights for key {key}: {e}")
    
    # Convert back to original dtype if needed
    if global_weights:
        for key in aggregated_weights:
            if key in global_weights:
                original_dtype = global_weights[key].dtype
                if original_dtype != np.float64:
                    try:
                        aggregated_weights[key] = aggregated_weights[key].astype(original_dtype)
                    except TypeError:
                        logger.warning(f"Could not cast {key} back to {original_dtype}, keeping as float64")
    
    return aggregated_weights

def save_aggregated_model(aggregated_weights, output_path):
    """Save the aggregated model to a file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with a context manager to ensure proper closing
        with open(output_path, 'wb') as f:
            np.savez(f, **aggregated_weights)
        
        # Verify the file was saved correctly
        with np.load(output_path, allow_pickle=True) as test_load:
            # Just check if we can access the file
            for key in test_load.files:
                _ = test_load[key].shape
        
        # Calculate hash of the model file
        with open(output_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
        
        logger.info(f"Saved aggregated model to {output_path}")
        logger.info(f"Model hash: {model_hash}")
        
        return model_hash
    except Exception as e:
        logger.error(f"Error saving aggregated model: {e}")
        return None

def compare_models(model1_path, model2_path):
    """Compare two models to check if they are the same"""
    logger.info(f"Comparing models: {model1_path} and {model2_path}")
    
    # Load both models
    model1 = load_model_weights(model1_path)
    model2 = load_model_weights(model2_path)
    
    if not model1 or not model2:
        logger.error("Failed to load one or both models for comparison")
        return False
    
    # Check if they have the same keys
    if set(model1.keys()) != set(model2.keys()):
        logger.warning("Models have different keys")
        return False
    
    # Compare each weight tensor
    all_equal = True
    max_diff = 0
    
    for key in model1:
        if not np.array_equal(model1[key], model2[key]):
            # Check if they're close (floating point precision issues)
            diff = np.abs(model1[key] - model2[key]).max()
            max_diff = max(max_diff, diff)
            
            if diff > 1e-5:  # Tolerance threshold
                logger.warning(f"Key {key} differs with max difference: {diff}")
                all_equal = False
    
    if all_equal:
        logger.info(f"Models are identical (max difference: {max_diff})")
    else:
        logger.warning(f"Models are different (max difference: {max_diff})")
    
    return all_equal

def main():
    # Directory containing model files
    models_dir = "models/npz"
    os.makedirs(models_dir, exist_ok=True)
    
    # Specify which global model version to verify
    version_to_verify = 3  # Change this to verify different versions
    
    # Step 1: Find the specific global model version
    logger.info(f"Step 1: Finding global model v{version_to_verify}")
    global_model_path, global_version = find_global_model(version=version_to_verify, models_dir=models_dir)
    
    if not global_model_path:
        logger.error(f"Global model v{version_to_verify} not found, exiting")
        return
    
    # For v1, we need special handling as it's the initial model
    if version_to_verify == 1:
        logger.info("Global model v1 is the initial model, no client models to verify")
        logger.info("Verification SKIPPED: Initial models don't have client models to aggregate")
        return
    
    # Step 2: Find client models that were used to create this global model
    logger.info(f"Step 2: Finding client models that were used to create global model v{global_version}")
    client_model_paths = find_client_models_for_verification(global_model_path, global_version, models_dir)
    
    if not client_model_paths:
        logger.error("No client models found for verification, exiting")
        return
    
    # Find the previous global model to use as a base for aggregation
    previous_version = global_version - 1
    previous_global_model_path, _ = find_global_model(version=previous_version, models_dir=models_dir)
    
    if not previous_global_model_path:
        logger.warning(f"Previous global model v{previous_version} not found, using client models only for verification")
    else:
        logger.info(f"Using previous global model v{previous_version} as base for aggregation")
    
    # Step 3: Perform independent aggregation
    logger.info(f"Step 3: Performing independent aggregation of v{previous_version} client models")
    aggregated_weights = aggregate_models(client_model_paths, previous_global_model_path)
    
    if not aggregated_weights:
        logger.error("Aggregation failed, exiting")
        return
    
    # Step 4: Save our independently aggregated model
    timestamp = int(time.time())
    output_path = os.path.join(models_dir, f"verification_model_v{global_version}_{timestamp}.npz")
    logger.info(f"Step 4: Saving independently aggregated model (v{global_version})")
    our_model_hash = save_aggregated_model(aggregated_weights, output_path)
    
    if not our_model_hash:
        logger.error("Failed to save aggregated model, exiting")
        return
    
    # Step 5: Compare with the actual global model
    logger.info(f"Step 5: Comparing our aggregation with global model v{global_version}")
    
    is_same = compare_models(output_path, global_model_path)
    
    if is_same:
        logger.info(f"✅ Verification SUCCESSFUL: Our aggregation of v{previous_version} client models matches the global model v{global_version}")
    else:
        logger.warning(f"❌ Verification FAILED: Our aggregation of v{previous_version} client models differs from the global model v{global_version}")
        
    # Print model details for debugging
    logger.info(f"Our model: {output_path}")
    logger.info(f"Global model: {global_model_path}")

if __name__ == "__main__":
    main()