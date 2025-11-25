# vendor_state_loader.py
from __future__ import annotations
import os
from typing import Generator, List, Optional, Any
from pathlib import Path
import data.vendor_state_store as vendor_state_store
from states.vendorState import vendorState


def load_all_vendor_states(
    base_dir: str = "data/vendor_plans",
    *,
    gzip_compress: bool = False,
    as_list: bool = False
) -> Generator[tuple[str, vendorState], None, None] | List[tuple[str, vendorState]]:
    """
    Iteratively load all vendor state objects from subdirectories.
    
    Args:
        base_dir: Base directory containing vendor subdirectories (default: "data/vendor_plans")
        gzip_compress: Whether to look for .json.gz files (default: False)
        as_list: If True, return a list; if False, return a generator (default: False)
    
    Yields/Returns:
        Tuple of (vendor_code, vendorState) for each vendor found
        
    Example:
        # As generator (memory efficient)
        for vendor_code, vendor_state in load_all_vendor_states():
            print(f"Processing {vendor_code}")
            
        # As list
        all_states = load_all_vendor_states(as_list=True)
    """
    def _generate_vendor_states():
        # Ensure base directory exists
        if not os.path.exists(base_dir):
            print(f"Warning: Base directory '{base_dir}' does not exist")
            return
        
        # Iterate through all subdirectories
        for vendor_dir_name in os.listdir(base_dir):
            vendor_dir_path = os.path.join(base_dir, vendor_dir_name)
            
            # Skip if not a directory
            if not os.path.isdir(vendor_dir_path):
                continue
            
            # Look for vendor_state file
            ext = ".json.gz" if gzip_compress else ".json"
            vendor_state_file = os.path.join(vendor_dir_path, f"vendor_state{ext}")
            
            if os.path.exists(vendor_state_file):
                try:
                    vendor_state_data = vendor_state_store.load_vendor_state_blob(vendor_state_file)
                    vendor_state_obj = vendorState.model_validate(vendor_state_data)
                    yield (vendor_dir_name, vendor_state_obj)
                except Exception as e:
                    print(f"Error loading vendor state from {vendor_state_file}: {e}")
            else:
                # Try the other format if specified format not found
                alt_ext = ".json" if gzip_compress else ".json.gz"
                alt_file = os.path.join(vendor_dir_path, f"vendor_state{alt_ext}")
                if os.path.exists(alt_file):
                    try:
                        vendor_state_data = vendor_state_store.load_vendor_state_blob(alt_file)
                        vendor_state_obj = vendorState.model_validate(vendor_state_data)
                        yield (vendor_dir_name, vendor_state_obj)
                    except Exception as e:
                        print(f"Error loading vendor state from {alt_file}: {e}")
                else:
                    print(f"Warning: No vendor_state file found in {vendor_dir_path}")
    
    generator = _generate_vendor_states()
    return list(generator) if as_list else generator


def load_vendor_state_by_code(
    vendor_code: str,
    base_dir: str = "data/vendor_plans",
    *,
    gzip_compress: bool = False
) -> Optional[vendorState]:
    """
    Load a specific vendor state by vendor code.
    
    Args:
        vendor_code: The vendor code to load
        base_dir: Base directory containing vendor subdirectories
        gzip_compress: Whether to look for .json.gz files
        
    Returns:
        The vendorState object, or None if not found
        
    Example:
        vendor_state = load_vendor_state_by_code("VENDOR001")
    """
    vendor_dir_path = os.path.join(base_dir, str(vendor_code))
    
    if not os.path.exists(vendor_dir_path):
        print(f"Warning: Vendor directory '{vendor_dir_path}' does not exist")
        return None
    
    # Try preferred format first
    ext = ".json.gz" if gzip_compress else ".json"
    vendor_state_file = os.path.join(vendor_dir_path, f"vendor_state{ext}")
    
    if os.path.exists(vendor_state_file):
        try:
            vendor_state_data = vendor_state_store.load_vendor_state_blob(vendor_state_file)
            return vendorState.model_validate(vendor_state_data)
        except Exception as e:
            print(f"Error loading vendor state from {vendor_state_file}: {e}")
            return None
    
    # Try alternative format
    alt_ext = ".json" if gzip_compress else ".json.gz"
    alt_file = os.path.join(vendor_dir_path, f"vendor_state{alt_ext}")
    if os.path.exists(alt_file):
        try:
            vendor_state_data = vendor_state_store.load_vendor_state_blob(alt_file)
            return vendorState.model_validate(vendor_state_data)
        except Exception as e:
            print(f"Error loading vendor state from {alt_file}: {e}")
            return None
    
    print(f"Warning: No vendor_state file found for vendor code '{vendor_code}'")
    return None


def list_available_vendors(base_dir: str = "data/vendor_plans") -> List[str]:
    """
    List all available vendor codes in the base directory.
    
    Args:
        base_dir: Base directory containing vendor subdirectories
        
    Returns:
        List of vendor codes (subdirectory names)
        
    Example:
        vendors = list_available_vendors()
        print(f"Found {len(vendors)} vendors: {vendors}")
    """
    if not os.path.exists(base_dir):
        print(f"Warning: Base directory '{base_dir}' does not exist")
        return []
    
    vendor_codes = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains a vendor_state file
            has_json = os.path.exists(os.path.join(item_path, "vendor_state.json"))
            has_json_gz = os.path.exists(os.path.join(item_path, "vendor_state.json.gz"))
            if has_json or has_json_gz:
                vendor_codes.append(item)
    
    return sorted(vendor_codes)


if __name__ == "__main__":
    # Example usage
    print("Available vendors:")
    vendors = list_available_vendors()
    print(f"  Found {len(vendors)} vendor(s): {vendors}\n")
    
    print("Loading all vendor states iteratively:")
    for vendor_code, vendor_state in load_all_vendor_states():
        print(f"  Loaded vendor: {vendor_code}")
        print(f"    Vendor Code: {vendor_state.vendor_Code}")
        print(f"    Vendor Name: {vendor_state.vendor_name}")
        print(f"    Number of SKUs: {len(vendor_state.ChewySku_info)}")
        print(f"    Number of Plans: {vendor_state.numberofPlans()}")
        if vendor_state.numberofPlans() > 0:
            print(f"    First Plan Metrics: {vendor_state.container_plans[0].metrics}")

