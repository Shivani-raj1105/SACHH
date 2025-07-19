import os
import json

def check_model_files():
    print("=== MODEL STATUS CHECK ===")
    
    saved_model_path = "saved_model"
    print(f"\nChecking {saved_model_path}/")
    
    if os.path.exists(saved_model_path):
        files = os.listdir(saved_model_path)
        for file in files:
            file_path = os.path.join(saved_model_path, file)
            size = os.path.getsize(file_path)
            print(f"  {file}: {size} bytes")
            
            if file == "config.json":
                try:
                    with open(file_path, 'r') as f:
                        config = json.load(f)
                    print(f"    Model type: {config.get('model_type', 'Unknown')}")
                    print(f"    Num labels: {config.get('num_labels', 'Unknown')}")
                except Exception as e:
                    print(f"    Error reading config: {e}")
    else:
        print("  Directory does not exist!")
    
    deployment_path = "deployment/saved_model"
    print(f"\nChecking {deployment_path}/")
    
    if os.path.exists(deployment_path):
        files = os.listdir(deployment_path)
        for file in files:
            file_path = os.path.join(deployment_path, file)
            size = os.path.getsize(file_path)
            print(f"  {file}: {size} bytes")
    else:
        print("  Directory does not exist!")
    
    print("\nLooking for model weight files...")
    weight_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(('.bin', '.safetensors', '.pt', '.pth')):
                weight_files.append(os.path.join(root, file))
    
    if weight_files:
        print("  Found model weight files:")
        for file in weight_files:
            size = os.path.getsize(file)
            print(f"    {file}: {size} bytes")
    else:
        print("  No model weight files found!")
        print("  This means the model needs to be trained!")

if __name__ == "__main__":
    check_model_files() 