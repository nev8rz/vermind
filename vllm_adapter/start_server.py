#!/usr/bin/env python3
"""
Start vLLM server with VerMind model registration.

Supports:
- VerMind: Pure language model
- VerMind-V: Vision-language model (text-only mode in vLLM)

The VerMind model is automatically registered via vLLM's plugin system
when the package is installed (via entry_points in pyproject.toml).
"""

import sys
import os
import json
import shutil

# Add parent directory to path (for development mode)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set PYTHONPATH for subprocesses
os.environ["PYTHONPATH"] = f"/root/vermind:{os.environ.get('PYTHONPATH', '')}"


def is_lora_checkpoint(model_path: str) -> bool:
    """Check if path is a LoRA checkpoint."""
    if not os.path.isdir(model_path):
        return False
    
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    config_json_path = os.path.join(model_path, "config.json")
    adapter_model_path = os.path.join(model_path, "adapter_model.safetensors")
    
    return (os.path.exists(adapter_config_path) and 
            (os.path.exists(adapter_model_path) or os.path.exists(os.path.join(model_path, "adapter_model.bin"))) and
            not os.path.exists(config_json_path))


def detect_model_type(model_path: str) -> str:
    """
    Detect whether the model is VerMind or VerMind-V.
    
    Returns:
        "vermind" for pure language model
        "vermind-v" for vision-language model
    """
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            model_type = config.get("model_type", "")
            if model_type == "vermind-v":
                return "vermind-v"
        except Exception:
            pass
    return "vermind"


def ensure_model_config_complete(model_path: str):
    """
    Ensure model folder has complete configuration.
    Copies necessary files from vllm_adapter if missing.
    """
    if not os.path.isdir(model_path):
        print(f"⚠️  Model path is not a directory: {model_path}")
        return
    
    if is_lora_checkpoint(model_path):
        print(f"⚠️  Detected LoRA checkpoint! Please merge first:")
        print(f"   python scripts/merge_lora.py --model_path <base> --lora_path {model_path}")
        sys.exit(1)
    
    model_type = detect_model_type(model_path)
    adapter_dir = os.path.dirname(__file__)
    
    if model_type == "vermind-v":
        # Use VLM config files
        source_config_py = os.path.join(adapter_dir, "vlm/configuration_vermind_v.py")
        source_modeling_py = os.path.join(adapter_dir, "vlm/modeling_vermind_v.py")
        print(f"📷 Detected VerMind-V (VLM) model")
    else:
        # Use standard config files
        source_config_py = os.path.join(adapter_dir, "core/configuration_vermind.py")
        source_modeling_py = os.path.join(adapter_dir, "core/modeling_vermind.py")
    
    config_json_path = os.path.join(model_path, "config.json")
    config_py_path = os.path.join(model_path, os.path.basename(source_config_py))
    modeling_py_path = os.path.join(model_path, os.path.basename(source_modeling_py))
    
    needs_update = False
    files_copied = []
    
    # Check and update config.json
    if os.path.exists(config_json_path):
        try:
            with open(config_json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if model_type == "vermind-v":
                required_auto_map = {
                    "AutoConfig": "configuration_vermind_v.VLMConfig",
                    "AutoModelForCausalLM": "modeling_vermind_v.VerMindVLM"
                }
            else:
                required_auto_map = {
                    "AutoConfig": "configuration_vermind.VerMindConfig",
                    "AutoModelForCausalLM": "modeling_vermind.VerMindForCausalLM"
                }
            
            if "auto_map" not in config:
                print(f"📝 Adding auto_map to config.json...")
                config["auto_map"] = required_auto_map
                needs_update = True
            else:
                auto_map = config["auto_map"]
                for key, value in required_auto_map.items():
                    if key not in auto_map or auto_map[key] != value:
                        print(f"📝 Updating auto_map in config.json...")
                        config["auto_map"] = required_auto_map
                        needs_update = True
                        break
            
            if needs_update:
                backup_path = config_json_path + ".backup"
                if not os.path.exists(backup_path):
                    shutil.copy2(config_json_path, backup_path)
                    print(f"   💾 Backed up config.json")
                
                with open(config_json_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print(f"   ✅ Updated config.json")
        except Exception as e:
            print(f"⚠️  Error updating config.json: {e}")
    else:
        print(f"⚠️  config.json not found: {config_json_path}")
    
    # Copy configuration file
    if not os.path.exists(config_py_path):
        if os.path.exists(source_config_py):
            try:
                shutil.copy2(source_config_py, config_py_path)
                files_copied.append(os.path.basename(source_config_py))
                print(f"   ✅ Copied {os.path.basename(source_config_py)}")
            except Exception as e:
                print(f"   ❌ Failed to copy: {e}")
        else:
            print(f"   ⚠️  Source not found: {source_config_py}")
    else:
        print(f"   ✓ {os.path.basename(config_py_path)} exists")
    
    # Copy modeling file
    if not os.path.exists(modeling_py_path):
        if os.path.exists(source_modeling_py):
            try:
                shutil.copy2(source_modeling_py, modeling_py_path)
                files_copied.append(os.path.basename(source_modeling_py))
                print(f"   ✅ Copied {os.path.basename(source_modeling_py)}")
            except Exception as e:
                print(f"   ❌ Failed to copy: {e}")
        else:
            print(f"   ⚠️  Source not found: {source_modeling_py}")
    else:
        print(f"   ✓ {os.path.basename(modeling_py_path)} exists")
    
    if needs_update or files_copied:
        print(f"   📋 Configuration complete")
    else:
        print(f"   ✓ All config files present")
    
    return model_type


# Extract model path from command line
model_path = None
original_argv = sys.argv.copy()

if len(original_argv) > 1:
    for arg in original_argv[1:]:
        if not arg.startswith('--') and (os.path.exists(arg) or os.path.isdir(arg)):
            model_path = arg
            break

# Check and complete model config
if model_path:
    print(f"🔍 Checking model configuration: {model_path}")
    model_type = ensure_model_config_complete(model_path)
    print()
else:
    model_type = "vermind"

# Register plugin BEFORE importing vLLM
try:
    from vllm_adapter.plugin import register_vermind_plugin
    register_vermind_plugin()
    print("✅ VerMind plugin registered")
except ImportError as e:
    print(f"⚠️  Could not import plugin: {e}")
    sys.exit(1)

# Also register VLM plugin if needed
if model_type == "vermind-v":
    try:
        from vllm_adapter.plugin import register_vermind_v_plugin
        register_vermind_v_plugin()
        print("✅ VerMind-V plugin registered (text-only mode)")
        print("   Note: For full VLM inference with images, use standard inference script")
    except Exception as e:
        print(f"⚠️  Could not register VLM plugin: {e}")

# Verify registration
try:
    from vllm import ModelRegistry
    supported = list(ModelRegistry.get_supported_archs())
    if "VerMindForCausalLM" not in supported:
        print(f"❌ ERROR: VerMindForCausalLM not registered!")
        sys.exit(1)
    print(f"✅ VerMindForCausalLM registered")
except Exception as e:
    print(f"⚠️  Could not verify registration: {e}")

# Default model path
if model_path is None:
    model_path = "/root/vermind/output/pretrain/pretrain_768/checkpoint_10000"
    if os.path.exists(model_path):
        print(f"🔍 Checking default model: {model_path}")
        ensure_model_config_complete(model_path)
        print()

# Set up vLLM CLI arguments
sys.argv = [
    "vllm",
    "serve",
    model_path,
    "--gpu-memory-utilization", "0.5",
    "--trust-remote-code",
    "--port", "8000",
    "--host", "0.0.0.0",
    "--served-model-name", model_type,
    "--max-model-len", "3072",
    "--tokenizer", model_path
]

# Run vLLM
if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main
    main()
