#!/usr/bin/env python3
"""
Start vLLM server with VerMind model registration.

The VerMind model is automatically registered via vLLM's plugin system
when the package is installed (via entry_points in pyproject.toml).

If the package is not installed, you can manually register by importing:
    from vllm_adapter.plugin import register_vermind_plugin
    register_vermind_plugin()
"""

import sys
import os
import json
import shutil

# Add parent directory to path (for development mode)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set PYTHONPATH for subprocesses
os.environ["PYTHONPATH"] = f"/root/vermind:{os.environ.get('PYTHONPATH', '')}"


def ensure_model_config_complete(model_path: str):
    """
    确保模型文件夹的配置完整：
    1. 检查 config.json 是否包含 auto_map（包含 AutoConfig 和 AutoModelForCausalLM）
    2. 检查是否有 configuration_vermind.py 和 modeling_vermind.py
    3. 如果缺失，自动从 vllm_adapter 目录复制
    
    Args:
        model_path: 模型文件夹路径
    """
    if not os.path.isdir(model_path):
        print(f"⚠️  模型路径不是目录: {model_path}")
        return
    
    adapter_dir = os.path.dirname(__file__)
    config_json_path = os.path.join(model_path, "config.json")
    config_py_path = os.path.join(model_path, "configuration_vermind.py")
    modeling_py_path = os.path.join(model_path, "modeling_vermind.py")
    
    source_config_py = os.path.join(adapter_dir, "configuration_vermind.py")
    source_modeling_py = os.path.join(adapter_dir, "modeling_vermind.py")
    
    needs_update = False
    files_copied = []
    
    # 1. 检查并更新 config.json
    if os.path.exists(config_json_path):
        try:
            with open(config_json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 检查 auto_map 是否存在且完整
            required_auto_map = {
                "AutoConfig": "configuration_vermind.VerMindConfig",
                "AutoModelForCausalLM": "modeling_vermind.VerMindForCausalLM"
            }
            
            if "auto_map" not in config:
                print(f"📝 检测到 config.json 缺少 auto_map，正在添加...")
                config["auto_map"] = required_auto_map
                needs_update = True
            else:
                # 检查是否完整
                auto_map = config["auto_map"]
                for key, value in required_auto_map.items():
                    if key not in auto_map or auto_map[key] != value:
                        print(f"📝 检测到 config.json 的 auto_map 不完整，正在更新...")
                        if "auto_map" not in config:
                            config["auto_map"] = {}
                        config["auto_map"][key] = value
                        needs_update = True
            
            if needs_update:
                # 备份原文件
                backup_path = config_json_path + ".backup"
                if not os.path.exists(backup_path):
                    shutil.copy2(config_json_path, backup_path)
                    print(f"   💾 已备份原 config.json 到 {os.path.basename(backup_path)}")
                
                # 写入更新后的配置
                with open(config_json_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print(f"   ✅ 已更新 config.json")
        except Exception as e:
            print(f"⚠️  读取/更新 config.json 时出错: {e}")
    else:
        print(f"⚠️  未找到 config.json: {config_json_path}")
    
    # 2. 检查并复制 configuration_vermind.py
    if not os.path.exists(config_py_path):
        if os.path.exists(source_config_py):
            try:
                shutil.copy2(source_config_py, config_py_path)
                files_copied.append("configuration_vermind.py")
                print(f"   ✅ 已复制 configuration_vermind.py 到模型文件夹")
            except Exception as e:
                print(f"   ❌ 复制 configuration_vermind.py 失败: {e}")
        else:
            print(f"   ⚠️  源文件不存在: {source_config_py}")
    else:
        print(f"   ✓ configuration_vermind.py 已存在")
    
    # 3. 检查并复制 modeling_vermind.py
    if not os.path.exists(modeling_py_path):
        if os.path.exists(source_modeling_py):
            try:
                shutil.copy2(source_modeling_py, modeling_py_path)
                files_copied.append("modeling_vermind.py")
                print(f"   ✅ 已复制 modeling_vermind.py 到模型文件夹")
            except Exception as e:
                print(f"   ❌ 复制 modeling_vermind.py 失败: {e}")
        else:
            print(f"   ⚠️  源文件不存在: {source_modeling_py}")
    else:
        print(f"   ✓ modeling_vermind.py 已存在")
    
    # 总结
    if needs_update or files_copied:
        print(f"   📋 配置补全完成: {'已更新 config.json' if needs_update else ''} {'已复制文件: ' + ', '.join(files_copied) if files_copied else ''}")
    else:
        print(f"   ✓ 所有配置文件完整，无需补全")


# 从命令行参数中提取模型路径（在设置 sys.argv 之前）
# 检查是否有模型路径参数
model_path = None
original_argv = sys.argv.copy()

if len(original_argv) > 1:
    # 查找第一个看起来像路径的参数（不是以 -- 开头）
    for arg in original_argv[1:]:
        if not arg.startswith('--') and (os.path.exists(arg) or os.path.isdir(arg)):
            model_path = arg
            break

# 如果从 sys.argv 中找到了模型路径，进行配置检查和补全
if model_path:
    print(f"🔍 检查模型配置完整性: {model_path}")
    ensure_model_config_complete(model_path)
    print()

# CRITICAL: Register plugin BEFORE importing any vLLM modules
# This ensures the model is registered before vLLM validates architectures
# Also ensure plugin is loaded in subprocesses by setting up the plugin system
try:
    from vllm_adapter.plugin import register_vermind_plugin
    register_vermind_plugin()
    print("✅ VerMind plugin registered successfully")
except ImportError as e:
    print(f"⚠️  Warning: Could not import plugin: {e}")
    # Try to register manually
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from vllm_adapter.plugin import register_vermind_plugin
        register_vermind_plugin()
        print("✅ VerMind plugin registered (fallback)")
    except Exception as e2:
        print(f"❌ Failed to register plugin: {e2}")
        sys.exit(1)

# Verify registration before proceeding
try:
    from vllm import ModelRegistry
    supported = list(ModelRegistry.get_supported_archs())
    if "VerMindForCausalLM" not in supported:
        print(f"❌ ERROR: VerMindForCausalLM not found in supported architectures!")
        print(f"   Supported: {supported[:10]}...")
        sys.exit(1)
    print(f"✅ VerMindForCausalLM is registered in ModelRegistry")
except Exception as e:
    print(f"⚠️  Warning: Could not verify registration: {e}")

# Ensure plugin is available for subprocesses by monkey-patching vLLM's plugin loader
# This is needed because vLLM loads plugins in subprocesses, and entry_points may not be available
try:
    # Import vLLM's plugin system and ensure our plugin is registered
    import vllm.plugins as vllm_plugins
    # Register our plugin function so it's available when vLLM loads plugins
    if not hasattr(vllm_plugins, '_manual_plugins'):
        vllm_plugins._manual_plugins = {}
    vllm_plugins._manual_plugins['vllm.general_plugins'] = vllm_plugins._manual_plugins.get('vllm.general_plugins', {})
    from vllm_adapter.plugin import register_vermind_plugin
    vllm_plugins._manual_plugins['vllm.general_plugins']['vermind'] = register_vermind_plugin
    
    # Monkey-patch load_plugins_by_group to include our manual plugin
    original_load = getattr(vllm_plugins, 'load_plugins_by_group', None)
    if original_load:
        def patched_load_plugins_by_group(group):
            result = original_load(group) if original_load else {}
            # Add our manual plugin if not already loaded
            if group == 'vllm.general_plugins' and 'vermind' not in result:
                result['vermind'] = register_vermind_plugin
            return result
        vllm_plugins.load_plugins_by_group = patched_load_plugins_by_group
        print("✅ VerMind plugin patched into vLLM plugin system")
except Exception as e:
    # If patching fails, that's okay - manual registration should still work
    print(f"⚠️  Note: Could not patch plugin system (manual registration should work): {e}")

# Now use vLLM's CLI interface
# Set sys.argv to match vLLM's expected arguments
# 如果之前没有从 sys.argv 提取到模型路径，使用默认值
if model_path is None:
    model_path = "/root/vermind/output/pretrain/pretrain_768/checkpoint_10000"
    # 使用默认路径时，也需要检查配置
    if os.path.exists(model_path):
        print(f"🔍 检查模型配置完整性: {model_path}")
        ensure_model_config_complete(model_path)
        print()

sys.argv = [
    "vllm",
    "serve",
    model_path,
    "--gpu-memory-utilization", "0.1",
    "--trust-remote-code",
    "--port", "8000",
    "--host", "0.0.0.0",
]

# Import and run vLLM's main entry point
# This import happens AFTER registration
if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main
    main()
