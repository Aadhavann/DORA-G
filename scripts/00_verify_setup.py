"""
Script 0: Verify installation and environment setup.
Run this first to ensure everything is configured correctly.
"""

import sys
import subprocess
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_header(text):
    """Print section header."""
    print(f"\n{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BLUE}{text:^80}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*80}{Colors.RESET}\n")


def print_check(name, passed, details=""):
    """Print check result."""
    status = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"
    print(f"{status} {name}")
    if details:
        print(f"  {details}")


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    required = (3, 10)

    passed = version >= required
    details = f"Found: Python {version.major}.{version.minor}.{version.micro}"
    if not passed:
        details += f" (Required: >= {required[0]}.{required[1]})"

    print_check("Python version", passed, details)
    return passed


def check_package_installed(package_name, import_name=None):
    """Check if a package is installed."""
    import_name = import_name or package_name

    try:
        __import__(import_name)
        print_check(f"Package: {package_name}", True)
        return True
    except ImportError:
        print_check(f"Package: {package_name}", False, f"Run: pip install {package_name}")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9

            details = f"Found {device_count} GPU(s): {device_name} ({memory:.1f} GB)"
            print_check("CUDA", True, details)

            if memory < 40:
                print(f"  {Colors.YELLOW}⚠ Warning: GPU has <40GB memory. Consider reducing batch size.{Colors.RESET}")
        else:
            print_check("CUDA", False, "No CUDA-capable GPU found. Training will be slow.")

        return cuda_available
    except ImportError:
        print_check("CUDA", False, "PyTorch not installed")
        return False


def check_directories():
    """Check required directories exist."""
    required_dirs = [
        "configs",
        "src/models",
        "src/data",
        "src/retrieval",
        "src/evaluation",
        "scripts",
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        exists = path.exists()
        if not exists:
            all_exist = False
        print_check(f"Directory: {dir_path}", exists)

    return all_exist


def check_config_files():
    """Check configuration files exist."""
    required_configs = [
        "configs/base.yaml",
        "configs/dora.yaml",
        "configs/lora.yaml",
        "configs/rag.yaml",
    ]

    all_exist = True
    for config_path in required_configs:
        path = Path(config_path)
        exists = path.exists()
        if not exists:
            all_exist = False
        print_check(f"Config: {config_path}", exists)

    return all_exist


def check_wandb():
    """Check Weights & Biases configuration."""
    try:
        import wandb

        # Check if logged in
        try:
            api = wandb.Api()
            username = api.viewer.get("username", "unknown")
            print_check("W&B Login", True, f"Logged in as: {username}")
            return True
        except Exception:
            print_check("W&B Login", False, "Run: wandb login")
            return False
    except ImportError:
        print_check("W&B", False, "Run: pip install wandb")
        return False


def check_disk_space():
    """Check available disk space."""
    import shutil

    stats = shutil.disk_usage(".")
    free_gb = stats.free / (1024**3)

    required_gb = 200
    passed = free_gb >= required_gb

    details = f"Available: {free_gb:.1f} GB"
    if not passed:
        details += f" (Required: {required_gb} GB)"

    print_check("Disk space", passed, details)
    return passed


def check_huggingface_cache():
    """Check HuggingFace cache directory."""
    import os

    cache_dir = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    cache_path = Path(cache_dir)

    print_check("HuggingFace cache", True, f"Location: {cache_path}")

    if not cache_path.exists():
        print(f"  {Colors.YELLOW}⚠ Cache directory will be created on first use{Colors.RESET}")

    return True


def main():
    """Run all verification checks."""
    print_header("DoRA-G Setup Verification")

    print(f"{Colors.BLUE}Checking installation and configuration...{Colors.RESET}\n")

    results = {}

    # Python version
    print_header("1. Python Environment")
    results["python"] = check_python_version()

    # Required packages
    print_header("2. Required Packages")
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("datasets", "datasets"),
        ("faiss", "faiss"),
        ("sentence-transformers", "sentence_transformers"),
        ("hydra", "hydra"),
        ("wandb", "wandb"),
    ]

    results["packages"] = all(check_package_installed(pkg, imp) for pkg, imp in packages)

    # CUDA
    print_header("3. GPU & CUDA")
    results["cuda"] = check_cuda()

    # Directories
    print_header("4. Project Structure")
    results["dirs"] = check_directories()

    # Configs
    print_header("5. Configuration Files")
    results["configs"] = check_config_files()

    # W&B
    print_header("6. Experiment Tracking")
    results["wandb"] = check_wandb()

    # Storage
    print_header("7. Storage")
    results["disk"] = check_disk_space()
    results["hf_cache"] = check_huggingface_cache()

    # Summary
    print_header("Summary")

    passed_count = sum(results.values())
    total_count = len(results)

    print(f"Passed: {passed_count}/{total_count} checks\n")

    if all(results.values()):
        print(f"{Colors.GREEN}✓ All checks passed! You're ready to start.{Colors.RESET}")
        print(f"\n{Colors.BLUE}Next steps:{Colors.RESET}")
        print("  1. Review configs in configs/")
        print("  2. Run: python scripts/01_prepare_data.py")
        print("  3. See QUICKSTART.md for detailed guide")
    else:
        print(f"{Colors.RED}✗ Some checks failed. Please fix the issues above.{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Common fixes:{Colors.RESET}")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Login to W&B: wandb login")
        print("  - Check CUDA installation: nvidia-smi")

    print(f"\n{Colors.BLUE}{'='*80}{Colors.RESET}\n")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
