import os
import subprocess
import sys

def run_command(command):
    print(f"Executing: {command}")
    try:
        # We use shell=True to allow for environment variable expansion
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

def install_mamba():
    # 1. Dynamically find the Conda Prefix
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        print("Error: CONDA_PREFIX not found. Please run this inside your conda environment.")
        return

    print(f"Found Conda Prefix: {conda_prefix}")

    # 2. Configure Environment Variables for the Compiler
    # This points specifically to the headers we found in your 'targets' folder
    cuda_include = os.path.join(conda_prefix, "targets/x86_64-linux/include")
    cuda_lib = os.path.join(conda_prefix, "targets/x86_64-linux/lib")
    
    os.environ["CUDA_HOME"] = conda_prefix
    os.environ["CPATH"] = f"{cuda_include}:{os.environ.get('CPATH', '')}"
    os.environ["C_INCLUDE_PATH"] = f"{cuda_include}:{os.environ.get('C_INCLUDE_PATH', '')}"
    os.environ["CPLUS_INCLUDE_PATH"] = f"{cuda_include}:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"
    os.environ["LIBRARY_PATH"] = f"{cuda_lib}:{os.environ.get('LIBRARY_PATH', '')}"
    os.environ["PATH"] = f"{os.path.join(conda_prefix, 'bin')}:{os.environ.get('PATH', '')}"

    print("Environment variables configured for CUDA 12.x compilation.")

    # 3. Install Prerequisites
    print("Step 1/3: Installing build tools...")
    run_command("pip install wheel setuptools check-manifest packaging ninja --no-cache-dir")

    # 4. Install Causal Conv1d
    # This is the heavy lifting for the 1D spectral convolutions
    print("Step 2/3: Building causal-conv1d (this takes 2-5 minutes)...")
    run_command("pip install causal-conv1d>=1.4.0 --no-build-isolation --no-cache-dir")

    # 5. Install Mamba SSM
    print("Step 3/3: Building mamba-ssm...")
    run_command("pip install mamba-ssm --no-build-isolation --no-cache-dir")

    # 6. Final Verification
    print("\n" + "="*40)
    print("Verification: Testing Mamba and Causal Conv1d Imports")
    try:
        import torch
        import causal_conv1d_cuda
        from mamba_ssm import Mamba
        print("SUCCESS: All Mamba kernels are successfully compiled and linked!")
    except ImportError as e:
        print(f"VERIFICATION FAILED: {e}")
        print("Tip: Try restarting your terminal and activating your environment again.")

if __name__ == "__main__":
    install_mamba()
