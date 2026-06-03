import os
import subprocess
import sys


def check_docker():
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print(f"Docker detected: {result.stdout.strip()}")
            return True
        else:
            print("Warning: Docker is not available (not found in PATH).")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("Warning: Docker is not available.")
        return False


def ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Directory ready: {os.path.abspath(d)}")


def run_checks():
    print("--- Pre-flight checks ---")
    check_docker()
    ensure_dirs(["results", "models"])
    print("--- All checks passed ---\n")