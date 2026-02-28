import subprocess

print("Running pipeline...")
subprocess.run(["python", "main.py"])

print("\nRunning tests...")
subprocess.run(["python", "-m", "pytest"])