import subprocess

subprocess.run(["python3", "dismantle.py"], check=True)
subprocess.run(["gcc", "../c_backend/block_power.c", "-o", "../c_backend/block_power", "-lm"], check=True)
subprocess.run(["../c_backend/block_power"], check=True)
subprocess.run(["python3", "reconstruction.py"], check=True)
subprocess.run(["python3", "forbenius.py"], check = True)