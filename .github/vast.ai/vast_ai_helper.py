import argparse
import json
import re
import subprocess
import sys
from time import sleep
from typing import Any


def run_command(cmd: str, print_all: bool = False) -> str:
    # return subprocess.run([cmd], stdout=subprocess.PIPE, shell=True).stdout.decode(
    #     "utf-8"
    # )
    process = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    output_str_list = []
    total_output_str = ""
    for line in iter(process.stdout.readline, b""):  # mypy: ignore[union-attr]
        decoded = line.decode("utf-8")
        output_str_list.append(decoded)
        total_output_str += decoded + "\n"
        if print_all:
            print(decoded, end="")

    return total_output_str


def cleanup() -> None:
    out = run_command(
        "./vast.py show instances --raw",
    )
    dict_out = json.loads(out)

    for server in dict_out:
        print(f"Start destroying {server['id']}")
        out = run_command(
            f"./vast.py destroy instance {server['id']}",
        )
        print(out)


def startup() -> tuple[str, int]:
    out = run_command(
        "./vast.py search offers 'reliability > 0.98  num_gpus==1 rentable==True"
        " inet_down > 100 disk_space > 30 dph_total < 0.25 inet_down_cost < 0.021"
        " inet_up_cost < 0.021 cuda_vers >= 11.2' -o 'dph_total' --storage=32 --raw"
    )
    dict_out = json.loads(out)

    print("Starting best server")
    if len(dict_out) == 0:
        print("NO SERVER FOUND")
        sys.exit(1)
    print(dict_out[0])

    out = run_command(
        f"./vast.py create instance {dict_out[0]['id']} "
        "--image joennlae/halutmatmul-conda-gpu:latest --disk 32"
    )
    print(out)

    starting = True

    counter = 1

    ssh_host = ""
    ssh_port = 0
    while starting:
        print(f"Starting {counter}")
        sleep(5)
        out = run_command(f"./vast.py show instances --raw")
        out_dict = json.loads(out)
        if len(out_dict):
            print(out_dict[0]["status_msg"])
            if ssh_port == 0:
                ssh_host = out_dict[0]["ssh_host"]
                ssh_port = out_dict[0]["ssh_port"]
            if out_dict[0]["actual_status"] == "running":
                starting = False
        counter += 1

    return ssh_host, ssh_port


def run_ssh_commands(ssh_host: str, ssh_port: int, debug: bool = False) -> int:
    # commands to execute the tests
    # mv /venv/ /venv2 # because vast.ai already has venv
    # source /venv2/bin/activate
    # git clone https://github.com/joennlae/halutmatmul.git
    # cd halutmatmul
    # pytest -n4 -srPA src/python/test/test_kernel_gpu.py

    # currently using 4 jobs in parallel
    commands = 'cd /; mv /venv/ /venv2; source /venv2/bin/activate; \
      git clone https://github.com/joennlae/halutmatmul.git; \
      cd halutmatmul; pytest -n0 -srPA -k "gpu"; \
      echo "ERROR CODE: $?";'

    ssh_identity_str = ""
    if debug:
        ssh_identity_str = "-i .ssh/id_rsa"
    print("SSH host", ssh_host)
    print("SSH port", ssh_port)
    out = run_command(
        f"ssh -o StrictHostKeyChecking=no {ssh_identity_str} "
        f'-p {ssh_port} root@{ssh_host} "{commands}"',
        print_all=True,
    )
    error_code = re.findall(r"(?<=ERROR CODE: )\d+", out)
    print("ERROR CODE: ", error_code)

    failures = re.findall(r"\d+ failed", out)
    return int(error_code[0]) + len(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vast.ai helper")
    parser.add_argument("--cleanup", "-c", action="store_true", help="run only cleanup")
    parser.add_argument(
        "--debug", "-d", action="store_true", help="set ssh key offline"
    )
    args = parser.parse_args()

    print(args)
    if args.cleanup:
        cleanup()
    else:
        cleanup()
        ssh_host, ssh_port = startup()
        # ssh_host = "ssh4.vast.ai"
        # ssh_port = 11182
        sleep(5)
        error_code = run_ssh_commands(ssh_host, ssh_port, args.debug)
        cleanup()
        sys.exit(error_code)
