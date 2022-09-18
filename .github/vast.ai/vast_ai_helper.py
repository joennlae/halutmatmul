import argparse
import json
import os
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


# image_uuid
# joennlae/halutmatmul-conda-gpu:latest
# joennlae/halutmatmul-conda-hw:latest
def cleanup(image_name: str = "") -> None:
    out = run_command(
        "./vast.py show instances --raw",
    )
    list_out = json.loads(out)

    if len(image_name) > 0:
        list_out = list(filter(lambda x: x["image_uuid"] == image_name, list_out))

    for server in list_out:
        print(f"Start destroying {server['id']}")
        out = run_command(
            f"./vast.py destroy instance {server['id']}",
        )
        print(out)


GPU_TEST_SEARCH = (
    "./vast.py search offers 'reliability > 0.98  num_gpus==1 rentable==True"
    " inet_down > 100 disk_space > 50 dph_total < 0.35 inet_down_cost < 0.021"
    " inet_up_cost < 0.021 cuda_vers >= 11.2' -o 'cpu_cores_effective-' --storage=32 --raw"
)

HW_TEST_SEARCH = (
    "./vast.py search offers 'reliability > 0.98 rentable==True inet_down > 100"
    " disk_space > 52 dph_total < 0.40 inet_down_cost < 0.021 inet_up_cost < 0.021"
    " ' -o 'cpu_cores_effective-' --storage=32 --raw"
)

GPU_TEST_COMMAND = "--image joennlae/halutmatmul-conda-gpu:latest --disk 32"
HW_TEST_COMMAND = "--image joennlae/halutmatmul-conda-hw:latest --disk 32"


def startup(
    image: str = "joennlae/halutmatmul-conda-gpu:latest", is_hardware: bool = False
) -> tuple[str, int]:
    out = run_command(HW_TEST_SEARCH if is_hardware else GPU_TEST_SEARCH)
    list_out = json.loads(out)

    print("Starting best server")
    if len(list_out) == 0:
        print("NO SERVER FOUND")
        sys.exit(1)
    if is_hardware:
        for idx, elem in enumerate(list_out):
            ratio = elem["cpu_cores_effective"] / elem["cpu_cores"]
            cpu_ram_effective = elem["cpu_ram"] * ratio
            list_out[idx]["cpu_ram_effective"] = cpu_ram_effective
        list_out.sort(key=lambda x: x["cpu_ram_effective"], reverse=True)
        list_out = list(
            filter(lambda x: x["cpu_ram_effective"] > 30000, list_out)
        )  # larger than 30 GB
        print(type(list_out))
        for idx, elem in enumerate(list_out):
            print(
                f"Option {idx + 1}",
                elem["cpu_cores"],
                elem["cpu_cores_effective"],
                elem["cpu_ram"],
                elem["cpu_ram_effective"],
                elem["dph_total"],
            )

    if is_hardware and len(list_out) == 0:
        print("NO SERVER FOUND")
        sys.exit(1)

    out = run_command(
        f"./vast.py create instance {list_out[0]['id']} "
        f"{(HW_TEST_COMMAND if is_hardware else GPU_TEST_COMMAND)}"
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
        try:
            out_list = json.loads(out)
            if len(image) > 0:
                out_list = list(filter(lambda x: x["image_uuid"] == image, out_list))
            if len(out_list):
                print(out_list)
                print(out_list[0]["status_msg"])
                if ssh_port in (0, None):
                    ssh_host = out_list[0]["ssh_host"]
                    if isinstance(out_list[0]["ssh_port"], int):
                        ssh_port = out_list[0]["ssh_port"]
                if out_list[0]["actual_status"] == "running" and ssh_port != 0:
                    starting = False
            counter += 1
        except json.JSONDecodeError:
            print("raw output", out)

    return ssh_host, ssh_port


# GPU TESTS
# commands to execute the tests
# mv /venv/ /venv2 # because vast.ai already has venv
# source /venv2/bin/activate
# git clone https://github.com/joennlae/halutmatmul.git
# cd halutmatmul
# pytest -n4 -srPA src/python/test/test_kernel_gpu.py

# currently using 4 jobs in parallel

GPU_TEST_COMMANDS = 'cd /; mv /venv/ /venv2; source /venv2/bin/activate; \
      git clone https://github.com/joennlae/halutmatmul.git; \
      cd halutmatmul; pytest -n0 -srPA -k "gpu"; \
      echo "ERROR CODE: $?";'

# fix for QT5 https://stackoverflow.com/a/43122457/13050816
# cp -R /venv2/plugins/platforms/* /venv2/bin/platforms/;
HW_TEST_COMMANDS = """cd /; mv /venv/ /venv2; source /venv2/bin/activate;
      mkdir -p /venv2/bin/platforms/;
      cp -R /venv2/plugins/platforms/* /venv2/bin/platforms/;
      git clone https://github.com/joennlae/halutmatmul.git;
      cd halutmatmul; python hardware/util/vendor.py hardware/flow/openroad.vendor.hjson -v;
      cd hardware;
      NUM_M=2 NUM_DECODER_UNITS=1 fusesoc --cores-root=. run --target=openroad_asap7 halut:ip:halut_top;
      echo "ERROR CODE: $?";
      fusesoc --cores-root=. run --target=openroad_asap7_encoder_4 halut:ip:halut_top;
      echo "ERROR CODE: $?";
      fusesoc --cores-root=. run --target=openroad_asap7_decoder halut:ip:halut_top;
      echo "ERROR CODE: $?";
      cd build;
      mkdir flow_reports;
      mkdir -p flow_reports/halut_matmul;
      mkdir -p flow_reports/halut_encoder_4;
      mkdir -p flow_reports/halut_decoder;
      cp halut_ip_halut_top_0.1/openroad_asap7-openroad/metrics.html flow_reports/halut_matmul;
      cp halut_ip_halut_top_0.1/openroad_asap7-openroad/metrics.json flow_reports/halut_matmul;
      cp -R halut_ip_halut_top_0.1/openroad_asap7-openroad/reports/ flow_reports/halut_matmul;
      cp -R halut_ip_halut_top_0.1/openroad_asap7-openroad/logs/ flow_reports/halut_matmul;
      cp halut_ip_halut_top_0.1/openroad_asap7_encoder_4-openroad/metrics.html flow_reports/halut_encoder_4;
      cp halut_ip_halut_top_0.1/openroad_asap7_encoder_4-openroad/metrics.json flow_reports/halut_encoder_4;
      cp -R halut_ip_halut_top_0.1/openroad_asap7_encoder_4-openroad/reports/ flow_reports/halut_encoder_4;
      cp -R halut_ip_halut_top_0.1/openroad_asap7_encoder_4-openroad/logs/ flow_reports/halut_encoder_4;
      cp halut_ip_halut_top_0.1/openroad_asap7_decoder-openroad/metrics.html flow_reports/halut_decoder;
      cp halut_ip_halut_top_0.1/openroad_asap7_decoder-openroad/metrics.json flow_reports/halut_decoder;
      cp -R halut_ip_halut_top_0.1/openroad_asap7_decoder-openroad/reports/ flow_reports/halut_decoder;
      cp -R halut_ip_halut_top_0.1/openroad_asap7_decoder-openroad/logs/ flow_reports/halut_decoder;
      tar -cvf report.tar.gz flow_reports
      cp report.tar.gz /report.tar.gz
"""


def run_ssh_commands(
    ssh_host: str, ssh_port: int, debug: bool = False, is_hardware: bool = False
) -> int:

    commands = HW_TEST_COMMANDS if is_hardware else GPU_TEST_COMMANDS

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

    if is_hardware:
        out_scp = run_command(
            f"scp -o StrictHostKeyChecking=no {ssh_identity_str} "
            f"-P {ssh_port} root@{ssh_host}:/report.tar.gz .",
            print_all=True,
        )
        file_exists = os.path.exists("report.tar.gz")

    if not is_hardware:
        error_code = re.findall(r"(?<=ERROR CODE: )\d+", out)
        print("ERROR CODE: ", error_code)
        failures = re.findall(r"\d+ failed", out)
        return int(error_code[0]) + len(failures)
    else:
        error_code = re.findall(r"(?<=exited with an error: )\d+", out)
        print("ERROR CODE: ", error_code)
        if len(error_code) == 0:
            error_code.append(0)
        return int(error_code[0]) + 1 if not file_exists else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vast.ai helper")
    parser.add_argument("--cleanup", "-c", action="store_true", help="run only cleanup")
    parser.add_argument(
        "--debug", "-d", action="store_true", help="set ssh key offline"
    )
    parser.add_argument(
        "--hardware", "-hw", action="store_true", help="set is used for hardware tests"
    )
    parser.add_argument(
        "--image",
        "-i",
        default="",
        help="set image name to look after for cleanup",
        type=str,
    )
    args = parser.parse_args()

    print(args)
    if args.cleanup:
        cleanup(args.image)
    else:
        cleanup(args.image)
        ssh_host, ssh_port = startup(args.image, args.hardware)
        # ssh_host = "ssh4.vast.ai"
        # ssh_port = 11182
        sleep(60)
        error_code = run_ssh_commands(ssh_host, ssh_port, args.debug, args.hardware)
        cleanup(args.image)
        sys.exit(error_code)
