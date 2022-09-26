# pylint: disable=anomalous-backslash-in-string, consider-using-with
import argparse
import json
from json import decoder
import os
import re
import subprocess
from pathlib import Path
import shutil
from sys import prefix
import tempfile
from typing import Any, List

# BASE_PATH = "/scratch2/janniss/outputs"
BASE_PATH = "/usr/scratch2/vilan2/janniss/outputs"
GIT_BASE_PATH = "/home/msc22f5/Documents/halutmatmul"


def run_command(cmd: str, print_all: bool = True) -> None:
    # pylint: disable=subprocess-run-check
    process = subprocess.Popen(
        [cmd], stdout=subprocess.PIPE, shell=True, executable="/bin/bash"
    )
    output_str_list = []
    total_output_str = ""
    for line in iter(process.stdout.readline, b""):  # type: ignore[union-attr]
        decoded = line.decode("utf-8")
        output_str_list.append(decoded)
        total_output_str += decoded + "\n"
        if print_all:
            print(decoded, end="")


def clone_git_repo(repo_url: str, clone_dir: str, rev: str = "main") -> str:
    print("Cloning upstream repository %s @ %s", repo_url, rev)

    # Clone the whole repository
    cmd = ["git", "clone", "--no-single-branch"]
    cmd += [repo_url, str(clone_dir)]
    subprocess.run(cmd, check=True)

    # Check out exactly the revision requested
    cmd = ["git", "-C", str(clone_dir), "checkout", "--force", rev]
    subprocess.run(cmd, check=True)

    # Get revision information
    cmd = ["git", "-C", str(clone_dir), "rev-parse", "HEAD"]
    rev = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        universal_newlines=True,
    ).stdout.strip()
    print("Cloned at revision %s", rev)
    return rev


# TODO left
# MATRIX
# C=[16,32,64]
# FREQ=[1600, 1700] ASAP7 | FREQ=[1700, 1800]
# Vth=[L, SL]
# 8_4, 32_16

C = [16, 32, 64]
size_option = ["8_4"]
vth = ["L", "SL"]
clk_period_asap7_sl = [1400, 1500]
clk_period_asap7_l = [1500, 1600]
clk_period_gf22_sl = [1700, 1800]
clk_period_gf22_l = [2900, 3000]
# only TT at the moment

MFLOWGEN_PATH = "/usr/scratch2/pisoc8/janniss/mflowgen_build/mflowgen-iis"
github_repo_url = "git@github.com:joennlae/mflowgen-iis.git"

# pylint: disable=too-many-nested-blocks, use-maxsplit-arg, unused-variable
def make_designs(tech: str = "asap7", add_on: str = "") -> None:
    if tech == "asap7":
        clk_period = clk_period_asap7_sl
        output_folder = BASE_PATH + "/"  # + "/designs/"
        prefix = "7"
        check_path = "openroad_asap7-openroad/results/asap7/halut_matmul/base/6_final.v"
    elif tech == "gf22":
        clk_period = clk_period_gf22_sl
        output_folder = BASE_PATH + "/"  # MFLOWGEN_PATH + "/outputs/"
        prefix = "22"
        check_path = "6-cadence-innovus-place-route/results/halut_matmul.vcs.v"
    else:
        raise Exception(f"tech = {tech} not supported")
    for C_ in C:
        for v in vth:
            for s in size_option:
                decoders = s.split("_")[0]
                decoders_per_subunit = s.split("_")[1]
                if tech == "gf22" and v == "L":
                    clk_period = clk_period_gf22_l
                elif tech == "gf22" and v == "SL":
                    clk_period = clk_period_gf22_sl
                if tech == "asap7" and v == "L":
                    clk_period = clk_period_asap7_l
                elif tech == "asap7" and v == "SL":
                    clk_period = clk_period_asap7_sl
                if add_on == "ssg_":
                    clk_period[0] = clk_period[0] + 700
                    clk_period[1] = clk_period[1] + 700
                for clk in clk_period:
                    # check if design exists
                    folder_name = (
                        f"{prefix}_{decoders}_{decoders_per_subunit}"
                        f"_{C_}_{clk}_{'TT'}_{add_on}{v}"
                    )
                    output_path = output_folder + folder_name
                    print(
                        f"Starting design {tech}: C: {C_}, size: {s}, clk: {clk}, vth: {v} \n"
                        f"folder: {output_path}"
                    )
                    if os.path.isdir(output_path):
                        # path_to_check = Path(output_path + "/" + check_path)
                        # if path_to_check.exists():
                        print("Already done -> skipping...")
                        continue

                    # run command
                    if tech == "asap7":
                        vth_string = "ASAP7_USELVT=1" if v == "L" else "ASAP7_USESLVT=1"
                        # set clock period
                        run_command(
                            f"sed -i '/set clk_period/c\set clk_period {clk}' "
                            f"{GIT_BASE_PATH}/hardware/flow/asap7/constraint.sdc"
                        )

                        run_command(
                            f"{vth_string} NUM_C={C_} "
                            f"NUM_M={decoders} NUM_DECODER_UNITS={decoders_per_subunit} "
                            f"CLK_PERIOD={clk} "
                            f"fusesoc --cores-root=. run --target=openroad_asap7 "
                            f"--build-root={output_path} halut:ip:halut_top"
                        )
                    elif tech == "gf22":
                        vth_string = "VTH=L" if v == "L" else "VTH=SL"

                        # export design.v
                        run_command(
                            f"NUM_C={C_} NUM_M={decoders} "
                            f"NUM_DECODER_UNITS={decoders_per_subunit} fusesoc --cores-root=. "
                            f"run --target=export_design halut:ip:halut_top"
                        )
                        with tempfile.TemporaryDirectory() as tmpdirname:
                            print("created temporary directory", tmpdirname)
                            clone_git_repo(
                                github_repo_url, tmpdirname, rev="iis-version"
                            )
                            tmp_path = f"{tmpdirname}"
                            shutil.rmtree(output_path, ignore_errors=True)
                            Path(output_path).mkdir(parents=True, exist_ok=True)

                            # run_command(f"mkdir -p {tmp_path}")
                            # run_command(f"cp -R {MFLOWGEN_PATH} {tmp_path}")
                            # mv design.v
                            run_command(
                                f"mv design.v {tmp_path}/designs/halut_matmul/rtl/outputs/"
                            )
                            # update clk_period
                            run_command(
                                # pylint: disable=line-too-long
                                f"sed -i \"/    'clock_period'   : /c\    'clock_period'   : {clk},\" "
                                f" {tmp_path}/designs/halut_matmul/construct-commercial.py"
                            )

                            # run flow
                            run_command(
                                f"cd {tmp_path}; export {vth_string}; ls -a; source iis.sh; "
                                f"bash gf_22_adk_setup.sh; "
                                f"python3 -m venv venv; source venv/bin/activate; source iis.sh; "
                                f"pip install -e .; mflowgen --info; cd {output_path}; "
                                f"mflowgen run --design {tmp_path}/designs/halut_matmul; "
                                f"make;"
                            )


clean_sim_path = BASE_PATH + "/clean_sim/"


def prepare_neat_folders_sim() -> None:
    folders = list(os.listdir(BASE_PATH))
    folders = sorted(folders)
    folders = list(filter(lambda x: x.startswith(("7_", "22_")), folders))

    for des in folders:
        design_name = des
        design_path = BASE_PATH + "/" + design_name
        if des.startswith("7_"):
            tech = "asap7"
        elif des.startswith("22_"):
            tech = "gf22"
        else:
            raise Exception(f"tech not supported: {des}")
        design_clean_path = clean_sim_path + design_name
        if os.path.isdir(design_clean_path):
            print("Already done -> skipping...")
            continue
        Path(design_clean_path).mkdir(parents=True, exist_ok=True)
        design_name_splitted = design_name.split("_")
        decoders = int(design_name_splitted[1])
        decoders_per_subunit = int(design_name_splitted[2])
        num_c = int(design_name_splitted[3])
        if tech == "asap7":
            if not os.path.isfile(
                f"{design_path}/openroad_asap7-openroad/metrics.json"
            ):
                print(
                    f"WARNING: {design_path}/openroad_asap7-openroad/metrics.json doesn't exists"
                )
                print("skipping...")
                continue
            with open(f"{design_path}/openroad_asap7-openroad/metrics.json") as f:
                data = json.load(f)
                clk_period = data[0]["constraints__clocks__details"][0].split(" ")[1]
            with open(
                f"{design_path}/openroad_asap7-openroad/logs/asap7/halut_matmul/base/6_report.json"
            ) as f:
                data = json.load(f)
                wns = data["finish__timing__setup__ws"]
            clk_period = float(clk_period)
            wns = float(wns)
            # for simulation
            shutil.copy(
                f"{design_path}/openroad_asap7-openroad/results/asap7/halut_matmul/base/6_final.v",
                f"{design_clean_path}/design_with_buffers.v",
            )
            shutil.copy(
                f"{design_path}/openroad_asap7-openroad/results/asap7/halut_matmul/base/6_final.v",
                f"{design_clean_path}/design.v",
            )
            # remove filler cells
            for cell in [
                "TAPCELL_ASAP7_75t_R",
                "TAPCELL_ASAP7_75t_L",
                "TAPCELL_ASAP7_75t_SL",
                "FILLERxp5_ASAP7_75t_R",
                "FILLERxp5_ASAP7_75t_L",
                "FILLERxp5_ASAP7_75t_SL",
            ]:
                run_command(
                    f'cd {design_clean_path}; grep -v "{cell}" design.v > design_1.v;'
                    f"mv design_1.v design.v"
                )
        if tech == "gf22":
            with open(f"{design_path}/.mflowgen/0-constraints/mflowgen-run") as f:
                infos = re.findall(r"export clock_period=(\d+)", f.read())
                print(infos)
                clk_period = float(infos[0])

            with open(
                f"{design_path}/6-cadence-innovus-place-route/reports/signoff.summary"
            ) as f:
                infos = re.findall(r"(?<=WNS \(ns\):\| ).\d\.\d+", f.read())
                print(infos)
                wns = float(infos[0]) * 1000  # to get to ps

            shutil.copy(
                f"{design_path}/6-cadence-innovus-place-route/outputs/design.vcs.v",
                f"{design_clean_path}/design.v",
            )

        sim_info = {
            "clk_period": clk_period,
            "wns": wns,
            "decoders": decoders,
            "decoders_per_subunit": decoders_per_subunit,
            "C": num_c,
        }
        with open(f"{design_clean_path}/sim_info.json", "w") as outfile:
            json.dump(sim_info, outfile, indent=2)

        dump_vcd_code = (
            '`ifdef COCOTB_SIM\\n  initial begin\\n    $dumpfile("run.vcd");\\n    '
            "$dumpvars(0, halut_matmul);\\n    #1;\\n  end\\n`endif"
        )
        run_command(
            f"set -x; cd {design_clean_path}; sed "
            f"'/^module halut_matmul /,/);/!b;/);/a\{dump_vcd_code}'"
            f" design.v > design_changed.v; mv design_changed.v design.v"
        )


def run_power() -> None:
    sim_outputs = BASE_PATH + "/simulations/"
    folders = list(os.listdir(sim_outputs))
    folders = sorted(folders)

    for folder in folders:
        folder_name_splitted = folder.split("-")
        design_name = folder_name_splitted[0]
        # sim_clk_period = int(folder_name_splitted[1])
        infos = design_name.split("_")
        tech = infos[0]
        # decoders = int(infos[1])
        subunits_per_decoders = int(infos[2])
        # num_c = int(infos[3])
        vth = infos[-1]

        if tech == "7":
            sim_out_folder_name = "asap7-cocotb"
        elif tech == "22":
            sim_out_folder_name = "gf22-cocotb"
        else:
            raise Exception(f"tech no supported: {tech}")

        # check if simulation finished
        if os.path.exists(
            sim_outputs + "/" + folder + "/" + sim_out_folder_name + "/results.xml"
        ):
            for type_ in ["write", "execute"]:
                power_output_folder = f"{BASE_PATH}/power/{folder}-{type_}"
                if os.path.exists(power_output_folder):
                    print("already done -> skipping ...")
                    continue
                Path(power_output_folder).mkdir(parents=True, exist_ok=True)
                # create base folder structure
                run_command(
                    f"cd {power_output_folder}; "
                    f"mkdir logs; mkdir reports; mkdir outputs; mkdir -p inputs/adk;"
                )
                if tech == "7":
                    run_command(
                        f"ln -s {BASE_PATH}/{design_name}/openroad_asap7-openroad/"
                        f"results/asap7/halut_matmul/base/6_final.v "
                        f"{power_output_folder}/inputs/design.v;"
                        f"ln -s {BASE_PATH}/{design_name}/openroad_asap7-openroad/"
                        f"results/asap7/halut_matmul/base/6_final.spef "
                        f"{power_output_folder}/inputs/design.spef;"
                        f"ln -s {BASE_PATH}/{design_name}/openroad_asap7-openroad/"
                        f"results/asap7/halut_matmul/base/6_final.sdc "
                        f"{power_output_folder}/inputs/design.sdc;"
                    )
                    # uncomment current_design in sdc
                    run_command(
                        f"sed -i '/current_design/c\# current_design' "
                        f"{power_output_folder}/inputs/design.sdc"
                    )
                    # add libs
                    run_command(
                        f"ln -s {GIT_BASE_PATH}/hardware/pdks/asap7/stdcells_{vth}.db "
                        f"{power_output_folder}/inputs/adk/stdcells.db;"
                    )

                elif tech == "22":
                    run_command(
                        f"ln -s {BASE_PATH}/{design_name}/"
                        f"6-cadence-innovus-place-route/outputs/design.vcs.v "
                        f"{power_output_folder}/inputs/design.v;"
                        f"ln -s {BASE_PATH}/{design_name}/"
                        f"6-cadence-innovus-place-route/outputs/design.spef.gz "
                        f"{power_output_folder}/inputs/design.spef;"
                        f"ln -s {BASE_PATH}/{design_name}/"
                        f"6-cadence-innovus-place-route/outputs/design.pt.sdc "
                        f"{power_output_folder}/inputs/design.sdc;"
                    )
                    # add libs
                    run_command(
                        f"ln -s {GIT_BASE_PATH}/hardware/pdks/gf22/stdcells_{vth}.db "
                        f"{power_output_folder}/inputs/adk/stdcells.db;"
                    )
                    run_command(
                        f"ln -s {GIT_BASE_PATH}/hardware/pdks/gf22/iocells.db "
                        f"{power_output_folder}/inputs/adk/iocells.db;"
                    )
                    run_command(
                        f"ln -s {GIT_BASE_PATH}/hardware/pdks/gf22/iocells.lib "
                        f"{power_output_folder}/inputs/adk/iocells.lib;"
                    )

                # link vcd
                run_command(
                    f"ln -s {sim_outputs}/{folder}/{sim_out_folder_name}/run.vcd "
                    f"{power_output_folder}/inputs/run.vcd;"
                )
                run_command(
                    f"cp {GIT_BASE_PATH}/hardware/power/primetime_power.tcl "
                    f"{power_output_folder}/primetime_power.tcl;"
                )
                run_command(
                    f"cp {GIT_BASE_PATH}/hardware/power/pt_shell "
                    f"{power_output_folder}/pt_shell;"
                )
                # write end time
                with open(
                    f"{sim_outputs}/{folder}/{sim_out_folder_name}/sim_build/sim.log"
                ) as f:
                    subunit_str = (
                        f"{subunits_per_decoders - 1}\/{subunits_per_decoders}"
                    )
                    infos = re.findall(
                        r"(?<=finished writing " + subunit_str + r"\n\#\s{3})\d+.?\d+",
                        f.read(),
                    )
                    print(infos)
                    finished_writing = float(infos[0])
                if type_ == "write":
                    start_time_ns = 0.0
                    end_time_ns = finished_writing
                elif type_ == "execute":
                    start_time_ns = finished_writing
                    end_time_ns = "-1"  # type: ignore[assignment]
                else:
                    raise Exception(f"type {type_} not found")
                # run command
                run_command(
                    f"cd {power_output_folder}; export START_TIME_NS={start_time_ns}; "
                    f"export END_TIME_NS={end_time_ns}; "
                    f"./pt_shell -f primetime_power.tcl -output_log_file logs/pt.log || exit 1"
                )


def run_simulations() -> None:
    folders = list(os.listdir(clean_sim_path))
    folders = sorted(folders)
    print(folders)
    for des in folders:
        design_name = des
        if des.startswith("7_"):
            sim_target = "asap7"
        elif des.startswith("22_"):
            sim_target = "gf22"
        else:
            raise Exception(f"sim_target not supported: {des}")
        design_clean_path = clean_sim_path + design_name
        if os.path.isdir(design_clean_path):
            with open(f"{design_clean_path}/sim_info.json") as f:
                data = json.load(f)
                wns = data["wns"]
                clk_period = data["clk_period"]
                decoders = data["decoders"]
                decoders_per_subunit = data["decoders_per_subunit"]
                num_c = data["C"]

            sim_base_clk_period = int(clk_period + abs(wns))
            sim_base_clk_period = sim_base_clk_period + (
                sim_base_clk_period % 2
            )  # make it even (to work with ps sim scale)
            for offset in [0, 200]:
                sim_clk_period = sim_base_clk_period + offset
                sim_folder_name = f"{design_name}-{sim_clk_period}"

                output_path = BASE_PATH + "/simulations/" + sim_folder_name
                if os.path.isdir(output_path):
                    print("Already done -> skipping...")
                    continue
                # ready for simulation
                run_command(
                    f"cp {design_clean_path}/design.v {GIT_BASE_PATH}/hardware/sim/design_file/"
                )
                run_command(
                    f"NUM_C={num_c} "
                    f"NUM_M={decoders} NUM_DECODER_UNITS={decoders_per_subunit} "
                    f"CLK_PERIOD={sim_clk_period} "
                    f"fusesoc --cores-root=. run --target={sim_target} "
                    f"--build-root={output_path} halut:sim:halut_matmul"
                )


def cleanup() -> None:
    folders = list(os.listdir(BASE_PATH))
    folders = sorted(folders)
    folders = list(filter(lambda x: x.startswith(("7_")), folders))

    print(folders)
    for folder in folders:
        # check if it can be cleaned up
        if os.path.exists(
            f"{BASE_PATH}/{folder}/openroad_asap7-openroad/results/asap7/"
            "halut_matmul/base/4_cts.odb"
        ):
            print(f"cleaning up {BASE_PATH}/{folder}")
            run_command(
                f"cd {BASE_PATH}/{folder}/openroad_asap7-openroad/; "
                f"mkdir -p results_new/asap7/halut_matmul/base/; "
                f"mv results/asap7/halut_matmul/base/6_final.spef "
                f"results_new/asap7/halut_matmul/base/6_final.spef; "
                f"mv results/asap7/halut_matmul/base/6_final.v "
                f"results_new/asap7/halut_matmul/base/6_final.v; "
                f"mv results/asap7/halut_matmul/base/6_final.sdc "
                f"results_new/asap7/halut_matmul/base/6_final.sdc; "
                f"rm -rf results;"
                f"mv results_new results;"
            )

    # TODO: add sim cleanup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep run helper")
    parser.add_argument(
        "--prepsim", "-ps", action="store_true", help="prepare_simulator"
    )
    parser.add_argument("--runpower", "-rp", action="store_true", help="run_power")
    parser.add_argument("--runflow", "-rf", action="store_true", help="run_flow")

    parser.add_argument("--runsim", "-rs", action="store_true", help="run_sim")

    parser.add_argument("--cleanup", "-c", action="store_true", help="cleanup")
    parser.add_argument(
        "--tech",
        "-t",
        default="",
        help="set name of tech to use",
        type=str,
    )

    parser.add_argument(
        "--addon",
        "-a",
        default="",
        help="set addon for name",
        type=str,
    )

    args = parser.parse_args()

    print(args)

    if args.prepsim:
        prepare_neat_folders_sim()

    if args.runsim:
        run_simulations()

    if args.runpower:
        run_power()

    if args.runflow:
        if args.tech == "":
            make_designs()
        else:
            make_designs(tech=args.tech, add_on=args.addon)

    if args.cleanup:
        cleanup()
