import json
import tempfile
import subprocess

resulting_string = ""
preview_url = "https://htmlpreview.github.io/?"
github_repo_url = "https://github.com/joennlae/halutmatmul-openroad-reports/"
total_preview_url = preview_url + github_repo_url
base_raw_url = (
    "https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/"
)

units = ["halut_matmul", "halut_encoder_4", "halut_decoder"]
titles = ["Full Design", "Encoder", "Decoder"]


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


def format_ge(size: float) -> str:
    power = 1e3
    n = 0
    power_labels = {0: "", 1: "k", 2: "M", 3: "G", 4: "tera"}
    while size > power:
        size /= power
        n += 1
    return f"{size} {power_labels[n] + 'GE'}"


with tempfile.TemporaryDirectory() as tmpdirname:
    print("created temporary directory", tmpdirname)

    clone_git_repo(github_repo_url, tmpdirname)

    area_nangate45_list = []
    freq_nangate45_list = []
    ge_count_nangate45_list = []
    std_cells_count_nangate45_list = []
    util_nangate45_list = []
    tns_nangate45_list = []

    for idx, unit in enumerate(units):
        area_nangate45 = 0
        tns_nangate45 = 0
        clk_period_nangate45 = 0
        std_cells_count_nangate45 = 0
        utilization_nangate45 = 0

        with open(f"{tmpdirname}/latest/nangate45/{unit}/reports/metadata.json") as f:
            data = json.load(f)
            area_nangate45 = data["finish__design__instance__area"]
            tns_nangate45 = data["finish__timing__setup__tns"]
            clk_period_nangate45 = data["constraints__clocks__details"][0].split(" ")[1]
            std_cells_count_nangate45 = data["finish__design__instance__count__stdcell"]
            utilization_nangate45 = data["finish__design__instance__utilization"]

        freq_nangate45 = 1.0 / (float(clk_period_nangate45) * (1e-3))

        util_nangate45 = float(utilization_nangate45) * 100

        ge_nangate45 = 0.798000
        ge_count_nangate45 = int(area_nangate45 / ge_nangate45)

        area_nangate45_list.append(area_nangate45)
        freq_nangate45_list.append(freq_nangate45)
        ge_count_nangate45_list.append(ge_count_nangate45)
        std_cells_count_nangate45_list.append(std_cells_count_nangate45)
        util_nangate45_list.append(util_nangate45)
        tns_nangate45_list.append(tns_nangate45)

    table_string = f"""
| NanGate45      |  {units[0]}  |  {units[1]}  |  {units[2]}  |
| -------------  |  -------------  |
| Area [μm^2]    |  {area_nangate45_list[0]} |  {area_nangate45_list[1]}  |  {area_nangate45_list[2]}  | 
| Freq [Mhz]     |  {freq_nangate45_list[0]:.1f} |  {freq_nangate45_list[1]:.1f}  |  {freq_nangate45_list[2]:.1f}  |
| GE             |  {format_ge(ge_count_nangate45_list[0])} | {format_ge(ge_count_nangate45_list[1])}  |  {format_ge(ge_count_nangate45_list[2])}  |
| Std Cell [#]   |  {std_cells_count_nangate45_list[0]} |  {std_cells_count_nangate45_list[1]}  |  {std_cells_count_nangate45_list[2]}  |
| Voltage [V]    |   1.1             |
| Util [%]       |  {util_nangate45_list[0]:.1f} |  {util_nangate45_list[1]:.1f}  |  {util_nangate45_list[2]:.1f}  |
| TNS            |  {tns_nangate45_list[0]} |  {tns_nangate45_list[1]}  |  {tns_nangate45_list[2]}  |
| Clock Net      | <img src="{base_raw_url}main/latest/nangate45/{units[0]}/reports/final_clocks.webp" alt="Clock Net" width="150">  | <img src="{base_raw_url}main/latest/nangate45/{units[1]}/reports/final_clocks.webp" alt="Clock Net" width="150">  | <img src="{base_raw_url}main/latest/nangate45/{units[2]}/reports/final_clocks.webp" alt="Clock Net" width="150">  |
| Routing        | <img src="{base_raw_url}main/latest/nangate45/{units[0]}/reports/final_routing.webp" alt="Routing" width="150">   | <img src="{base_raw_url}main/latest/nangate45/{units[1]}/reports/final_routing.webp" alt="Routing" width="150">   | <img src="{base_raw_url}main/latest/nangate45/{units[2]}/reports/final_routing.webp" alt="Routing" width="150">   |
| GDS            | [GDS Download]({base_raw_url}main/latest/nangate45/{units[0]}/results/6_final.gds)  | [GDS Download]({base_raw_url}main/latest/nangate45/{units[1]}/results/6_final.gds)  | [GDS Download]({base_raw_url}main/latest/nangate45/{units[2]}/results/6_final.gds)  |
"""

    print(table_string)
