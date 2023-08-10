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
titles = ["Total Circuit (M=2)", "Encoder", "Decoder"]


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

    for idx, unit in enumerate(units):
        # area_asap7 = 0
        area_nangate45 = 0
        # tns_asap7 = 0
        tns_nangate45 = 0
        # clk_period_asap7 = 0
        clk_period_nangate45 = 0
        # std_cells_count_asap7 = 0
        std_cells_count_nangate45 = 0
        # utilization_asap7 = 0
        utilization_nangate45 = 0

        # with open(f"{tmpdirname}/latest/asap7/{unit}/metrics.json") as f:
        #     data = json.load(f)
        #     area_asap7 = data[0]["finish__design__instance__area"]
        #     tns_asap7 = data[0]["finish__timing__setup__tns"]
        #     clk_period_asap7 = data[0]["constraints__clocks__details"][0].split(" ")[1]
        #     std_cells_count_asap7 = data[0]["finish__design__instance__count__stdcell"]
        #     utilization_asap7 = data[0]["finish__design__instance__utilization"]

        with open(f"{tmpdirname}/latest/nangate45/{unit}/metrics.json") as f:
            data = json.load(f)
            area_nangate45 = data[0]["finish__design__instance__area"]
            tns_nangate45 = data[0]["finish__timing__setup__tns"]
            clk_period_nangate45 = data[0]["constraints__clocks__details"][0].split(
                " "
            )[1]
            std_cells_count_nangate45 = data[0][
                "finish__design__instance__count__stdcell"
            ]
            utilization_nangate45 = data[0]["finish__design__instance__utilization"]

        # freq_asap7 = 1.0 / (float(clk_period_asap7) * (1e-6))
        freq_nangate45 = 1.0 / (float(clk_period_nangate45) * (1e-3))

        # util_asap7 = float(utilization_asap7) * 100
        util_nangate45 = float(utilization_nangate45) * 100

        # GE
        # ASAP7 NAND2x1_ASAP7_75t_L -> 0.08748
        # ge_asap7 = 0.08748
        # NanGate45 NAND2_X1 -> 0.798000
        ge_nangate45 = 0.798000

        # ge_count_asap7 = int(area_asap7 / ge_asap7)
        ge_count_nangate45 = int(area_nangate45 / ge_nangate45)

        current_table = f"""
### {titles[idx]}
| {unit}         |  NanGate45      |
| -------------  |  -------------  |
| Area [μm^2]    | {area_nangate45} |
| Freq [Mhz]     |  {freq_nangate45:.1f} |
| GE             |  {format_ge(ge_count_nangate45)} |
| Std Cell [#]   |  {std_cells_count_nangate45} | 
| Voltage [V]    |   1.1             |
| Util [%]       |  {util_nangate45:.1f} | 
| TNS            |  {tns_nangate45} |
| Clock Net      | ![Clock_net]({base_raw_url}main/latest/nangate45/{unit}/reports/nangate45/{unit}/base/final_clocks.webp)  |
| Gallery        | [Gallery Viewer]({total_preview_url}blob/main/latest/nangate45/{unit}/reports/report-gallery-{unit}.html)  |
| Metrics        | [Metrics Viewer]({total_preview_url}blob/main/latest/nangate45/{unit}/metrics.html)  |
| Report         | [Report Viewer]({total_preview_url}blob/main/latest/nangate45/{unit}/reports/report-table.html)  |

"""
        resulting_string += current_table

    print(resulting_string)
