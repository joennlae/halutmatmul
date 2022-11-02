# pylint: disable=consider-using-dict-items, redefined-builtin, f-string-without-interpolation
from typing import Any, List

import pandas as pd

D_to_C = [4, 8, 16, 32, 64]
C = 32

data_frame = pd.read_csv("power_data.csv")

sorted_ = data_frame.sort_values(
    ["design_name", "power_type", "sim_clk"], ascending=[True, True, True]
)

data_frame = sorted_.groupby(["design_name", "power_type"]).first().reset_index()
# print(data_frame)


def add_technology(input: Any) -> str:
    if input["tech"] == 7:
        return "ASAP7"
    elif input["tech"] == 22:
        return "GF22FDX"
    else:
        raise Exception(f"tech {input['tech']} not supported")


def add_vdd(input: Any) -> float:
    if input["tech"] == 7:
        if input["addon"] == "ssg":
            return 0.63
        else:
            return 0.70
    elif input["tech"] == 22:
        if input["addon"] == "nominal":
            return 0.80
        else:
            return 0.65
    else:
        raise Exception(f"tech {input['tech']} not supported")


def add_temp(input: Any) -> int:
    if input["tech"] == 7:
        if input["addon"] == "ssg":
            return 100
        else:
            return 25
    elif input["tech"] == 22:
        if input["addon"] == "nominal":
            return 25
        else:
            return 25
    else:
        raise Exception(f"tech {input['tech']} not supported")


def add_corner(input: Any) -> str:
    if input["tech"] == 7:
        if input["addon"] == "ssg":
            return "SSG"
        else:
            return "TT"
    elif input["tech"] == 22:
        return "TT"
    else:
        raise Exception(f"tech {input['tech']} not supported")


def add_track(input: Any) -> str:
    if input["tech"] == 7:
        return "7.5T"
    elif input["tech"] == 22:
        return "8T"
    else:
        raise Exception(f"tech {input['tech']} not supported")


def add_leakage_percentage(input: Any) -> float:
    return input["power_cell_leakage"] / input["power_total"] * 100


def add_switching_percentage(input: Any) -> float:
    return input["power_net_switching"] / input["power_total"] * 100


def add_internal_percentage(input: Any) -> float:
    return input["power_cell_internal"] / input["power_total"] * 100


def add_clock_percentage(input: Any) -> float:
    return input["power_clock_network"] / input["power_total"] * 100


def add_register_percentage(input: Any) -> float:
    return input["power_register"] / input["power_total"] * 100


def add_comb_percentage(input: Any) -> float:
    return input["power_combinational"] / input["power_total"] * 100


data_frame["corner"] = data_frame.apply(add_corner, axis=1)
data_frame["technology"] = data_frame.apply(add_technology, axis=1)
data_frame["temp"] = data_frame.apply(add_temp, axis=1)
data_frame["vdd"] = data_frame.apply(add_vdd, axis=1)

data_frame["freq_hz"] = 1e9 / (data_frame["sim_clk"] / 1000)
data_frame["freq_mhz"] = 1e9 / (data_frame["sim_clk"] / 1000) * 1e-6

data_frame["track"] = data_frame.apply(add_track, axis=1)

data_frame["area_mm2"] = data_frame["area"] * 1e-6

data_frame["power_leakage_percentage"] = data_frame.apply(
    add_leakage_percentage, axis=1
)
data_frame["power_switching_percentage"] = data_frame.apply(
    add_switching_percentage, axis=1
)
data_frame["power_internal_percentage"] = data_frame.apply(
    add_internal_percentage, axis=1
)

data_frame["power_clock_percentage"] = data_frame.apply(add_clock_percentage, axis=1)
data_frame["power_register_percentage"] = data_frame.apply(
    add_register_percentage, axis=1
)
data_frame["power_comb_percentage"] = data_frame.apply(add_comb_percentage, axis=1)
# remove ssg so far
data_frame = data_frame[data_frame["corner"] == "TT"]

data_frame = (
    data_frame.groupby(["C", "decoders", "power_type", "vth", "tech", "vdd", "corner"])
    .first()
    .reset_index()
)
data_frame = data_frame.sort_values(["C", "tech", "vdd"], ascending=[True, False, True])
data_frame = data_frame[data_frame["power_type"] == "execute"]

for proportion in D_to_C:
    data_frame["macs_per_cycle_1_" + str(proportion)] = (
        data_frame.loc[:, "decoders"].values * proportion
    )

    data_frame.loc[:, "macs_per_s_1_" + str(proportion)] = (
        data_frame["macs_per_cycle_1_" + str(proportion)].values
        * data_frame["freq_hz"].values
        * 1e-12
    )
    data_frame.loc[:, "gmacs_per_s_1_" + str(proportion)] = (
        data_frame["macs_per_cycle_1_" + str(proportion)].values
        * data_frame["freq_hz"].values
        * 1e-9
    )
    data_frame["Energy Efficiency 1:" + str(proportion)] = data_frame[
        "macs_per_s_1_" + str(proportion)
    ] / (data_frame["power_total"])
    data_frame["Area Efficiency 1:" + str(proportion)] = data_frame[
        "macs_per_s_1_" + str(proportion)
    ] / (data_frame["area_mm2"])

data_frame = data_frame.sort_values(
    ["C", "decoders", "vth"], ascending=[True, True, True]
)
data_frame = data_frame[
    (data_frame["technology"] == "GF22FDX")
    & (data_frame["vth"] == "L")
    & (data_frame["C"] == 32)
    & (data_frame["vdd"] == 0.65)
]
print("TEST", data_frame)
df_table = data_frame[
    [
        "technology",
        "track",
        "vth",
        f"vdd",
        f"temp",
        "corner",
        "decoders",
        "C",
        "freq_mhz",
        "area_mm2",
        f"Area Efficiency 1:8",
        f"Area Efficiency 1:16",
        f"Area Efficiency 1:32",
        f"Area Efficiency 1:64",
        f"Energy Efficiency 1:8",
        f"Energy Efficiency 1:16",
        f"Energy Efficiency 1:32",
        f"Energy Efficiency 1:64",
        "gmacs_per_s_1_8",
        "gmacs_per_s_1_16",
        "gmacs_per_s_1_32",
        "gmacs_per_s_1_64",
        "power_switching_percentage",
        "power_leakage_percentage",
        "power_internal_percentage",
        "power_clock_percentage",
        "power_register_percentage",
        "power_comb_percentage",
    ]
].copy()

print(df_table.head(20))

cidx = pd.MultiIndex.from_arrays(
    [
        [
            "Tech",
            "Track",
            "Vth",
            "PVT Corner",
            "PVT Corner",
            "PVT Corner",
            "Decs",
            "C",
            "Freq",
            "Area",
            "Area Eff. [TMAC/s mm2]",
            "Area Eff. [TMAC/s mm2]",
            "Area Eff. [TMAC/s mm2]",
            "Area Eff. [TMAC/s mm2]",
            "Energy Eff. [TMAC/s W]",
            "Energy Eff. [TMAC/s W]",
            "Energy Eff. [TMAC/s W]",
            "Energy Eff. [TMAC/s W]",
            "Throughput [GMAC/s]",
            "Throughput [GMAC/s]",
            "Throughput [GMAC/s]",
            "Throughput [GMAC/s]",
            "Power [%]",
            "Power [%]",
            "Power [%]",
            "Power Parts [%]",
            "Power Parts [%]",
            "Power Parts [%]",
        ],
        [
            "",
            "",
            "",
            "Vdd",
            "T",
            "",
            "",
            "",
            "",
            "",
            f"[C:D]",
            f"[C:D]",
            f"[C:D]",
            f"[C:D]",
            f"[C:D]",
            f"[C:D]",
            f"[C:D]",
            f"[C:D]",
            f"[C:D]",
            f"[C:D]",
            f"[C:D]",
            f"[C:D]",
            "Dynamic",
            "Static",
            "Static",
            "",
            "",
            "",
        ],
        [
            "",
            "",
            "",
            "[V]",
            "[C]",
            "",
            "",
            "",
            "[Mhz]",
            "[mm2]",
            "1:8",
            "1:16",
            "1:32",
            "1:64",
            "1:8",
            "1:16",
            "1:32",
            "1:64",
            "1:8",
            "1:16",
            "1:32",
            "1:64",
            "Switching",
            "Leaking",
            "Internal",
            "Clock",
            "Register",
            "Comb",
        ],
    ]
)
df_table.columns = cidx
print(df_table.head(5))

idx = pd.IndexSlice
styler = df_table.style
styler.format(subset="Freq", precision=0).format(
    precision=1, subset="Energy Eff. [TMAC/s W]"
).format(precision=2, subset=(idx[:], idx[:, :, "[V]"])).format(
    precision=0, subset=(idx[:], idx[:, :, "[C]"])
).format(
    subset="Area", precision=3
).format(
    subset="Area Eff. [TMAC/s mm2]", precision=2
).format(
    subset="Throughput [GMAC/s]", precision=0
).format(
    precision=1, subset="Power [%]"
).format(
    precision=1, subset="Power Parts [%]"
).format_index(
    escape="latex", axis=1
).format_index(
    escape="latex", axis=0
).hide(
    level=0, axis=0
)
# .format(
#    subset="Power [mW]", precision=1
# )

# styler.background_gradient(
#     axis=None,
#     cmap="Greens",
#     subset=["Area Eff. [GMAC/s mm2]"],
# )

styler.background_gradient(
    axis=None,
    cmap="Greens",
    subset=["Energy Eff. [TMAC/s W]"],
)

styler.background_gradient(
    axis=None,
    cmap="Greens",
    subset=["Throughput [GMAC/s]"],
)

styler.background_gradient(
    axis=None,
    cmap="Reds",
    vmax=200,
    subset=["Power [%]"],
)

styler.background_gradient(
    axis=None,
    cmap="Blues",
    vmax=200,
    subset=["Power Parts [%]"],
)
styler.background_gradient(
    axis=None,
    cmap="Reds",
    vmax=30,
    subset=["Area Eff. [TMAC/s mm2]"],
)

styler.to_latex(
    f"table_tt.tex",
    clines="skip-last;data",
    convert_css=True,
    position_float="centering",
    multicol_align="|c|",
    hrules=True,
    # float_format="%.2f",
)

# create subunit information
subunit_name = [
    "encoder",
    "encoder_4",
    "decoder",
    "lut",
    "fp_adder",
]
subunit_table_names = ["Encoder", "Encoder4x", "Decoder", "LUT", "FP Adder"]

subunit_data = []

#  0.199 µm2
for idx, sub in enumerate(subunit_name):
    row = {}
    row["unit"] = subunit_table_names[idx]
    row["area"] = data_frame[f"area_{sub}"].iloc[0]
    row["kge"] = data_frame[f"area_{sub}"].iloc[0] / 0.199 * 1e-3
    row["instance_count"] = data_frame[f"instance_count_{sub}"].iloc[0]
    row["total_power"] = data_frame[f"power_{sub}_total_power"].iloc[0]
    row["int_power"] = data_frame[f"power_{sub}_int_power"].iloc[0]
    row["leak_power"] = data_frame[f"power_{sub}_leak_power"].iloc[0]
    row["switch_power"] = data_frame[f"power_{sub}_switch_power"].iloc[0]
    row["total_power_mw"] = data_frame[f"power_{sub}_total_power"].iloc[0] * 1e3
    row["energy_per_op"] = (
        data_frame[f"power_{sub}_total_power"].iloc[0] / data_frame[f"freq_hz"].iloc[0]
    ) * 1e12
    print(row)
    subunit_data.append(row)

subunit_df = pd.DataFrame(subunit_data)


def add_leak_percentage(input: Any) -> float:
    return input["leak_power"] / input["total_power"] * 100


def add_switch_percentage(input: Any) -> float:
    return input["switch_power"] / input["total_power"] * 100


def add_int_percentage(input: Any) -> float:
    return input["int_power"] / input["total_power"] * 100


subunit_df["power_leakage_percentage"] = subunit_df.apply(add_leak_percentage, axis=1)
subunit_df["power_switching_percentage"] = subunit_df.apply(
    add_switch_percentage, axis=1
)
subunit_df["power_internal_percentage"] = subunit_df.apply(add_int_percentage, axis=1)

print(subunit_df)
df_table2 = subunit_df[
    [
        "unit",
        "area",
        "kge",
        "total_power_mw",
        "power_switching_percentage",
        "power_leakage_percentage",
        "power_internal_percentage",
        "energy_per_op",
    ]
].copy()

print(df_table2.head(20))

cidx = pd.MultiIndex.from_arrays(
    [
        [
            "Unit",
            "Area",
            "GE",
            "Power",
            "Power [%]",
            "Power [%]",
            "Power [%]",
            "Energy/Op",
        ],
        [
            "",
            "",
            "",
            "",
            "Dynamic",
            "Static",
            "Static",
            "",
        ],
        ["", "[um2]", "[kGE]", "[mW]", "Switching", "Leaking", "Internal", "[pJ]"],
    ]
)
df_table2.columns = cidx

idx = pd.IndexSlice
styler = df_table2.style
styler.format(subset="Area", precision=0).format(precision=0, subset="GE").format(
    precision=1, subset="Power [%]"
).format(precision=2, subset="Power").format_index(escape="latex", axis=1).format(
    precision=2, subset="Energy/Op"
).format_index(
    escape="latex", axis=0
).hide(
    level=0, axis=0
)
styler.background_gradient(
    axis=None,
    cmap="Reds",
    vmax=200,
    subset=["Power [%]"],
)

styler.to_latex(
    f"table_tt_subunits.tex",
    clines="skip-last;data",
    convert_css=True,
    position_float="centering",
    multicol_align="|c|",
    hrules=True,
    # float_format="%.2f",
)
