from typing import Any, Literal
import pandas as pd

C = 16

data_frame = pd.read_csv("power_data.csv")

data_frame = data_frame[data_frame["power_type"] == "matmul"]
print(data_frame)


# create subunit information
subunits = [
    "encoder",
    "gen_decoderX_units",
    "gen_encoder_units",
    "gen_decoders",
    "gen_int_accumulation_int_adder",
]

subunit_table_names = [
    "Encoder 4x",
    "Decoder 8x",
    "Encoder",
    "Decoder",
    "INT8/INT24 adder",
]
areas = [
    3190,
    12650,
    790,
    1570,
    35,
]

subunit_data = []

#  0.199 µm2
# 14 --> 0.48 * 0.252 = 0.12096
for idx, sub in enumerate(subunits):
    row = {}
    row["unit"] = subunit_table_names[idx]
    row["area"] = areas[idx]
    row["kge"] = areas[idx] / 0.12096 * 1e-3
    row["total_power"] = data_frame[f"power_{sub}_total_power"].iloc[0]
    row["int_power"] = data_frame[f"power_{sub}_internal_power"].iloc[0]
    row["leak_power"] = data_frame[f"power_{sub}_leaking_power"].iloc[0]
    row["switch_power"] = data_frame[f"power_{sub}_switching_power"].iloc[0]
    row["total_power_mw"] = data_frame[f"power_{sub}_total_power"].iloc[0] * 1e3
    row["energy_per_op"] = (
        data_frame[f"power_{sub}_total_power"].iloc[0]
        / (1e9 / data_frame["sim_clk_period"].iloc[0])
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

# reorder rows
df_table2 = df_table2.reindex([2, 0, 3, 1, 4])

print(df_table2.head(20))

cidx = pd.MultiIndex.from_arrays(
    [
        [
            "Unit",
            "Area",
            "GE",
            "Power",
            "Power",
            "Power",
            "Power",
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
# .format( precision=1, subset="Power [%]"")
styler.format(subset="Area", precision=0).format(precision=1, subset="GE").format(
    precision=2, subset="Power"
).format_index(escape="latex", axis=1).format(
    precision=2, subset="Energy/Op"
).format_index(
    escape="latex", axis=0
).hide(
    level=0, axis=0
)
# pylint: disable=anomalous-backslash-in-string
styler.format("{:.1f} \%", subset=(idx[:], idx[:, :, "Switching"]))
styler.format("{:.1f} \%", subset=(idx[:], idx[:, :, "Leaking"]))
styler.format("{:.1f} \%", subset=(idx[:], idx[:, :, "Internal"]))
# styler.background_gradient(
#     axis=None,
#     cmap="Reds",
#     vmax=200,
#     subset=["Power [%]"],
# )

styler.background_gradient(
    axis=None,
    cmap="Reds",
    # vmax=100,
    subset=(idx[:], idx[:, :, "[pJ]"]),
)

styler.to_latex(
    "table_tt_subunits.tex",
    clines="skip-last;data",
    convert_css=True,
    position_float="centering",
    multicol_align="|c|",
    hrules=True,
    # float_format="%.2f",
)
