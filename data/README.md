# Data

Raw CSV files are **not** committed (they are listed in `.gitignore` as they are too large).

## Source
LBNL Fault Detection and Diagnostics Data Sets: Single Duct Air Handling Unit
(Granderson et al., 2022). Download page given in the dataset documentation:
https://fdd.hyperarts.com/dataset/simulated-sd-ahu/



## Files used in this project
Save these into this `data/` folder:

| File | Case |
|---|---|
| `AHU_annual.csv` | Fault-free |
| `oa_bias_2_annual.csv`, `oa_bias_4_annual.csv` | Outdoor air temperature sensor +2, +4 degC |
| `oa_bias_-2_annual.csv`, `oa_bias_-4_annual.csv` | Outdoor air temperature sensor -2, -4 degC |

Data are 1-minute readings for one simulated year (Chicago TMY weather).
Original units: degF, CFM, in. w.g. Loading code converts to SI.

## Sample data
Put a small excerpt (about a week) in `data/sample/` so the tests and README
example run without the full download. Check the dataset licence allows this.
