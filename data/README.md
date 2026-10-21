# Data

Raw CSV files are **not** committed (they are large and are listed in `.gitignore`).

## Source
LBNL Fault Detection and Diagnostics Data Sets: Single Duct Air Handling Unit
(Granderson et al., 2022). Download page given in the dataset documentation:
https://fdd.hyperarts.com/dataset/simulated-sd-ahu/

Check the DOI on the LBNL site before citing. The documentation PDF shows two
slightly different DOIs (`10.25984/188132` and `10.25984/1881321`).

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
