"""Unit conversions between the imperial units used in the LBNL dataset and SI.

The dataset logs temperatures in degF, airflow in CFM, duct pressure in
in. w.g. and areas are quoted in ft^2. All analysis in this project is done
in SI (degC, L/s, Pa, m^2). Convert once on loading and keep unit suffixes in
variable names (e.g. ``oa_flow_Ls``) so units are never mixed.
"""

CFM_TO_LS = 0.47194745      # 1 cubic foot per minute in litres per second
INWG_TO_PA = 249.08891      # 1 inch of water column (4 degC) in pascals
FT2_TO_M2 = 0.09290304      # 1 square foot in square metres


def f_to_c(temp_f):
    """Convert an absolute temperature from degF to degC.

    Args:
        temp_f: Temperature in degF (scalar, array or pandas Series).

    Returns:
        Temperature in degC, same type as the input.
    """
    return (temp_f - 32.0) * 5.0 / 9.0


def c_to_f(temp_c):
    """Convert an absolute temperature from degC to degF."""
    return temp_c * 9.0 / 5.0 + 32.0


def delta_f_to_delta_c(delta_f):
    """Convert a temperature *difference* from degF to degC (K).

    Differences have no 32 degree offset: a 3.6 degF difference is 2.0 degC.
    Use this for biases and temperature rises, not for absolute temperatures.
    """
    return delta_f * 5.0 / 9.0


def cfm_to_ls(flow_cfm):
    """Convert volumetric airflow from CFM to litres per second."""
    return flow_cfm * CFM_TO_LS


def ls_to_cfm(flow_ls):
    """Convert volumetric airflow from litres per second to CFM."""
    return flow_ls / CFM_TO_LS


def inwg_to_pa(pressure_inwg):
    """Convert pressure from inches of water column to pascals."""
    return pressure_inwg * INWG_TO_PA


def ft2_to_m2(area_ft2):
    """Convert area from square feet to square metres."""
    return area_ft2 * FT2_TO_M2
