try:
    from general_utils import get_script_dir, get_main_dir, read_csv
    from calculate_solar_geometry import cos_zenith

except (ImportError, ModuleNotFoundError):
    from TractLSM.Utils.general_utils import get_script_dir, get_main_dir
    from TractLSM.Utils.general_utils import read_csv
    from TractLSM.Utils.calculate_solar_geometry import cos_zenith
