try:
    from models_2_fit import fres
    from calib_utils import soil_water, check_idealised_files
    from calib_utils import prep_training_N_target, extract_calib_info

except (ImportError, ModuleNotFoundError):
    from calibrations.models_2_fit import fres
    from calibrations.calib_utils import soil_water, check_idealised_files
    from calibrations.calib_utils import prep_training_N_target
    from calibrations.calib_utils import extract_calib_info
