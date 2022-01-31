try:
    from Utils.constants_and_conversions import ConvertUnits
    from Utils.constants_and_conversions import Constants

except (ImportError, ModuleNotFoundError):
    from TractLSM.Utils.constants_and_conversions import ConvertUnits
    from TractLSM.Utils.constants_and_conversions import Constants

# initialization of the unit and constant "libraries"
conv = ConvertUnits()  # unit converter
cst = Constants()  # general constants

# rename the weather data module (easier import) & save default inputs
try:
    from Utils import build_final_forcings
    from Utils.default_params import default_params

except (ImportError, ModuleNotFoundError):
    from TractLSM.Utils import build_final_forcings
    from TractLSM.Utils.default_params import default_params

dparams = default_params()  # default parameter class


class InForcings(object):

    def __init__(self):

        self.run = build_final_forcings.main  # generate the inputs
        self.defparams = dparams  # pre-defined params

        return


# other modules
try:
    from run_leaf_level import run as hrun

except (ImportError, ModuleNotFoundError):
    from TractLSM.run_leaf_level import run as hrun
