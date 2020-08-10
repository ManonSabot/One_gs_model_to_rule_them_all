try:
    from soil import water_potential
    from hydraulics import f, Weibull_params, k_regulate, hydraulics
    from fregulate import fwsoil, fwLWPpd, fLWP, phiLWP, fPLC, hydraulic_cost
    from fregulate import kcost, dcost_dpsi
    from canatm import vpsat, slope_vpsat, LH_water_vapour, psychometric
    from canatm import emissivity, net_radiation
    from leaf import conductances, leaf_temperature, leaf_energy_balance
    from leaf import calc_colim_Ci, calc_photosynthesis, rubisco_limit

except (ImportError, ModuleNotFoundError):
    from TractLSM.SPAC.soil import water_potential
    from TractLSM.SPAC.hydraulics import f, Weibull_params, k_regulate
    from TractLSM.SPAC.hydraulics import hydraulics
    from TractLSM.SPAC.fregulate import fwsoil, fwLWPpd, fLWP, phiLWP, fPLC
    from TractLSM.SPAC.fregulate import hydraulic_cost, kcost, dcost_dpsi
    from TractLSM.SPAC.canatm import vpsat, slope_vpsat, LH_water_vapour
    from TractLSM.SPAC.canatm import psychometric, emissivity, net_radiation
    from TractLSM.SPAC.leaf import conductances, leaf_temperature
    from TractLSM.SPAC.leaf import leaf_energy_balance, calc_colim_Ci
    from TractLSM.SPAC.leaf import calc_photosynthesis, rubisco_limit
