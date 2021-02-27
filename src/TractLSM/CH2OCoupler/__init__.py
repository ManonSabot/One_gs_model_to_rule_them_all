try:
    from coupler_utils import calc_trans
    from coupler_utils import Ci_sup_dem
    from coupler_utils import A_trans
    from Medlyn import solve_std
    from Tuzet import Tuzet
    from SOX import supply_max
    from WUE_LWP import WUE_gs
    from ProfitMax import profit_psi
    from ProfitMax2 import profit_AE
    from CGain import Cgain_plc
    from CMax import Cmax_gs
    from LeastCost import least_cost
    from CAP import CAP
    from MES import MES

except (ImportError, ModuleNotFoundError):
    from TractLSM.CH2OCoupler.coupler_utils import calc_trans
    from TractLSM.CH2OCoupler.coupler_utils import Ci_sup_dem
    from TractLSM.CH2OCoupler.coupler_utils import A_trans
    from TractLSM.CH2OCoupler.Medlyn import solve_std
    from TractLSM.CH2OCoupler.Tuzet import Tuzet
    from TractLSM.CH2OCoupler.SOX import supply_max
    from TractLSM.CH2OCoupler.WUE_LWP import WUE_gs
    from TractLSM.CH2OCoupler.ProfitMax import profit_psi
    from TractLSM.CH2OCoupler.ProfitMax2 import profit_AE
    from TractLSM.CH2OCoupler.CGain import Cgain_plc
    from TractLSM.CH2OCoupler.CMax import Cmax_gs
    from TractLSM.CH2OCoupler.LeastCost import least_cost
    from TractLSM.CH2OCoupler.CAP import CAP
    from TractLSM.CH2OCoupler.MES import MES
