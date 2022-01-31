# TractLSM-leaf (a Tractable simplified Land Surface Model applied to the leaf)

The **TractLSM** is adjusted to compare a suite of stomatal optimization
approaches and empirical approaches (the latter are all used in LSMs) at the
leaf-level.

The model is organised as follows:

```bash
TractLSM
├── run_leaf_level.py
├── run_utils.py
├── CH2OCoupler
│   ├── CAP.py
│   ├── CGain.py
│   ├── CMax.py
│   ├── LeastCost.py
│   ├── MES.py
│   ├── Medlyn.py
│   ├── ProfitMax.py
│   ├── ProfitMax2.py
│   ├── SOX.py
│   ├── Tuzet.py
│   ├── WUE_LWP.py
│   └── coupler_utils.py
├── SPAC
│   ├── canatm.py
│   ├── fregulate.py
│   ├── hydraulics.py
│   ├── leaf.py
│   └── soil.py
└── Utils
    ├── build_final_forcings.py
    ├── calculate_solar_geometry.py
    ├── constants_and_conversions.py
    ├── default_params.py
    ├── general_utils.py
    └── weather_generator.py
```

&nbsp;

`run_leaf_level.py` is where the forcings are read, the main routines called,
and the output writen. `run_utils.py` contains support functions for
these actions.

&nbsp;

The `CH2OCoupler/` is where you can find the various gas exchange coupling
routines / stomatal schemes, with `coupler_utils.py` containing support
functions for these schemes:
* `CAP.py` is derived/adapted from the work of
[Dewar et al. (2018)](https://doi.org/10.1111/nph.14848);
* `CGain.py` is derived/adapted from the work of
[Lu et al. (2020)](https://doi.org/10.1111/nph.16207);
* `CMax.py` is derived/adapted from the work of
[Wolf et al. (2016)](https://doi.org/10.1073/pnas.1615144113);
* `LeastCost.py` is derived/adapted from the work of
[Prentice et al. (2014)](https://doi.org/10.1111/ele.12211);
* `MES.py` is derived/adapted from the work of
[Dewar et al. (2018)](https://doi.org/10.1111/nph.14848);
* `Medlyn.py` is derived/adapted from the work of
[Medlyn et al. (2011)](https://doi.org/10.1111/j.1365-2486.2010.02375.x);
* `ProfitMax.py` is derived/adapted from the work of
[Sperry et al. (2017)](https://doi.org/10.1111/pce.12852);
* `ProfitMax2.py` is derived/adapted from the work of
[Wang et al. (2020)](https://doi.org/10.1111/nph.16572);
* `SOX.py` is derived/adapted from the work of
[Eller et al. (2018)](https://doi.org/10.1098/rstb.2017.0315) and
[Eller et al. (2020)](https://doi.org/10.1111/nph.16419);
* `Tuzet.py` is derived/adapted from the work of
[Tuzet et al. (2003)](https://doi.org/10.1046/j.1365-3040.2003.01035.x);
* `WUE_LWP.py` is derived/adapted from the work of
[Wolf et al. (2016)](https://doi.org/10.1073/pnas.1615144113).

&nbsp;

The model's biogeophysical routines can be found in the `SPAC/` repository,
ranging from atmosphere-to-leaf micrometeorology (`canatm.py`) to plant
hydraulics (`hydraulics.py`).

&nbsp;

All support routines (automating the format of the input files, etc.) can be
found in `Utils/`.

&nbsp;

Manon Sabot: [m.e.b.sabot@gmail.com](mailto:m.e.b.sabot@gmail.com?subject=[One_gs_model_to_rule_Code]%20Source%20Han%20Sans)
