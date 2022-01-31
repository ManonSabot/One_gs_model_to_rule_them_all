# One stomatal model to rule them all?

Manon E. B. Sabot, Martin G. De Kauwe, Andy J. Pitman, Belinda E. Medlyn, David
S. Ellsworth, Nicolas Martin-StPaul, Jin Wu, Brendan Choat, Jean-Marc Limousin,
Patrick J. Mitchell, Alistair Rogers, Shawn P. Serbin

## Overview

Repository containing all the code to reproduce the results in Sabot et al.
(2022): One stomatal model to rule them all? Towards improved representation of
carbon and water exchange in global models. Journal of Advances in Modeling
Earth Systems, Accepted.

## General instructions

To make sure the model and support files are properly set up, simply type:
```
make -f setup.mk
```
> N.B.: You need anaconda to use the existing environment files within which
> to run the model. By default, python3.8 will be used to run the model.
> If you would rather install the dependencies locally, you will need python.

Followed by:
```
source activate multi_gs_craze
```

&nbsp;

To regenerate all our results and figures, type:
```
make
```
> N.B.1: In this demo version, none of the scripts called by `Makefile` are
> setup to run in parallel. This will be very slow, so it would be worth
> parallelising `src/calibrations/calib_2_model.py`,
> `src/calibrations/calib_2_obs.py` and
> `src/simulations/sim_sensitivities.py` shall you wish to recreate our
> results.

> N.B.2: In the current `Makefile`, the commands to calibrate the models are
> commented out as this takes a long time without parallelisation, and the
> outputs are already written to files stored in `output/calibrations/`

&nbsp;

## Model experiments

Our model experiments were meant as a versatile comparison approach to assess
the parameter identifiability, in-sample predictive performance, and
functional adequacy of 12 empirical or optimisation-based stomatal conductance
(<it>g<sub>s</sub></it>) schemes. These experiments are summarised at a glance
in the schematics below. They are detailed in the associated publication, and
the code is also thoroughly commented for anyone interested.

![alt text](src/extra/model_experiments_setup.jpg)

&nbsp;

### Idealised experiments

We conducted two idealised calibration experiments to test the identifiability
of 11 of the canopy gas exchange schemes' parameters. If you wish to reproduce
these experiments (not advised without first parallelising the code), you can
use the following command:
```
src/calibrations/calib_2_model.py -c
```

To analyse the parameter outputs from these experiments, use:
```
src/calibrations/calib_2_model.py
```

To generate the plots associated with these experiments, type:
```
src/plots/plot_params_idealised_calibs.py
```

&nbsp;

In-calibration-sample model simulations that show the <it>g<sub>s</sub></it>
schemes' harmonized behaviours can be reproduced through:
```
src/simulations/sim_idealised.py -r
```

And analysed using:
```
src/simulations/sim_idealised.py
```

To plot the outputs, type:
```
src/plots/plot_idealised.py -xpe
```
And:
```
src/plots/plot_deriv_vs_not.py
```

&nbsp;

We also performed a global sensitivity analysis (which runs outside the
calibration conditions, hence it is out-of-sample). The analysis can be
reproduced using the following command (but again this is not advised without
first parallelising the code):
```
src/simulations/sim_sensitivities.py
```

Associated plots can be generated from:
```
src/plots/plot_sensitivities.py
```

&nbsp;

### Evaluation against observations

Finally, the 12 schemes were all calibrated to observations of
<it>g<sub>s</sub></it> before being ran in-sample, which tests their
functional adequacy. To recalibrate the schemes (again, not recommended
without prior parallelisation of the code), simply use:

```
src/calibrations/calib_2_obs.py -c
```

And to analyse the parameter outputs:
```
src/calibrations/calib_2_obs.py
```

To plot the calibration data and calibrated parameters, type:
```
src/plots/plot_params_obs_calibs.py
```

To get a sense of the models' "goodness-of-fit", you can use:
```
src/plots/plot_goodness_of_fit.py
```

But for a proper assessment of their skill that relies on several statistical
metrics of performance, including advanced metrics which characterize
specific aspects of model behaviour, then you'll need:
```
src/plots/plot_skill_obs_calibs.py
```

Finally, a series of functional relationships between different variables can
be assessed from:
```
src/plots/plot_obs_driven.py
```

&nbsp;

## The model

The model used here is a leaf-level adaptation of the **TractLSM**
[(Sabot et al., 2019)](https://doi.org/10.5281/zenodo.3566722)), modified to
embeds 12 gas exchange schemes. The **TractLSM** is further described
[here](https://github.com/ManonSabot/Profit_Maximisation_European_Forests).
Consulting the ReadMe in `src/TractLSM/` might also help get an idea of the
model structure and of the processes accounted for.

&nbsp;

## Observational data

These data are stored in `input/calibrations/obs_driven/` and
`input/simulations/obs_driven/`. They are compiled from:
* [Wu et al. (2020)](https://doi.org/10.1111/gcb.14820)
* [Choat et al. (2006)](https://doi.org/10.1093/treephys/26.5.657)
* [Heroult et al. (2013)](https://doi.org/10.1111/j.1365-3040.2012.02570.x)
* [Mitchell et al. (2009)](https://doi.org/10.1016/j.agrformet.2008.07.008)
* [Limousin et al. (2013)](https://doi.org/10.1111/pce.12089)
* [Martin St-Paul et al. (2012)](https://doi.org/10.1071/FP11090)

&nbsp;

## License

This project is licensed under the MIT License - see the [License](License) file for details

&nbsp;

## Contact

Manon Sabot: [m.e.b.sabot@gmail.com](mailto:m.e.b.sabot@gmail.com?subject=[One_gs_model_to_rule_Code]%20Source%20Han%20Sans)
