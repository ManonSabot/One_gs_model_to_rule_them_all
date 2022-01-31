all: calibrations simulations plots

calibrations:
  # src/calibrations/calib_2_model.py -c
  # src/calibrations/calib_2_obs.py -c
	src/calibrations/calib_2_model.py
	src/calibrations/calib_2_obs.py

simulations:
	src/simulations/sim_idealised.py -r
	src/simulations/sim_idealised.py
	src/simulations/sim_sensitivities.py
	src/simulations/sim_obs_driven.py -r
	src/simulations/sim_obs_driven.py

plots:
	src/plots/plot_params_idealised_calibs.py
	src/plots/plot_deriv_vs_not.py
	src/plots/plot_idealised.py -xpe
	src/plots/plot_sensitivities.py
	src/plots/plot_params_obs_calibs.py
	src/plots/plot_goodness_of_fit.py
	src/plots/plot_skill_obs_calibs.py
	src/plots/plot_obs_driven.py
