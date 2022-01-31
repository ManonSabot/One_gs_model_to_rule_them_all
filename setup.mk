no_conda_msg1="Anaconda is not installed. Install it and try again!"
no_conda_msg2="To install Anaconda, head to: https://docs.anaconda.com/anaconda/install/"

all: combine_files cleanup_files check_conda create_env

combine_files:
	cat input/simulations/idealised/split_*.csv > input/simulations/idealised/sensitivity_mtx.csv
	cat output/simulations/idealised/sensitivities/split_*.csv > output/simulations/idealised/sensitivities/model_sensitivities.csv

cleanup_files:
	rm input/simulations/idealised/split_*.csv
	rm output/simulations/idealised/sensitivities/split_*.csv

check_conda:
	@conda info --envs || (echo ${no_conda_msg1}; echo ${no_conda_msg2}; exit)

create_env:
	@conda env create -f src/extra/multi_gs_craze.yml; echo "please, activate multi_gs_craze"
