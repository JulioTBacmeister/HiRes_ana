from pysr import PySRRegressor
import numpy as np
import xarray as xr
import os
from data_loader_helper import *


"""
Notes: 

- !!Be sure to experiment with large populations and niterations (O(1000)?) early, as equations from smaller experiments don't always reflect larger runs with the same hyperparameters/settings - 
new "regimes" sometimes emerge. This includes new variables appearing as important (in place of others earlier in the run) or complicated expressions with many variables replaced by more nonlinear functions of fewer vars.

- exp() or more nonlinear operators (higher powers) require more constraints, and often nested constraints (i.e. "exp": {"exp": 0} ). In some cases, multiple nested (i.e. "exp": {"exp": {"exp": 0} ...})
Functions like x1^x2 and x1*exp(x2^x3) can be dealt with by adjusting the nested_constraints (note this is more precise than using the "complexity_of_operators" parameter, which tends to get rid of certain operators altogether). 
Adjusting `complexity_of_variables` parameter can also help address these issues (especially when the power operator relies on multiple variables). 

Arguments in the power operator tend to become complicated, the constraint `constraints={"^": (-1, 1)}` helps limits the complexity of the power argument (you can experiment with more complex args `constraints={"^": (-1, 2)}). 

- The ratio complexity_of_variables/complexity_of_constants also is important (not just each individually) 
Using complexity_of_variables = 1 (or too small) leads to complicated combinations of variables early on [often with the power operator, such as x_1^(x2*x3)].

- Note the best `complexity_of_variables` value will likely depend on number of variables (chosen for 8 input variables in this example)

- The native PySR `select_k_features` parameter didn't work well for me. 
For reducing number of variables in the final equations, increasing the complexity of variables + doing very large runs can help identify which variables are important.

- Batching is critical to make datasets of size > ~O(10^4) manageable. Compute is generally better spent on longer runs (with smaller batches) vs shorter runs with no batching (and more data points). 

- refer to : https://astroautomata.com/PySR/tuning/, although note some functionality mentioned is outdated or no longer supported



- experiment with different variable scalings. E.g. z-scoring a strictly positive variable x1 might get rid of any x1^a where a<1. In theory and depending on the complexity settings, the algorithm should recover (x1+offset)^a but in practice it doesn't always and so scaling choices can change the form and performance of the equations

- see https://ai.damtp.cam.ac.uk/pysr/api/ for the many options in PySRRegressor

"

  Two things I'd keep in your back pocket: scale your variables to O(1)–O(10) before handing them to PySR (that was the difference between R²=0.87 and an exact recovery), and always conda activate rather than calling the env's python by path, or juliapkg goes looking in the read-only JupyterHub prefix.

  Also worth flagging: /glade/work is at 96% quota, and each of those Julia backends is ~1 GB. Not urgent, but it's a tight ceiling if you go building more envs.

"""

data_path = "../normalized_training_nearest_neig_particle_i6_m31_exp51.nc"

X_vars = ["mix_len_pi1", "mix_len_pi2", "mix_len_pi3", "tke", "bgrad", "strain", "z_obu", "res_obu"]
sym_reg_names = ["pi_1", "pi_2", "pi_3", "tke", 'bgrad', "strain", "z_obu",'res_obu']


y_var = "lmix"
X, Y = load_and_process_data(data_path, X_vars, y_var, N_samples=None)

outputs_dir = 'pysr_outputs' # optional directory for outputs (pysr creates a folder within this for each run)
os.makedirs(outputs_dir, exist_ok=True)


print("Started")
model = PySRRegressor(
    niterations=2000,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=[ "exp", "log", "abs", "neg", "sqrt"], #, "tan", "tanh", "cosh", "erf", "log10", "abs", "erf", "cube", "square", "cos", "sin", 
    populations = 80,
    procs = 29,
    maxsize = 50,
    # complexity_of_operators={
    #      "-": 5,
    # },
    turbo = True,
    complexity_of_variables = 2,
    complexity_of_constants = 1,
    nested_constraints= {"exp": {"exp": 0, "^": 0, "log": 0}, # careful to keep keys unique
                        "^": {"^": 0},
                        "sqrt": {"sqrt": 0, "log": 0},
                        "inv": {"inv": 0,},
                        "log": {"exp": 0, "log":0}},
    constraints={"^": (-1, 1)},
    batching = True,
    batch_size = 1000,
    output_directory=outputs_dir # optional, a folder with the timestamp gets created in this directory for each run
    )

    # select_k_features=3,
    # cluster_manager="slurm",
    # tempdir=tempdir,
    # progress = False,
    # ncyclesperiteration = 1000,
    # temp_equation_file=True,
    # delete_tempfiles=False,
    #     "cos": 3, 
    #     "cube": 3,
    #     "sin": 3,
    #     "square": 3,
    #     "-": 2,
    #     "pow": 3,
    # },
    # extra_sympy_mappings={"inv": lambda x: 1 / x},
    # extra_sympy_mappings={"pow": "x^c"},
    # extra_sympy_mappings={"^": lambda x: x**c},
    # complexity_of_variables = 2,
    # loss="loss(prediction, target) = (prediction - target)^2",
    # optimizer_algorithm="BFGS",
    # model_selection = "best", # "accuracy ""



# test 
# X = 2 * np.random.randn(500, 2) + 10
# y = 2.5382 * (X[:, 1]**(2.5))


model.fit(X, Y, variable_names= sym_reg_names)
print(model)


