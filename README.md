# Robust Neural Network-Based Discovery of Dynamical Systems

This module builds on top of [Djemou _et al_'s](https://github.com/wuwushrek/physics_constrained_nn) work in order to implement techniques that allow for more robustness against noise in the training data for the discovery of dynamics. The corresponding draft of the paper can be found [here](Paper.pdf).

## Installation

The code is written both in C++ and Python.

This package requires [``jax``](https://github.com/google/jax) to be installed. The package further requires [``dm-haiku``](https://github.com/deepmind/dm-haiku) for neural networks in jax and [``optax``](https://github.com/deepmind/optax) a gradient processing and optimization library for JAX.

However, there are some incompatibilities between versions of packages that are used for different purposes. Two requirements.txt files are included, with the intention of them being used for the creation of two virtual environments. Both virtual environments were used along with Python 3.10.12, although more recent versions will probably also work.

## Installation Steps
### 1. Create two virtual environments, "default"  and "gen".
The "default" environment will be used for running most scripts, while "gen" will be used for generating training data (trajectories). More information on that follows.
```
python3 -m venv default
source default/bin/activate # Unix/Linux/macOS
default\Scripts\activate # Windows
pip install -r requirements_default.txt
deactivate
```
```
python3 -m venv gen
source gen/bin/activate # Unix/Linux/macOS
gen\Scripts\activate # Windows
pip install -r requirements_gen.txt
deactivate
```
You can later activate any of the two environments with 
```
source envname/bin/activate # Unix/Linux/macOS
envname\Scripts\activate # Windows
```

### 2. Clone this repo and install physics_constrained_nn
You can do it by running the following commands in the terminal:
```
git clone https://github.com/santiago-a-serrano/robust_physics_constrained_nn.git
cd physics_constrained_nn/
python3 -m pip install -e . 
```

### 3. MuJoCo

This package implements several jax primitives in C++ of ``MuJoCo`` functions that can be used as prior physics knowledge. Then it uses [``pybind11``]() to import the primitives and use it in Python. To include such primitives, ``MuJoCo`` needs to be installed on the target computer with a valid activation key. 

#### MuJoCo Experiments
Follow [the installation procedure](https://www.roboti.us/) to install ``MuJoCo``. Then, set the environment variables ``MUJOCO_PY_MJKEY_PATH`` and ``MUJOCO_PY_MUJOCO_PATH`` ( these names are typically used by ``mujoco-py`` to find the MuJoCo library files). For example, if the binaries, include and libraries of MuJoCo are unzipped in ``~/.mujoco/mujoco200_linux``, then you can excute the following
```
echo 'export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco200_linux' >> ~/.bashrc 
echo 'export MUJOCO_PY_MJKEY_PATH=~/.mujoco/mujoco200_linux/bin/mjkey.txt' >> ~/.bashrc 
source ~/.bashrc
```

Finally, install the following dependencies to compile the C++ code
```
sudo apt install build-essential libomp-dev
```

## Examples

### Generate data (trajectories)

To first generate the data required to train the neural network, modify the ``dataset_gen.yaml`` file inside the ``physics_constrained_nn/examples/double_pendulum`` directory and generate the dataset as follows:
```
cd physics_constrained_nn/examples/double_pendulum
python generate_sample.py --cfg dataset_gen.yaml --output_file generated/trajectories/traj_name
```

### Training

After the file has been generated, modify the parameters of your training from ``nets_params.yaml`` and proceed to the training as follows:
```
python train.py --cfg nets_params.yaml --input_file generated/trajectories/traj_name.pkl --output_file generated/trained_models/model_name --baseline base --side_info 0
```
where the baseline is either `base` or `rk4` and the side info is either `0` (no side information), `1` (structural knowledge of vector field), and `2` (structural knowledge + symmetry constraints).

### Model comparison

Finally, to plot the results, execute the command line
```
python perform_comparison.py --logdirs generated/trained_models/model_name1.pkl generated/trained_models/model_name2.pkl ... --legend 'Model 1' 'Model 2' ... --colors red green ... --num_traj 100 --num_point_in_traj 100 --seed 5 --show_constraints --window 5
```

### Other comparisons

Other comparisons, like the ones shown in the paper, can be achieved through the following scripts inside the ``physics_constrained_nn/examples/double_pendulum/`` directory.

* ``adv_noise_comparison.py`` compares the accumulated error of 4 approaches (with various levels of adversarial noise):
    * A1 with and without gradient regulatization.
    * A2 with and without gradient regularization.
(See paper for A1 and A2 definitions).
    * The trained models can be specified by changing the default values of the ``trained_models`` variable. Some other parameters (that won't have to be modified in most cases) can be modified as needed in the first lines of the ``main`` function. 
* ``nn_vs_gp_vs_sindy.py`` compares the accumulated error of SINDy, GPSINDy, and four chosen trained models.
    * The trained models can be specified by changing the default values of the ``trained_models`` variable. The optimized hyperparameters for the Gaussian Process regression must be specified in ``optimized_hyperparams``. The paths of the data to train SINDy and GPSINDy with can be specified in ``sindy_train_paths``. Some other parameters (that won't have to be modified in most cases) can be modified as needed in the first lines of the ``main`` function. 
* ``nn_vs_gradregnn.py`` compares the accumulated error of four chosen trained models.
    * The trained models can be specified by changing the default values of the ``trained_models`` variable. Some other parameters (that won't have to be modified in most cases) can be modified as needed in the first lines of the ``main`` function. 
* ``no_gpr_vs_gpr.py`` compares the accumulated error of a basic neural network and that network but with Gaussian Process regression.
    * The trained models can be specified by changing the default values of the ``trained_models`` variable. Some other parameters (that won't have to be modified in most cases) can be modified as needed in the first lines of the ``main`` function. 
* ``sindy_vs_gpsindy.py`` compares the accumulated error of SINDy and GPSINDy for the specified noise level.
    * Most important parameters will be asked as input in the terminal as the script runs. Some other parameters (that won't have to be modified in most cases) can be modified as needed in the first lines of the ``main`` function. 


### Adversarial noise generation
``adversarial_noise_finder.py`` generates the adversarial noise that can be later specified in ``dataset_gen.yaml``.

Four parameters must be specified at the end of the file, when calling the ``main`` function:
1. ∥ϵ∥ (``max_x_noise``).
2. The model to generate adversarial noise against (``trained_model_path``).
3. Trajectories (input data) to be used when generating the noise (``trajectories_path``).
4. If A1 (``False``) or A2 (``True``) is to be implemented (``make_traj_noise``).

To generate the noise, just run the script.


## Pregenerated Files
In order to help save training and data generation time/computing, some pregenerated trajectories, trained models, and adversarial noise are available under ``physics_constrained_nn/examples/double_pendulum/pregenerated/``.

### Suggested Extension (optional)
The pregenerated file formats can be pre-visualized within VS Code with the ``vscode-pydata-viewer`` extension. Although it is not necessary to visualize them in order to use them, it may be helpful to understand them better.