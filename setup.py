from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()


def _parse_requirements(requirements_txt_path):
    with open(requirements_txt_path) as fp:
        return fp.read().splitlines()


setup(
    name='physics_constrained_nn',
    version='0.0.1',
    description='A module for training neural networks of unknown dynamical systems that can incorporate physics knowledge in form of structural knowledge of constraints on the internal state of the model. Also contains robustness improvements made by Santiago Andrés Serrano-Vacca.',
    license="GNU 3.0",
    long_description=long_description,
    author='Franck Djeumou, Cyrus Neary, Santiago Serrano',
    author_email='fdjeumou@utexas.edu, cneary@utexas.edu, santiagoserrano334@gmail.com',
    url="https://github.com/santiago-a-serrano/robust_physics_constrained_nn.git",
    packages=['physics_constrained_nn'],
    package_dir={'physics_constrained_nn': 'physics_constrained_nn'},
)
