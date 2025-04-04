# Simple image processing based art ipart.py

Repository to develop algorithms for image processing based art

## Python environment

### Using Anaconda

Anaconda can be downloaded from the [Anaconda website](https://www.anaconda.com/products/individual).

After installing Anaconda, open a Anaconda prompt and create a new environment. Here we name it ipart, but the name can be anything.

```shell
conda create --name pyipart python=3.10
# and to activate
conda activate pyipart
# to deactivate
conda deactivate
```

To remove the environment, if you want/need to start fresh.

```shell
conda env remove --name pyipart
```

## Installation

Once you have your environment setup you can install the package and its requirements.

- [Numpy](https://numpy.org/), [opencv](https://opencv.org/), ...
```shell
pip install numpy==1.26.4 opencv-python==4.10.0.84 tqdm packaging psutil imageio
```
- You may also want to have [matplotlib](https://matplotlib.org/stable/install/index.html) for some debugging
```shell
conda install matplotlib
```
and
- the package itself
```shell
cd path/to/repo/ipart.py
pip install -e .
```
