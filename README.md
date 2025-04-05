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
pip install numpy==1.26.4 opencv-python==4.10.0.84 tqdm packaging psutil
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

## The game of life

The [original algorithm](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) also known as Conway's Game of Life is a cellular automaton for a zero-player game where the evolution is determined by its initial state. One interacts with the Game of Life by creating an initial configuration and observing how it evolves. For binary images the algorithm is as follows:
- Birth: if unpopulated pixel surrounded by exactly n populated pixels, becomes populated next generation.
- Death by loneliness: if populated pixel is surrounded by n - k or fewer, the pixel becomes unpopulated next generation.
- Death by overpopulation: if populated pixel is surrounded by n + k or more, the pixel becomes unpopulated next generation.

My proposed modifications for color images:
- the difference between the pixel and the neighborhood is calculated using deltaE formula
- the neighborhood is characterized by the average and standard deviation of the pixels around the given pixel.
- Underpopulated neighborhood: if the standard deviation of the neighborhood is less than a given minimum standard deviation.
- Overpopulation neighborhood: if the standard deviation of the neighborhood is greater than a given maximum standard deviation.
- Death: the pixel will die if the distance between the pixel and the neighborhood is greater than a giving minimum difference.
- Death: the pixel will die if the neighborhood is overpopulated.
- Birth: the pixel will be born if the distance between the pixel and the neighborhood is less than a given difference or is already dead and the neighborhood is underpopulated.
- Death: set to a random color + noise
- Birth: set to the average of the neighborhood + noise

The following are examples of the output of the algorithm. Note that there are a few parameters that could be tuned in order to get different results
```python
wsize: int = 5 # Controls the window size of the neighborhood
mu: float = 3 # Controls the threshold how different can the pixel be
sigma: tuple[float, float] = (5, 15) # The limits of over and under population
add_noise: tuple[float, float] = (10 / 255, 0.02) # Added initial random noise to the image
```
Example images:

![test_0_gol.gif](/data/test_0_gol.gif)
![test_1_gol.gif](/data/test_1_gol.gif)
![test_2_gol.gif](/data/test_2_gol.gif)
![test_3_gol.gif](/data/test_3_gol.gif)
![test_4_gol.gif](/data/test_4_gol.gif)
