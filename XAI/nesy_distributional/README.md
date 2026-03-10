## NeSy Distributional RL Lab

In this repository you can find the code to run the C51 algorithm and its heuristic-guided variant on the DoorKey environment from Minigrid. The code is adapted from CleanRL.

### Repository Structure

- **baselines/**:  
  This folder contains the implementation of the normal C51 algorithm without any additional rules.

- **Heuristic-Guided variant**:
  `h_c51_product.py` contains the heuristic-guided variant of C51 that you'll have to implement.

- **base_config.py and product_config.py**:  
  This files are used to set the different parameters for each algorithm (learning rate, batch size, environment settings, etc.).

- **params.env** sets the training steps and environment to run the experiments.

### How to run

1. **Create a Python environment**  
   Create a virtual environment with Python 3.12.6 and install the packages. For example, using conda:
   ```bash
   conda create -n xai-project-minigrid python=3.12.6
   conda activate xai-project-minigrid
   pip install -r requirements.txt
   ```
   Or you can install `uv` and do `uv sync`.


4. **Configure the params.env file** \
   You can change the hyperparameters in the `params.env` file to train the algorithms. The default parameters are suitable for debug purposes only.

5. **Run the scripts** \
   Run ```python run_experiments.py``` or ```uv run run_experiments.py``` if you are using `uv` to generate a plot that compares the training returns of both base C51 and the heuristic-guided version you implemented.
