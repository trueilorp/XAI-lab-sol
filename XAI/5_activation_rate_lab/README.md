## How to install

The installation requires python>=3.10.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), which is a fast and reliable dependeny manager for python.
2. Run `uv sync` inside the repo and wait for the dependencies to install
3. Activate the virtual environment created at step 2 to start working on the repository

## Task

Go to `src/runners/parallel_runner.py` and implement the function at the end of the file: `_does_rule_activate'.

Feel free to reduce the value if the BATCH_SIZE parameter or remove some theory indexes to speed up computation in the debugging phase!

Once you are done, you can log the activation rate. Just run "./run.sh" and wait for it to end (it can take a bit).
The results (a .csv) will be saved in the 'plots' folder. To plot the activation rate, just run './plot.sh' and check the pdf in the same folder.

## License
This repo has is a modified version of https://github.com/uoe-agents/epymarl, done by Edoardo Zorzi. See the original repo for all the details.
