## Requirements

```bash
conda create -n stl_rl python=3.8
pip install -r requirements.txt --force-reinstall
plotly_get_chrome
```

## Run

To run all the experiments use the following bash scripts:

```bash
./pendulum_experiments.sh
python plot.py
```

The results are saved in results/pendulum/media

