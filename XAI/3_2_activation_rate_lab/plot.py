import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

ROOT = os.getcwd()
PLOT_DIR = os.path.join(ROOT, "plots")

plt.rcParams["figure.figsize"] = (6, 3)
THEORY_MAX_STEPS = 20001000

TITLE = False

_AXVLINE = True
axvline_pos = 15000000
axvline_color = "#888888"

latest = 0
for file in os.listdir(PLOT_DIR):
    if not file.startswith("activation_rate"):
        continue
    new_time = int(file.split("activation_rate_")[1].split(".csv")[0])
    if new_time > latest:
        latest = new_time

activate_rate = pd.read_csv(
    os.path.join(PLOT_DIR, f"activation_rate_{str(latest)}.csv")
)

activate_rate = activate_rate[activate_rate["action"] != "noop"]

g = sns.lineplot(
    data=activate_rate,
    x="steps",
    hue="action",
    y="activation_rate",
    style="action",
    palette="muted6",
    markers=True,
)
if _AXVLINE:
    g.axvline(axvline_pos, color=axvline_color)
g.set(xscale="log")
if TITLE:
    plt.title("Activation rate (by action)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "plot_activation_rate.pdf"))
