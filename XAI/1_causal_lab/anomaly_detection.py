"""
==============================================================
 Causal Discovery & Anomaly Detection (Tigramite)
==============================================================

Workflow:
1. Data Loading
2. Causal Model Learning (offline PCMCI)
3. Offline Coefficient Estimation
4. Online Monitoring (moving-window coefficient updates)
5. Anomaly Detection
6. Metrics Computation

Author: Simone Dario
==============================================================
"""

# ================== IMPORTS ==================
import os
import numpy as np
import pandas as pd
import warnings
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from scipy.stats import ConstantInputWarning
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=ConstantInputWarning)

# ================== GLOBAL CONFIG ==================
ALPHA = 0.05
TRAINING_FRAC = 0.7
PREFIX = "/home/trueilorp/xai-course/XAI/1_causal_lab/pepper_csv/"
TASK = "pepper"
MAX_FREQ_COMPONENTS = 5

# ================================================================
#                         DATA LOADING
# ================================================================
def read_data(path: str, task: str) -> pd.DataFrame:
	"""
	Load CSV data, handle timestamp indexing.

	Visual:
		CSV -> DataFrame indexed by Timestamp
	"""
	df = pd.read_csv(path, delimiter="," if task == "pepper" else ";")
	if task == "pepper":
		df["Timestamp"] = df["timestamp"]
		df.set_index("Timestamp", inplace=True)
		df.drop(columns=["timestamp"], inplace=True)
	else:
		df["Timestamp"] = pd.to_datetime(df[" Timestamp"].str.strip(),
										 format="%d/%m/%Y %I:%M:%S %p")
		df.set_index("Timestamp", inplace=True)
		df.drop(columns=[" Timestamp"], inplace=True)
	return df


# ================================================================
#                     LEARN CAUSAL MODEL
# ================================================================
def learn_causal_model(normal_csv_path: str, save_path: str):
	"""
	Learn the causal graph using PCMCI.
	Tau_max is automatically computed from the dominant frequency.

	Steps:
		1️⃣ Load normal data
		2️⃣ Filter top frequency components
		3️⃣ Remove near-constant variables
		4️⃣ Compute tau_max from max frequency
		5️⃣ Run PCMCI
		6️⃣ Save the model
	"""
	print("Learning causal model...")

	df = read_data(normal_csv_path, TASK)
	normal_data = pp.DataFrame(np.nan_to_num(df.values))
	#restrict to training_frac
	normal_data.values[0] = normal_data.values[0][:int(TRAINING_FRAC*np.shape(normal_data.values[0])[0]), :]
	# attack_data = pp.DataFrame(np.nan_to_num(attack_df.values))

	frequencies = []
	for index in range(np.shape(normal_data.values[0])[1]):
		#the signal is made of continuous variables
		if any([el for el in normal_data.values[0][:,index] if int(el)!=el]):
			w = np.fft.fft(normal_data.values[0][:,index])
			freqs = np.fft.fftfreq(len(w))
			mods = abs(w)
			max_indices = np.argsort(mods)[::-1][:MAX_FREQ_COMPONENTS]
			main_freq = []
			for i in max_indices:
				freq = freqs[i]
				main_freq.append(freq)
			frequencies += main_freq

	# print(np.sort(frequencies)[::-1][:MAX_FREQ_COMPONENTS])
	sorted_freq = np.sort([el for el in frequencies if el>0])[::-1]
	for freq in sorted_freq:
		if len([fr for fr in sorted_freq if fr < freq]) / len(sorted_freq) < 0.95:
			max_freq = freq
			sorted_freq = [s for s in sorted_freq if s <= max_freq]
			break

	subsample = max(1, int(np.floor(1/10/max_freq)))
	normal_data.values[0] = normal_data.values[0][::max(1, subsample), :]
	nonconst = [idx for idx in range(np.shape(normal_data.values[0])[1]) if np.std(normal_data.values[0][:, idx]) > 0.01 * np.mean(normal_data.values[0][:, idx])]
	nonconst_data = normal_data.values[0][:, nonconst]
	for j in range(np.shape(nonconst_data)[1]):
		nonconst_data[:,j] /= (np.max(nonconst_data[:,j]) - np.min(nonconst_data[:,j])) + np.min(nonconst_data[:,j])
	print(np.shape(nonconst_data))
	
	# Evaluate links
	tau_max = int(np.floor(max_freq / np.mean(np.unique(sorted_freq))))
	print(tau_max)

	# TODO --- 5. Run PCMCI ---
	cond_ind_test = ParCorr()
	dataframe = pp.DataFrame(data=df, var_names=df.columns.tolist()) 
	pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=1)
	all_parents = pcmci.run_pc_stable(tau_max=tau_max, pc_alpha=ALPHA)
	results = pcmci.run_pcmci(tau_max=tau_max)

	# --- 6. Save model for reuse ---
	np.savez(save_path,
			 val_matrix=results["val_matrix"],
			 p_matrix=results["p_matrix"],
			 var=df.columns,
			 subsample=subsample,
			 nonconst=nonconst)
	print(f"Saved causal model to {save_path}")
	return results, subsample, nonconst, tau_max


# ================================================================
#                  OFFLINE COEFFICIENT FITTING
# ================================================================
def fit_normal_coeffs(normal_data: np.ndarray, causal_matrix: np.ndarray):
	"""
	Compute offline (baseline) coefficients for each variable.

	Visual:
		X_t = sum_j (a_j * X_parent_j_(t-delay_j)) + bias
		------------------------------------------------
		offline coefficients: learned on full normal dataset
	"""
	indices = np.array(np.where(causal_matrix != 0))
	fine_coeffs = {}
	for var in np.unique(indices[1, :]):
		# TODO: compute fine_coeffs[var]
		fine_coeffs[var] = 0

	return fine_coeffs, indices

# ================================================================
#             ONLINE COEFFICIENTS & ERROR COMPUTATION
# ================================================================
def compute_online_errors(data: np.ndarray, fine_coeffs: dict,
						  causal_matrix: np.ndarray, indices: np.ndarray):
	"""
	Recompute coefficients online over a moving window and compute deviations.

	Visual:
		Time t-3  t-2  t-1  [t]
			  |---- moving window ----|
					  ↑ online regression
		norm_agg[t,i] = ||online_coeffs - offline_coeffs||
	"""
	max_time = data.shape[0] - causal_matrix.shape[2]
	err = {}
	norm_agg = np.zeros((max_time, len(np.unique(indices[1, :]))))

	for t in range(max_time):
		for i, var in enumerate(np.unique(indices[1, :])):
			var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
			var_indices.sort(key=lambda x: x[2])
			max_delay = var_indices[-1][2]

			# Fit online coefficients up to time t
			stack = [data[max_delay - el[2]: t + causal_matrix.shape[2] - el[2], el[0]]
					 for el in var_indices]
			stack.append(np.ones(t + causal_matrix.shape[2] - max_delay))

			coeffs = np.linalg.lstsq(np.column_stack(stack),
									 data[max_delay: t + causal_matrix.shape[2], var],
									 rcond=None)[0][:-1]

			# Store deviation
			if var not in err:
				err[var] = np.zeros((max_time, len(var_indices)))
			err[var][t, :] = coeffs - fine_coeffs[var]
			norm_agg[t, i] = np.linalg.norm(err[var][t, :])

	return err, norm_agg


# ================================================================
#                        ANOMALY DETECTION
# ================================================================
def detect_anomalies(err_normal, err_attack, normal_data_len, normal):
	"""
	Flag anomalies if online coefficients deviate significantly from offline baseline.

	Threshold = 0.8 * norm of offline deviations.
	"""
	indices_error = []
	for var in err_attack.keys():
		for j in range(err_attack[var].shape[1]):
			thresh = 0.8 * np.linalg.norm(err_normal[var][:normal_data_len, j])
			if not normal:
				indices_error += list(np.where(abs(err_attack[var][:, j]) > thresh)[0])
			else:
				indices_error += list(np.where(abs(err_attack[var][normal_data_len:, j]) > thresh)[0])
	return len(np.unique(indices_error))



# ================================================================
#          FEATURE IMPORTANCE PLOTTING (SUBPLOTS)
# ================================================================
def plot_feature_importance_subplots(norm_agg_list, indices, nonconst, var_names, attack_names, top_frac=0.1):
	"""
	Plot top anomalous variables per attack as barplots in subplots.
	
	Inputs:
		norm_agg_list : list of np.ndarray
			Aggregated online deviations for each attack (L2 norm over time)
		indices       : np.ndarray
			Indices of causal parents from offline coefficients
		nonconst      : list/np.ndarray
			Indices of non-constant variables
		var_names     : list
			Names of all variables
		attack_names  : list
			Names of attack datasets (for subplot titles)
		top_frac      : float
			Fraction of top variables to show
	"""
	n_attacks = len(norm_agg_list)
	fig, axes = plt.subplots(1, n_attacks, figsize=(6*n_attacks, 5), squeeze=False)
	plt.suptitle("Top Anomalous Variables per Attack", fontsize=16)

	for i, norm_agg_attack in enumerate(norm_agg_list):
		# Compute aggregated L2 norm per variable
		dep_vals = {var_names[nonconst[var]]: np.linalg.norm(norm_agg_attack[:, j])
					for j, var in enumerate(np.unique(indices[1, :]))}
		# Sort descending
		dep_sorted = dict(sorted(dep_vals.items(), key=lambda x: x[1], reverse=True))
		top_n = max(1, int(top_frac * len(dep_sorted)))
		top_items = list(dep_sorted.items())[:top_n]
		top_vars, top_vals = zip(*top_items)

		ax = axes[0, i]
		ax.bar(top_vars, top_vals, color='salmon')
		ax.set_xticklabels(top_vars, rotation=45, ha='right')
		ax.set_ylabel("Aggregated Error (L2 Norm)")
		ax.set_title(attack_names[i])

	plt.tight_layout()
	plt.show()


# ================================================================
#                         MAIN PIPELINE
# ================================================================
def main():
	print(f"\n========== TASK: {TASK.upper()} ==========")

	causal_path = os.path.join(PREFIX, f"{TASK}_normal.npz")
	print(f"Causal model path: {causal_path}")

	# 1️⃣ Learn or load causal model
	if not os.path.exists(causal_path):
		learn_causal_model(PREFIX + "normal.csv", causal_path)
	else:
		print("Causal model found, loading...")

	f = np.load(causal_path, allow_pickle=True)
	val_matrix, p_matrix = f["val_matrix"], f["p_matrix"]
	subsample, nonconst = int(f["subsample"]), f["nonconst"]

	normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > np.mean(abs(val_matrix)))

	# 2️⃣ Load normal data
	normal_df = read_data(PREFIX + "normal.csv", TASK)
	normal_data = np.nan_to_num(normal_df.values[:int(TRAINING_FRAC * len(normal_df))][::subsample, nonconst])
	normal_data_full = np.nan_to_num(normal_df.values[::subsample, nonconst])

	# 3️⃣ Offline coefficients
	fine_coeffs, indices = fit_normal_coeffs(normal_data, normal_matrix)

	# 4️⃣ Online deviations
	err_normal, norm_agg_normal = compute_online_errors(normal_data_full, fine_coeffs, normal_matrix, indices)

	# 5️⃣ Load and detect anomalies
	attack_paths = [
		PREFIX + "WheelsControl.csv",
		PREFIX + "JointControl.csv",
		PREFIX + "LedsControl.csv"
	]
	attack_dfs = [read_data(p, TASK) for p in attack_paths]

	tpos, fpos, fneg = [], [], []

	# False positives
	fpos.append(detect_anomalies(err_normal, err_normal, len(normal_data), normal=True))

	norm_agg_attacks = []
	attack_names = []

	for path, df_attack in zip(attack_paths, attack_dfs):
		attack_name = os.path.basename(path)
		attack_names.append(attack_name)
		print(f"\n--- Analyzing anomaly: {attack_name} ---")

		attack_data = np.nan_to_num(df_attack.values[::subsample, nonconst])
		err_attack, norm_agg_attack = compute_online_errors(attack_data, fine_coeffs, normal_matrix, indices)
		norm_agg_attacks.append(norm_agg_attack)

		tp_count = detect_anomalies(err_normal, err_attack, len(normal_data), normal=False)
		tpos.append(tp_count)
		fneg.append(attack_data.shape[0] - tp_count)

		# Top 10% variables with highest aggregated error
		dep_vals = {f["var"][nonconst][var]: np.linalg.norm(norm_agg_attack[:, i])
					for i, var in enumerate(np.unique(indices[1, :]))}
		dep_vals_sorted = dict(sorted(dep_vals.items(), key=lambda x: x[1], reverse=True))
		top_n = max(1, int(0.1 * len(dep_vals_sorted)))
		# print("Top 10% anomalous variables:")
		# for i, (k, v) in enumerate(list(dep_vals_sorted.items())[:top_n]):
		#     print(f"{k:<20} | Aggregate Error: {v:.3f}")
		# print("========================================")

	# 6️⃣ Metrics
	precision = np.sum(tpos) / (np.sum(tpos) + np.sum(fpos))
	recall = np.sum(tpos) / (np.sum(tpos) + np.sum(fneg))
	f1 = 2 * np.sum(tpos) / (2 * np.sum(tpos) + np.sum(fpos) + np.sum(fneg))

	print("\n========== METRICS ==========")
	print(f"Precision: {precision:.3f}")
	print(f"Recall:    {recall:.3f}")
	print(f"F1 Score:  {f1:.3f}")
	print("=============================\n")
	
	# ===== PLOT FEATURE IMPORTANCE FOR ALL ATTACKS =====
	plot_feature_importance_subplots(norm_agg_attacks, indices, nonconst, f["var"], attack_names, top_frac=0.1)


# ================================================================
#                          RUN SCRIPT
# ================================================================
if __name__ == "__main__":
	main()



