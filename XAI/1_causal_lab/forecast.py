import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import random
import seaborn as sns
from matplotlib import pyplot as plt

# Tigramite imports
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.models import Prediction

# sklearn preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tigramite.data_processing import DataFrame as tgDataFrame


tf.get_logger().setLevel("ERROR")

# -------------------------- CONFIG --------------------------
PC_ALPHA = 0.05
TRAIN_FRAC = 0.7
TARGET = "cooling_demand"
MAX_TAU = 5
T = 500

TCN_FILTERS = 32
TCN_KERNEL = 3
TCN_DROPOUT = 0.25
TCN_EPOCHS = 25
TCN_BATCH = 32
TCN_LR = 0.001

def set_seeds(seed=42):
	"""Fix all random seeds to ensure reproducibility."""
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)
	
	# Optional: force deterministic operations on GPU
	os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seeds(42)


# -------------------------- DATA LOADING --------------------------
def load_citylearn(building_id: int):
	"""Load CityLearn building data and auxiliary files, concatenate columns."""
	base = "citylearn"
	b = pd.read_csv(os.path.join(base, f"Building_{building_id}.csv"))
	carbon = pd.read_csv(os.path.join(base, "carbon_intensity.csv"))
	pricing = pd.read_csv(os.path.join(base, "pricing.csv"))
	weather = pd.read_csv(os.path.join(base, "weather.csv"))
	return pd.concat([b, carbon, pricing, weather], axis=1)


# -------------------------- PREPROCESS --------------------------
def variance_filter(df, threshold=1e-5):
	"""Filter out columns with variance below threshold."""
	cols = df.var()[df.var() > threshold].index.tolist()
	return df[cols]


def compute_tau_max(series, fallback=MAX_TAU):
	"""Estimate a reasonable tau_max from dominant FFT frequency."""
	n = len(series)
	x = series - np.mean(series)
	fft = np.fft.fft(x)
	freqs = np.fft.fftfreq(n)
	fft[0] = 0
	dom_idx = np.argmax(np.abs(fft))
	dom_freq = abs(freqs[dom_idx])
	if dom_freq == 0 or np.isclose(dom_freq, 0.0):
		return fallback
	tau = int(round(1.0 / dom_freq))
	return min(tau, fallback)


# -------------------------- CAUSAL MODEL --------------------------
def fit_causal_model(df, target_col=TARGET, tau_max=MAX_TAU):
	"""Fit Tigramite Prediction model using LinearRegression."""

	# TODO: fit the causal model using Tigramite and LinearRegression
	# X = df.iloc[:, :-1]
	# y = df.iloc[:, -1]
	# model = LinearRegression()
	# model.fit(X, y) 
	
	df_tg = pp.DataFrame(df.values, var_names=df.columns.tolist())
	print("\nDataFrame: ", df_tg)
	
	target_idx = df_tg.var_names.index(target_col)
	print("Target idx: ", target_idx)
	
	model = Prediction(dataframe=df_tg,
        cond_ind_test=ParCorr(),
        prediction_model = LinearRegression(),
    data_transform=StandardScaler(),
    train_indices= range(int(0.8*T)),
    test_indices= range(int(0.9*T), T),
    verbosity=1
    )
	
	print("Model created")	
	
	predictors = model.get_predictors(
                  selected_targets=[target_idx],
                  steps_ahead=1,
                  tau_max=tau_max,
                  pc_alpha=None
                  )
	print("Predictors created")
	
	print("Fitting model...")
	model.fit(target_predictors=predictors, 
                selected_targets=[target_idx],
                    tau_max=tau_max)
	print("Model fitted")
	
	# Extract aligned true values
	print("Getting predictions...")
	pred = model.predict(target_idx)
	
	true_matrix = model.get_test_array()
	
	if true_matrix.ndim == 1:
		true_test = true_matrix
	else:
		if true_matrix.shape[0] == len(df.columns):
			true_test = true_matrix[target_idx, :]
		else:
			true_test = true_matrix[0, :]
	true_aligned = true_test[-len(pred):]

	# Metrics
	value_range = true_aligned.max() - true_aligned.min()
	nmae = np.mean(np.abs(true_aligned - pred)) / (value_range + 1e-12)
	nstd = np.std(true_aligned - pred) / (value_range + 1e-12)

	return model, predictors, pred, true_aligned, nmae, nstd


# -------------------------- DILATED TCN --------------------------
def create_sequences(data, target_col, tau):
	"""Create sliding window sequences for TCN."""
	X, y = [], []
	for i in range(tau, len(data)):
		X.append(data[i-tau:i])
		y.append(data[i, target_col])
	return np.asarray(X), np.asarray(y)


def build_dilated_tcn(input_shape, filters=TCN_FILTERS, kernel_size=TCN_KERNEL,
					  dropout=TCN_DROPOUT, n_blocks=3):
	"""Build a small dilated TCN with residual blocks."""
	inp = Input(shape=input_shape)
	x = inp
	for b in range(n_blocks):
		dilation = 2 ** b
		conv = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation)(x)
		conv = BatchNormalization()(conv)
		conv = Activation("relu")(conv)
		conv = Dropout(dropout)(conv)

		conv2 = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation)(conv)
		conv2 = BatchNormalization()(conv2)
		conv2 = Activation("relu")(conv2)
		conv2 = Dropout(dropout)(conv2)

		skip = Conv1D(filters, 1, padding="same")(x) if x.shape[-1] != filters else x
		x = Add()([conv2, skip])

	x = Flatten()(x)
	x = Dense(64, activation="relu")(x)
	x = Dropout(0.3)(x)
	out = Dense(1, activation="linear")(x)
	return Model(inp, out)


def fit_tcn(X_train, y_train, X_test, y_test):
	"""Fit TCN and return aligned predictions."""
	input_shape = (X_train.shape[1], X_train.shape[2])
	model = build_dilated_tcn(input_shape)
	model.compile(optimizer=Adam(learning_rate=TCN_LR), loss="mse")

	es = EarlyStopping(monitor="loss", patience=4, restore_best_weights=True, verbose=0)
	rlr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2, verbose=0, min_lr=1e-6)

	model.fit(X_train, y_train, epochs=TCN_EPOCHS, batch_size=TCN_BATCH, verbose=0, callbacks=[es, rlr])

	pred_all = model.predict(X_test, verbose=0).flatten()
	pred = pred_all[-len(y_test):]  # align to test set
	value_range = y_test.max() - y_test.min()
	nmae = np.mean(np.abs(y_test - pred)) / (value_range + 1e-12)
	nstd = np.std(y_test - pred) / (value_range + 1e-12)
	return model, pred, nmae, nstd


# -------------------------- FEATURE IMPORTANCE --------------------------
def causal_feature_importance(causal_model, predictors, feature_names, target_idx):
	"""
	Extract predictor importance from the fitted Tigramite Prediction model.
	Uses absolute value of regression coefficients from the fitted LinearRegression.
	"""
	# Access the fitted LinearRegression for the target
	lr_model = causal_model.fitted_model
	coefs = abs(lr_model[target_idx]["model"].coef_)
	feat_names = []
	for var_idx, lag in predictors[target_idx]:
		feat_names.append(feature_names[var_idx] if lag == 0 else f"{feature_names[var_idx]} (t-{lag})")

	coefs = np.array(coefs)
	coefs /= coefs.max() + 1e-12  # normalize to [0,1]
	return feat_names, coefs



def tcn_feature_importance(tcn_model, feature_names):
	"""Approximate TCN input importance from first Conv1D layer weights."""
	conv1 = next((ly for ly in tcn_model.layers if isinstance(ly, Conv1D)), None)
	if conv1 is None:
		return [], []
	w = conv1.get_weights()[0]  # shape: (kernel, in_channels, out_filters)
	importance = np.mean(np.abs(w), axis=(0, 2))  # mean across kernel & filters
	return feature_names[:len(importance)], importance / (importance.max() + 1e-12)


# -------------------------- PLOTTING --------------------------
def plot_predictions(true, pred_causal, pred_tcn, nmae_causal, nstd_causal, nmae_tcn, nstd_tcn):
	"""Plot scatter of true vs predicted for Causal and TCN with NMAE/NSTD in titles."""
	fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
	vmin, vmax = min(true.min(), pred_causal.min(), pred_tcn.min()), max(true.max(), pred_causal.max(), pred_tcn.max())

	axes[0].scatter(true, pred_causal, alpha=0.5)
	axes[0].plot([vmin, vmax], [vmin, vmax], "k-")
	axes[0].set_title(f"Causal\nNMAE={nmae_causal:.4f}, NSTD={nstd_causal:.4f}")

	axes[1].scatter(true, pred_tcn, alpha=0.5, color="orange")
	axes[1].plot([vmin, vmax], [vmin, vmax], "k-")
	axes[1].set_title(f"Dilated-TCN\nNMAE={nmae_tcn:.4f}, NSTD={nstd_tcn:.4f}")

	for ax in axes:
		ax.set_xlabel("True Values")
		ax.set_ylabel("Predicted Values")
	plt.suptitle("Causal vs Dilated-TCN Predictions")
	plt.tight_layout()
	plt.show()


def plot_feature_importance(causal_feat_names, causal_importance, tcn_feat_names, tcn_importance, top_k=12):
	"""
	Plot barplots of feature importance for causal and TCN models in descending order.
	
	Parameters:
	- causal_feat_names: list of strings, feature names for causal model
	- causal_importance: array-like, normalized importance for causal model
	- tcn_feat_names: list of strings, feature names for TCN model
	- tcn_importance: array-like, normalized importance for TCN model
	- top_k: int, number of top features to show
	"""
	# Sort causal features by descending importance
	causal_idx_sorted = np.argsort(causal_importance)[::-1]
	causal_feat_names_sorted = [causal_feat_names[i] for i in causal_idx_sorted]
	causal_importance_sorted = causal_importance[causal_idx_sorted]

	# Sort TCN features by descending importance
	tcn_idx_sorted = np.argsort(tcn_importance)[::-1]
	tcn_feat_names_sorted = [tcn_feat_names[i] for i in tcn_idx_sorted]
	tcn_importance_sorted = tcn_importance[tcn_idx_sorted]

	# Keep only top_k features
	causal_feat_names_top = causal_feat_names_sorted[:top_k]
	causal_importance_top = causal_importance_sorted[:top_k]
	tcn_feat_names_top = tcn_feat_names_sorted[:top_k]
	tcn_importance_top = tcn_importance_sorted[:top_k]

	# Plot barplots
	fig, ax = plt.subplots(1, 2, figsize=(14, 6))
	sns.barplot(x=causal_importance_top, y=causal_feat_names_top, ax=ax[0], color="skyblue")
	ax[0].set_title("Causal: Normalized Regression Coefficients (Top Features)")
	sns.barplot(x=tcn_importance_top, y=tcn_feat_names_top, ax=ax[1], color="orange")
	ax[1].set_title("Dilated-TCN: Input Importance (Top Features)")
	plt.tight_layout()
	plt.show()



# -------------------------- MAIN --------------------------
def main():
	df_b1 = load_citylearn(1)
	df_b1 = variance_filter(df_b1)
	# print(df_b1.columns)
	print(df_b1.dtypes)
	tau_max = compute_tau_max(df_b1[TARGET].values)
	causal_model, predictors, pred_causal, true_causal, nmae_causal, nstd_causal = fit_causal_model(df_b1, TARGET, tau_max)

	# Prepare TCN data
	scaler = StandardScaler()
	data_scaled = scaler.fit_transform(df_b1.values)
	target_idx = df_b1.columns.get_loc(TARGET)
	X_all, y_all = create_sequences(data_scaled, target_col=target_idx, tau=tau_max)
	train_end = int(TRAIN_FRAC * len(df_b1))
	seq_test_start = train_end - tau_max
	X_train, y_train = X_all[:seq_test_start], y_all[:seq_test_start]
	X_test, y_test = X_all[seq_test_start:], y_all[seq_test_start:]
	y_test_aligned = y_test[-len(true_causal):]

	tcn_model, pred_tcn, nmae_tcn, nstd_tcn = fit_tcn(X_train, y_train, X_test, y_test_aligned)

	# Feature importances
	causal_feat_names, causal_importance = causal_feature_importance(causal_model, predictors, df_b1.columns, target_idx)
	tcn_feat_names, tcn_importance = tcn_feature_importance(tcn_model, df_b1.columns)

	# Plots
	plot_predictions(true_causal, pred_causal, pred_tcn, nmae_causal, nstd_causal, nmae_tcn, nstd_tcn)
	plot_feature_importance(causal_feat_names, causal_importance, tcn_feat_names, tcn_importance)


if __name__ == "__main__":
	main()

'''
['hour', 'day_type', 'indoor_dry_bulb_temperature',
       'average_unmet_cooling_setpoint_difference', 'indoor_relative_humidity',
       'non_shiftable_load', 'dhw_demand', 'cooling_demand',
       'solar_generation', 'occupant_count',
       'indoor_dry_bulb_temperature_set_point', 'carbon_intensity',
       'electricity_pricing', 'electricity_pricing_predicted_6h',
       'electricity_pricing_predicted_12h',
       'electricity_pricing_predicted_24h', 'outdoor_dry_bulb_temperature',
       'outdoor_relative_humidity', 'diffuse_solar_irradiance',
       'direct_solar_irradiance', 'outdoor_dry_bulb_temperature_predicted_6h',
       'outdoor_dry_bulb_temperature_predicted_12h',
       'outdoor_dry_bulb_temperature_predicted_24h',
       'outdoor_relative_humidity_predicted_6h',
       'outdoor_relative_humidity_predicted_12h',
       'outdoor_relative_humidity_predicted_24h',
       'diffuse_solar_irradiance_predicted_6h',
       'diffuse_solar_irradiance_predicted_12h',
       'diffuse_solar_irradiance_predicted_24h',
       'direct_solar_irradiance_predicted_6h',
       'direct_solar_irradiance_predicted_12h',
       'direct_solar_irradiance_predicted_24h'],
      dtype='object')
'''