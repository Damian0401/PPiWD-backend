import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq
from scipy.stats import entropy, kurtosis, skew
from tqdm import tqdm

from preprocessing import AXES, MEASUREMENT_TYPES

WINDOW_SIZE = 64
STEP_SIZE = 32
FS = 2


def zero_crossings(series):
    return ((series[:-1] * series[1:]) < 0).sum()


def compute_entropy(series, bins=10):
    hist, _ = np.histogram(series, bins=bins, density=True)
    hist += 1e-12
    return entropy(hist)


# Mean Absolute Deviation
def mad(arr):
    mean = np.mean(arr)
    return np.mean(np.abs(arr - mean))


# simple_moving_average
def sma(arr):
    return np.convolve(arr, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode="valid")


def extract_features(df):
    df_feats = []

    # To jak by sie okazało że df jest krótszy niż WINDOW_SIZE, to go uzupełnia zerami
    if len(df) < WINDOW_SIZE:
        pad_length = WINDOW_SIZE - len(df)
        df = pd.concat(
            [df, pd.DataFrame(np.zeros((pad_length, df.shape[1])), columns=df.columns)],
            ignore_index=True,
        )

    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE + 1
    with tqdm(total=total_windows, desc="Processing features") as pbar:
        for window_start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
            window = df[window_start : window_start + WINDOW_SIZE]

            row = {"timestamp": window.index[0]}

            for sensor_name in MEASUREMENT_TYPES.values():
                sensor_df = window[
                    [f"x_{sensor_name}", f"y_{sensor_name}", f"z_{sensor_name}"]
                ]
                for axis in AXES:
                    series = sensor_df[f"{axis}_{sensor_name}"]

                    row[f"tBody{sensor_name}-mean()-{axis.capitalize()}"] = np.mean(
                        series
                    )
                    row[f"tBody{sensor_name}-std()-{axis.capitalize()}"] = np.std(
                        series
                    )
                    row[f"tBody{sensor_name}-min()-{axis.capitalize()}"] = np.min(
                        series
                    )
                    row[f"tBody{sensor_name}-max()-{axis.capitalize()}"] = np.max(
                        series
                    )
                    row[f"tBody{sensor_name}-mad()-{axis.capitalize()}"] = mad(series)
                    row[f"tBody{sensor_name}-rms()-{axis.capitalize()}"] = np.sqrt(
                        np.mean(series**2)
                    )
                    row[f"tBody{sensor_name}-median()-{axis.capitalize()}"] = np.median(
                        series
                    )
                    row[f"tBody{sensor_name}-iqr()-{axis.capitalize()}"] = (
                        np.percentile(series, 75) - np.percentile(series, 25)
                    )
                    row[f"tBody{sensor_name}-zeroCross()-{axis.capitalize()}"] = (
                        zero_crossings(series)
                    )
                    row[f"tBody{sensor_name}-entropy()-{axis.capitalize()}"] = (
                        compute_entropy(series)
                    )
                    row[f"tBody{sensor_name}-energy()-{axis.capitalize()}"] = np.sum(
                        series**2
                    ) / len(series)

                    jerk = np.diff(series) * FS
                    row[f"tBody{sensor_name}Jerk-mean()-{axis.capitalize()}"] = np.mean(
                        jerk
                    )
                    row[f"tBody{sensor_name}Jerk-std()-{axis.capitalize()}"] = np.std(
                        jerk
                    )
                    row[f"tBody{sensor_name}Jerk-mad()-{axis.capitalize()}"] = mad(jerk)
                    row[f"tBody{sensor_name}Jerk-max()-{axis.capitalize()}"] = np.max(
                        jerk
                    )
                    row[f"tBody{sensor_name}Jerk-min()-{axis.capitalize()}"] = np.min(
                        jerk
                    )
                    row[f"tBody{sensor_name}Jerk-energy()-{axis.capitalize()}"] = (
                        np.sum(jerk**2)
                    )
                    row[f"tBody{sensor_name}Jerk-iqr()-{axis.capitalize()}"] = (
                        np.percentile(jerk, 75) - np.percentile(jerk, 25)
                    )
                    row[f"tBody{sensor_name}Jerk-entropy()-{axis.capitalize()}"] = (
                        compute_entropy(jerk)
                    )

                    fft_vals = np.abs(fft(series))
                    fft_freqs = fftfreq(len(series), d=1 / FS)
                    fft_vals = fft_vals[1 : len(fft_vals) // 2]
                    fft_freqs = fft_freqs[1 : len(fft_freqs) // 2]

                    if len(fft_vals) > 0:
                        row[f"fBody{sensor_name}-mean()-{axis.capitalize()}"] = np.mean(
                            fft_vals
                        )
                        row[f"fBody{sensor_name}-std()-{axis.capitalize()}"] = np.std(
                            fft_vals
                        )
                        row[f"fBody{sensor_name}-mad()-{axis.capitalize()}"] = mad(
                            fft_vals
                        )
                        row[f"fBody{sensor_name}-max()-{axis.capitalize()}"] = np.max(
                            fft_vals
                        )
                        row[f"fBody{sensor_name}-min()-{axis.capitalize()}"] = np.min(
                            fft_vals
                        )
                        row[f"fBody{sensor_name}-iqr()-{axis.capitalize()}"] = (
                            np.percentile(fft_vals, 75) - np.percentile(fft_vals, 25)
                        )
                        row[f"fBody{sensor_name}-entropy()-{axis.capitalize()}"] = (
                            entropy(fft_vals / np.sum(fft_vals))
                        )
                        row[f"fBody{sensor_name}-energy()-{axis.capitalize()}"] = (
                            np.sum(fft_vals**2)
                        )
                        row[f"fBody{sensor_name}-skewness()-{axis.capitalize()}"] = (
                            skew(fft_vals)
                        )
                        row[f"fBody{sensor_name}-kurtosis()-{axis.capitalize()}"] = (
                            kurtosis(fft_vals)
                        )
                        row[f"fBody{sensor_name}-maxInds()-{axis.capitalize()}"] = (
                            fft_freqs[np.argmax(fft_vals)]
                        )
                        row[f"fBody{sensor_name}-meanFreq()-{axis.capitalize()}"] = (
                            np.sum(fft_freqs * fft_vals) / np.sum(fft_vals)
                        )
                    else:
                        for feature in [
                            "mean",
                            "std",
                            "mad",
                            "max",
                            "min",
                            "iqr",
                            "entropy",
                            "energy",
                            "skew",
                            "kurtosis",
                            "maxInds",
                            "meanFreq",
                        ]:
                            row[
                                f"fBody{sensor_name}-{feature}()-{axis.capitalize()}"
                            ] = 0

                mag = np.linalg.norm(
                    window[[f"{x}_{sensor_name}" for x in AXES]], axis=1
                )
                row[f"tBody{sensor_name}Mag-mean()"] = np.mean(mag)
                row[f"tBody{sensor_name}Mag-std()"] = np.std(mag)
                row[f"tBody{sensor_name}Mag-mad()"] = mad(mag)
                row[f"tBody{sensor_name}Mag-min()"] = np.min(mag)
                row[f"tBody{sensor_name}Mag-max()"] = np.max(mag)
                row[f"tBody{sensor_name}Mag-energy()"] = np.sum(mag**2)
                row[f"tBody{sensor_name}Mag-entropy()"] = compute_entropy(mag)

                sma_tbody = (
                    np.sum(
                        np.abs(sensor_df[f"x_{sensor_name}"])
                        + np.abs(sensor_df[f"y_{sensor_name}"])
                        + np.abs(sensor_df[f"z_{sensor_name}"])
                    )
                    / WINDOW_SIZE
                )
                row[f"tBody{sensor_name}-sma()"] = sma_tbody

                jerk_xyz = np.diff(sensor_df, axis=0) * FS
                if jerk_xyz.shape[0] > 1:
                    sma_jerk = (
                        np.sum(
                            np.abs(jerk_xyz[:, 0])
                            + np.abs(jerk_xyz[:, 1])
                            + np.abs(jerk_xyz[:, 2])
                        )
                        / jerk_xyz.shape[0]
                    )
                else:
                    sma_jerk = 0
                row[f"tBody{sensor_name}Jerk-sma()"] = sma_jerk

                if jerk_xyz.shape[0] > 1:
                    fft_jx = np.abs(fft(jerk_xyz[:, 0]))[1 : len(jerk_xyz) // 2]
                    fft_jy = np.abs(fft(jerk_xyz[:, 1]))[1 : len(jerk_xyz) // 2]
                    fft_jz = np.abs(fft(jerk_xyz[:, 2]))[1 : len(jerk_xyz) // 2]
                    sma_fft_jerk = np.sum(fft_jx + fft_jy + fft_jz) / len(fft_jx)
                else:
                    sma_fft_jerk = 0
                row[f"fBody{sensor_name}Jerk-sma()"] = sma_fft_jerk

                fft_x = np.abs(fft(sensor_df[f"x_{sensor_name}"]))[1 : WINDOW_SIZE // 2]
                fft_y = np.abs(fft(sensor_df[f"y_{sensor_name}"]))[1 : WINDOW_SIZE // 2]
                fft_z = np.abs(fft(sensor_df[f"z_{sensor_name}"]))[1 : WINDOW_SIZE // 2]
                sma_fft = np.sum(fft_x + fft_y + fft_z) / len(fft_x)
                row[f"fBody{sensor_name}-sma()"] = sma_fft

                # Correlation-based features between axes
                row[f"tBody{sensor_name}-correlation()-X,Y"] = np.corrcoef(
                    sensor_df[f"x_{sensor_name}"], sensor_df[f"y_{sensor_name}"]
                )[0, 1]
                row[f"tBody{sensor_name}-correlation()-X,Z"] = np.corrcoef(
                    sensor_df[f"x_{sensor_name}"], sensor_df[f"z_{sensor_name}"]
                )[0, 1]
                row[f"tBody{sensor_name}-correlation()-X,Z"] = np.corrcoef(
                    sensor_df[f"y_{sensor_name}"], sensor_df[f"z_{sensor_name}"]
                )[0, 1]

                if jerk_xyz.shape[0] > 1:
                    row[f"tBody{sensor_name}Jerk-correlation()-X,Y"] = np.corrcoef(
                        jerk_xyz[:, 0], jerk_xyz[:, 1]
                    )[0, 1]
                    row[f"tBody{sensor_name}Jerk-correlation()-X,Z"] = np.corrcoef(
                        jerk_xyz[:, 0], jerk_xyz[:, 2]
                    )[0, 1]
                    row[f"tBody{sensor_name}Jerk-correlation()-Y,Z"] = np.corrcoef(
                        jerk_xyz[:, 1], jerk_xyz[:, 2]
                    )[0, 1]
                else:
                    row[f"tBody{sensor_name}Jerk-correlation()-X,Y"] = 0
                    row[f"tBody{sensor_name}Jerk-correlation()-X,Z"] = 0
                    row[f"tBody{sensor_name}Jerk-correlation()-Y,Z"] = 0

            df_feats.append(row)
            pbar.update(1)
    return pd.DataFrame(df_feats)
