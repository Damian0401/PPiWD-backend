import json

import pandas as pd

from preprocessing import AXES, MEASUREMENT_TYPES, RESAMPLE_FREQ


def extract(row):
    data = json.loads(row["data"])
    prefix = MEASUREMENT_TYPES[row["type_id"]]
    return pd.Series({f"{axis}_{prefix}": data[axis] for axis in AXES})


def marge_raw_data_to_dataframe_with_acc_gyro(raw_df):
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], format="mixed")
    acc_df = raw_df[raw_df["type_id"] == 14].copy()
    gyro_df = raw_df[raw_df["type_id"] == 16].copy()

    acc_df = pd.concat(
        [acc_df[["device_id", "timestamp"]], acc_df.apply(extract, axis=1)], axis=1
    )
    acc_df = acc_df.loc[:, ~acc_df.columns.duplicated()]
    gyro_df = pd.concat(
        [gyro_df[["device_id", "timestamp"]], gyro_df.apply(extract, axis=1)], axis=1
    )
    gyro_df = gyro_df.loc[:, ~gyro_df.columns.duplicated()]

    acc_df = acc_df.sort_values("timestamp")
    gyro_df = gyro_df.sort_values("timestamp")

    merged = pd.merge_asof(
        acc_df,
        gyro_df,
        on="timestamp",
        by="device_id",
        direction="nearest",
        tolerance=pd.Timedelta(RESAMPLE_FREQ),
    )
    return merged
