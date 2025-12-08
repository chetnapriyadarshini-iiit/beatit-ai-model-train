"""Feature engineers the churn dataset."""
import os, sys
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
import boto3
import subprocess
from pathlib import Path

WHL_S3 = "s3://beatit-ai-common-artifact-bucket/beatit_ai_common/beatit_ai_common_utilities-0.1.0-py3-none-any.whl"
LOCAL_WHL = "/tmp/beatit_ai_common_utilities-0.1.0-py3-none-any.whl"


def install_whl_from_s3():
    s3 = boto3.client("s3")

    # parse s3://bucket/key into bucket + key
    assert WHL_S3.startswith("s3://")
    bucket, key = WHL_S3.replace("s3://", "").split("/", 1)

    print(f"Downloading wheel from {bucket}/{key} -> {LOCAL_WHL}")
    s3.download_file(bucket, key, LOCAL_WHL)

    print("Installing wheel...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", LOCAL_WHL])
    print("Wheel successfully installed.")

try:
    install_whl_from_s3()
except Exception as e:
    print("ERROR installing wheel:", e)
    raise

# ---- Now your original import works ----
import common.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw-data-dir", type=str, required=True)
    parser.add_argument("--train-output-dir", type=str, required=True)
    parser.add_argument("--val-output-dir", type=str, required=True)
    parser.add_argument("--test-output-dir", type=str, required=True)

    return parser.parse_args()


def load_raw_kkbox_data(raw_dir: str):
    """
    Assumes you have uploaded the KKBox churn CSVs directly under raw_dir:
    - train.csv
    - members.csv
    - transactions.csv
    - user_logs.csv
    """
    train_path = os.path.join(raw_dir, "train.csv")
    members_path = os.path.join(raw_dir, "members.csv")
    transactions_path = os.path.join(raw_dir, "transactions.csv")
    user_logs_path = os.path.join(raw_dir, "user_logs.csv")

    print(f"Loading train from {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"Loading members from {members_path}")
    members_df = pd.read_csv(members_path)

    print(f"Loading transactions from {transactions_path}")
    transactions_df = pd.read_csv(transactions_path)

    print(f"Loading user_logs from {user_logs_path}")
    user_logs_df = pd.read_csv(user_logs_path)

    return train_df, members_df, transactions_df, user_logs_df


def build_feature_table(train, members, transactions, user_logs):
    """
    Build feature table by joining train, members, transactions, and user_logs.

    Returns a single DataFrame with:
    - engineered features
    - 'is_churn' label column available for stratification
    """
    key = "msno"


    """################################### Members ##########################################"""

    members["gender"] = utils.get_fill_na_dataframe(members, "gender", value="others")
    gender_mapping = {"male": 0, "female": 1, "others": 2}
    members["gender"] = utils.get_label_encoding_dataframe(members, "gender", gender_mapping)

    members["registered_via"] = utils.get_convert_column_dtype(members, "registered_via", data_type="str")
    members["city"] = utils.get_convert_column_dtype(members, "city", data_type="str")
    members = utils.fix_time_in_df(
        members, "registration_init_time", expand=True
    )
    members["registration_init_time"] = utils.fix_time_in_df(members, "registration_init_time", expand=False)

    average_age = round(members["bd"].mean(), 0)
    condition = f"{average_age} if (x <= 0 or x > 100) else x"
    members["bd"] = utils.get_apply_condiiton_on_column(members, "bd", condition)

    """################ Transactions Feature Engineering ###############################"""

    transactions  = utils.fix_time_in_df(
        transactions, "transaction_date", expand=True
    )
    
    #print(transactions.columns)
    transactions = utils.fix_time_in_df(
        transactions, "membership_expire_date", expand=True
    )
    
    #print(transactions.columns[transactions.columns.duplicated()])

    transactions["discount"] = utils.get_two_column_operations(
        transactions, "plan_list_price", "actual_amount_paid", "-"
    )

    condition = "1 if x > 0 else 0"
    transactions["is_discount"] = utils.get_apply_condiiton_on_column(
        transactions, "discount", condition
    )

    transactions["amt_per_day"] = utils.get_two_column_operations(
        transactions, "actual_amount_paid", "payment_plan_days", "/"
    )
    transactions["amt_per_day"] = utils.get_replace_value_in_df(
        transactions, "amt_per_day", [np.inf, -np.inf], replace_with=0
    )

    transactions["membership_duration"] = utils.get_two_column_operations(
        transactions, "membership_expire_date", "transaction_date", "-"
    )
    transactions["membership_duration"] = utils.get_timedelta_division(
        transactions, "membership_duration", td_type="D"
    )
    transactions["membership_duration"] = utils.get_convert_column_dtype(
        transactions, "membership_duration", data_type="int"
    )

    condition = "1 if x > 30 else 0"
    transactions["more_than_30"] = utils.get_apply_condiiton_on_column(
        transactions, "membership_duration", condition
    )

    agg = {
        "payment_method_id": ["count", "nunique"],
        "payment_plan_days": ["mean", "nunique"],
        "plan_list_price": "mean",
        "actual_amount_paid": "mean",
        "is_auto_renew": ["mean", "max"],
        "transaction_date": ["min", "max", "count"],
        "membership_expire_date": "max",
        "is_cancel": ["mean", "max"],
        "discount": "mean",
        "is_discount": ["mean", "max"],
        "amt_per_day": "mean",
        "membership_duration": "mean",
        "more_than_30": "sum",
    }

    transactions_features = utils.get_groupby(
        transactions, by_column=key, agg_dict=agg, agg_func="mean", simple_agg_flag=False, reset_index=True
    )
    transactions_features.columns = (
        transactions_features.columns.get_level_values(0)
        + "_"
        + transactions_features.columns.get_level_values(1)
    )
    transactions_features.rename(
        columns={
            "msno_": "msno",
            "payment_plan_days_nunique": "change_in_plan",
            "payment_method_id_count": "total_payment_channels",
            "payment_method_id_nunique": "change_in_payment_methods",
            "is_cancel_max": "is_cancel_change_flag",
            "is_auto_renew_max": "is_autorenew_change_flag",
            "transaction_date_count": "total_transactions",
        },
        inplace=True,
    )


    ################################ User logs #########################################

    user_logs = utils.fix_time_in_df(user_logs, column_name="date", expand=True)

    user_logs_transformed = utils.get_fix_skew_with_log(
        user_logs,
        ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"],
        replace_inf=True,
        replace_inf_with=0,
    )

    user_logs_transformed_base = utils.get_groupby(
        user_logs_transformed,
        "msno",
        agg_dict=None,
        agg_func="mean",
        simple_agg_flag=True,
        reset_index=True,
    )

    agg_dict = {"date": ["count", "max"]}
    user_logs_transformed_dates = utils.get_groupby(
        user_logs_transformed,
        "msno",
        agg_dict=agg_dict,
        agg_func="mean",
        simple_agg_flag=False,
        reset_index=True,
    )
    user_logs_transformed_dates.columns = user_logs_transformed_dates.columns.droplevel()
    user_logs_transformed_dates.rename(
        columns={"count": "login_freq", "max": "last_login"}, inplace=True
    )
    user_logs_transformed_dates.reset_index(inplace=True)
    user_logs_transformed_dates.drop("index", inplace=True, axis=1)
    user_logs_transformed_dates.columns = ["msno", "login_freq", "last_login"]

    user_logs_final = utils.get_merge(user_logs_transformed_base, user_logs_transformed_dates, on=key)
    print(user_logs_final.columns)

    """ ########################### Final joins & features ################################# """

    train_df_v01 = utils.get_merge(members, train, on=key, axis=1, how="inner")
    train_df_v02 = utils.get_merge(train_df_v01, transactions_features, on=key, axis=1, how="inner")
    train_df_final = utils.get_merge(train_df_v02, user_logs_final, on=key, axis=1, how="inner")

    train_df_final["registration_duration"] = utils.get_two_column_operations(
        train_df_final, "membership_expire_date_max", "registration_init_time", "-"
    )
    train_df_final["registration_duration"] = utils.get_timedelta_division(
        train_df_final, "registration_duration", td_type="D"
    )
    train_df_final["registration_duration"] = utils.get_convert_column_dtype(
        train_df_final, "registration_duration", data_type="int"
    )
    #
    date_cols = [
        "membership_expire_date_max",
        "last_login",
        "transaction_date_min",
        "transaction_date_max"]

    train_df_final = utils.transform_date_cols_for_xgboost(train_df_final, date_cols)
    train_df_final = train_df_final.drop(["registration_init_time", "msno", "date"], axis=1)
    train_df_final = train_df_final.drop(date_cols, axis=1)
    print(train_df_final.columns)

    # Optional: write a silver Parquet snapshot directly to S3
    # Be sure pyarrow/fastparquet + s3fs are installed if you keep this.
    # s3_parquet_path = "s3://beatit-ai-data/data-engineering/silver/train_df_final.parquet"
    # train_df_final.to_parquet(s3_parquet_path, index=False)

    # Ensure label and sensitive feature have fixed positionsS
    label_col = "is_churn"
    sensitive_col = "registered_via"

    # Start from all columns
    cols = list(train_df_final.columns)

    # Remove label and sensitive feature from the list (if present)
    cols = [c for c in cols if c not in [label_col, sensitive_col]]

    # Final column order:
    # 0: is_churn (label)
    # 1: registered_via (sensitive feature)
    # 2..N: rest of features
    train_df_final = train_df_final[[label_col, sensitive_col] + cols]

    return train_df_final


def split_and_save(df, train_dir, val_dir, test_dir, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits df into train/val/test and writes CSVs without header
    (to match the current pipeline's DatasetFormat configuration).
    """
    # First split off test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["is_churn"]
    )

    # Now split train+val
    val_fraction = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        random_state=random_state,
        stratify=train_val_df["is_churn"],
    )

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_path = os.path.join(train_dir, "train.csv")
    val_path = os.path.join(val_dir, "validation.csv")
    test_path = os.path.join(test_dir, "test.csv")

    # NOTE: header=False to match DatasetFormat.csv(header=False)
    train_df.to_csv(train_path, index=False, header=False)
    val_df.to_csv(val_path, index=False, header=False)
    test_df.to_csv(test_path, index=False, header=False)

    print(f"Saved train to {train_path}, shape={train_df.shape}")
    print(f"Saved val to {val_path}, shape={val_df.shape}")
    print(f"Saved test to {test_path}, shape={test_df.shape}")


def main():
    args = parse_args()

    print("Arguments:", args)

    train_df, members_df, transactions_df, user_logs_df = load_raw_kkbox_data(args.raw_data_dir)
    df = build_feature_table(train_df, members_df, transactions_df, user_logs_df)

    split_and_save(
        df,
        train_dir=args.train_output_dir,
        val_dir=args.val_output_dir,
        test_dir=args.test_output_dir,
    )


if __name__ == "__main__":
    main()
