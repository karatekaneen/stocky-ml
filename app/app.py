# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name

from typing import List
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# import tensorflow as tf

# Load saved model
# model = tf.keras.models.load_model("../out/model")

# Load saved scaler
scaler = joblib.load("model/scaler/scaler.pkl")

unwanted_columns = [
    "Unnamed: 0",
    # These values does not carry much importance
    "volume_omx",
    "owners_omx",
    "date",
    "high_stock",
    "low_stock",
    "owners_stock",
    # 'close_stock',
    "open_stock",
    "high_omx",
    "low_omx",
    "owners_omx",
    "close_omx",
    "open_omx",
    # Theses values are missing a lot of the time and would result in a lot of rows being dropped.
    # TODO: See if you can improve the data quality to be able to use more of these
    "owners_stock",
    "zs-20-volume_omx",
    "ma-slope-20-volume_omx",
    "std-slope-20-volume_omx",
    "percent-rng-20-volume_omx",
    "percent-std-20-volume_omx",
    "avg-log-volume-20_omx",
    "zs-50-volume_omx",
    "ma-slope-50-volume_omx",
    "std-slope-50-volume_omx",
    "percent-rng-50-volume_omx",
    "percent-std-50-volume_omx",
    "avg-log-volume-50_omx",
    "zs-100-volume_omx",
    "ma-slope-100-volume_omx",
    "std-slope-100-volume_omx",
    "percent-rng-100-volume_omx",
    "percent-std-100-volume_omx",
    "avg-log-volume-100_omx",
    "zs-200-volume_omx",
    "ma-slope-200-volume_omx",
    "std-slope-200-volume_omx",
    "percent-rng-200-volume_omx",
    "percent-std-200-volume_omx",
    "avg-log-volume-200_omx",
    "zs-200-volume_stock",
    "ma-slope-200-volume_stock",
    "std-slope-200-volume_stock",
    "percent-rng-200-volume_stock",
    "percent-std-200-volume_stock",
    "zs-100-volume_stock",
    "ma-slope-100-volume_stock",
    "std-slope-100-volume_stock",
    "percent-rng-100-volume_stock",
    "percent-std-100-volume_stock",
    "volume_stock",
    "volume_cash_omx",
]


app = Flask(__name__)


@app.route("/mirror", methods=["POST", "GET"])
def mirror():
    print(request.json)
    data = request.json
    return data, 200


@app.route("/predict", methods=["POST", "GET"])
def predict():
    data = request.json

    print(data)
    try:
        validate_input(data["omx_data"], 201)
        validate_input(data["stock_data"], 201)
    except ValueError as err:
        print(f"Error")
        # print(f"Error: {str(err)}")
        return str(err), 400

    v_merged = parse_data(data)

    # Grab the last row
    original_row = v_merged.tail(1).iloc[-1:]

    # Scale the validation data with the same scaler used for the training data
    original_x = scaler.transform(original_row.values)

    return original_x.tolist()  # jsonify({"scaled": original_x})

    # # Run predictions on the validation dataset
    # original_pred = model.predict(original_x)

    # return jsonify({"prediciton": original_pred[0][0].astype("float64")}), 200


def parse_data(data) -> pd.DataFrame:
    omx_df = pd.DataFrame(data["omx_data"])
    stock_df = pd.DataFrame(data["stock_data"])

    stock_df = add_calculated_columns(stock_df)
    omx_df = add_calculated_columns(omx_df)

    merged_df = pd.merge(omx_df, stock_df, on="date", suffixes=("_stock", "_omx"))
    merged_df = merged_df.assign(
        trades_this_year=data["trades_this_year"],
        days_since_last_trade=data["days_since_last_trade"],
    )
    merged_df = pd.merge(merged_df, merge_index_stock_df(omx_df, stock_df), on="date")
    merged_df.drop(columns=unwanted_columns, inplace=True, errors="ignore")

    return merged_df


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.DataFrame:
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def merge_index_stock_df(omxdf: pd.DataFrame, stockdf: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(
        stockdf[["date", "close"]],
        omxdf[["date", "close"]],
        on="date",
        suffixes=("_stock", "_omx"),
    )
    # df['date'] = df['date_stock']
    df["stock_quota"] = df["close_stock"] / df["close_omx"]

    df["stock_hist_relative_perf20"] = df["stock_quota"].shift(20) / df["stock_quota"]
    df["stock_hist_relative_perf50"] = df["stock_quota"].shift(50) / df["stock_quota"]
    df["stock_hist_relative_perf100"] = df["stock_quota"].shift(100) / df["stock_quota"]

    df["stock_hist_perf20"] = df["close_stock"].shift(20) / df["close_stock"]
    df["stock_hist_perf50"] = df["close_stock"].shift(50) / df["close_stock"]
    df["stock_hist_perf100"] = df["close_stock"].shift(100) / df["close_stock"]

    for p in [3, 10, 34, 100]:
        df[f"stock_relative_rsi_{p}"] = calculate_rsi(df["stock_quota"], p) / 100
        df[f"stock_rsi_{p}"] = calculate_rsi(df["close_stock"], p) / 100

    df.drop(columns=["close_stock", "close_omx"], inplace=True)

    return df


# pylint: disable=cell-var-from-loop
def add_calculated_columns(price: pd.DataFrame) -> pd.DataFrame:
    lookbacks = [20, 50, 100, 200]
    values = ["close", "volume"]

    price["volume_cash"] = round(
        price["volume"]
        * (price["close"] * 2 + price["open"] * 2 + price["low"] + price["high"])
        / 6
    )
    for value in values:
        for lookback in lookbacks:
            # Get the rolling average and std:
            price["average"] = price[value].rolling(lookback).mean()
            price["std"] = price[value].rolling(lookback).std()
            high = price["high"].rolling(lookback).max()
            low = price["low"].rolling(lookback).min()

            # Normalize distance to mean. This could be done with the data above but dont know how.
            price[f"zs-{lookback}-{value}"] = (price[value] - price["average"]) / price[
                "std"
            ]

            # Get slope of rolling average and std
            price[f"ma-slope-{lookback}-{value}"] = price["average"] / price[
                "average"
            ].shift(1)
            price[f"std-slope-{lookback}-{value}"] = price["std"] / price["std"].shift(
                1
            )

            # Get range
            price[f"rng-{lookback}"] = high / low
            price[f"percent-rng-{lookback}-{value}"] = (high / low) / price[value]
            price[f"percent-std-{lookback}-{value}"] = price["std"] / price[value]

            if value == "volume":
                price["temp_volume"] = round(
                    price["volume"]
                    * (
                        price["close"] * 2
                        + price["open"] * 2
                        + price["low"]
                        + price["high"]
                    )
                    / 6
                )
                price.drop(columns=["temp_volume"], inplace=True)
            else:
                # Hehe, so bad code
                # apply kaufmanns_efficiency_ratio function to a rolling window of the close column
                price[f"kaufmanns_efficiency_ratio-{lookback}"] = (
                    price["close"]
                    .rolling(window=lookback)
                    .apply(lambda x: kaufmanns_efficiency_ratio(x.tolist(), lookback))
                )

            # Drop the actual values since they carry no interest:
            price.drop(columns=["average", "std"], inplace=True)

        # TODO: Add calculations for volume
        # TODO: Add calculations for owners
    return price


def kaufmanns_efficiency_ratio(prices, lookback):
    """
    Calculates Kaufmann's Efficiency Ratio over a lookback of n.
    :param prices: list of prices
    :param n: lookback period
    :return: Kaufmann's Efficiency Ratio
    """
    change = abs(prices[len(prices) - 1] - prices[0])
    volatility = sum(abs(prices[i] - prices[i - 1]) for i in range(1, lookback))
    return change / volatility if volatility != 0 else 0


def validate_input(lst: List, wanted_length: int) -> None:
    if len(lst) != wanted_length:
        # TODO: Fix this to show expected length
        raise ValueError("The length is not what is expected")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
