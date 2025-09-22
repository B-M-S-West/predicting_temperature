import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.sklearn
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    return load_dotenv, mlflow, pd


@app.cell
def _(load_dotenv):
    load_dotenv()
    return


@app.cell
def _(mlflow):
    # Test logging to verify connection
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    with mlflow.start_run():
        print("✓ Successfully connected to MLflow!")
    return


@app.cell
def _(pd):
    # Read in the data
    weather = pd.read_csv("london_weather.csv")
    return (weather,)


@app.cell
def _(weather):
    # Give a summary of column names, count of values and data types
    weather.info()
    return


@app.cell
def _(pd, weather):
    # Convert data and extract information
    weather["date"] = pd.to_datetime(weather["date"], format="%Y%m%d")
    weather["year"] = weather["date"].dt.year
    weather['month'] = weather['date'].dt.month
    return


@app.cell
def _(weather):
    # Aggregate and calculate average metrics
    weather_metrics = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'precipitation', 'pressure', 'snow_depth']
    weather_per_month = weather.groupby(['year', 'month'], as_index = False)[weather_metrics].mean()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
