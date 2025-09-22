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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
