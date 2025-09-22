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
    from sklearn.metrics import root_mean_squared_error
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    return (
        DecisionTreeRegressor,
        LinearRegression,
        RandomForestRegressor,
        SimpleImputer,
        StandardScaler,
        load_dotenv,
        mlflow,
        np,
        pd,
        plt,
        root_mean_squared_error,
        sns,
        train_test_split,
    )


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
    return (weather_per_month,)


@app.cell
def _(plt, sns, weather, weather_per_month):
    # Visualize relationships in the data
    sns.lineplot(x="year", y="mean_temp", data=weather_per_month, errorbar=None)
    plt.show()
    sns.heatmap(weather.corr(), annot=True)
    plt.show()
    return


@app.cell
def _(weather):
    # Choose features, define the target, and drop null values
    feature_selection = ['month', 'cloud_cover', 'sunshine', 'precipitation', 'pressure', 'global_radiation']
    target_var = 'mean_temp'
    weather.dropna(subset=['mean_temp'], inplace=True) # inplace=True allows this to work without needing to reassign it
    return feature_selection, target_var


@app.cell
def _(feature_selection, target_var, weather):
    # Subset feature and target sets
    X = weather[feature_selection]    
    y = weather[target_var]
    return X, y


@app.cell
def _(np, y_train):
    print(np.isnan(y_train).sum())  # number of NaNs
    return


@app.cell
def _(SimpleImputer, StandardScaler, X, train_test_split, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    # Fit on the training data
    X_train = imputer.fit_transform(X_train)
    # Transform on the test data
    X_test  = imputer.transform(X_test)
    # Scale the data
    scaler = StandardScaler()
    # Fit on the training data
    X_train = scaler.fit_transform(X_train)
    # Transform on the test data
    X_test = scaler.transform(X_test) 
    return X_test, X_train, y_test, y_train


@app.cell
def _(
    DecisionTreeRegressor,
    LinearRegression,
    RandomForestRegressor,
    X_test,
    X_train,
    mlflow,
    root_mean_squared_error,
    y_test,
    y_train,
):
    # Predict, evaluate, and log the parameters and metrics of your models
    for idx, depth in enumerate([1, 2, 10]): 
        run_name = f"run_{idx}"
        with mlflow.start_run(run_name=run_name):
            # Create models
            lin_reg = LinearRegression().fit(X_train, y_train)
            tree_reg = DecisionTreeRegressor(random_state=42, max_depth=depth).fit(X_train, y_train)
            forest_reg = RandomForestRegressor(random_state=42, max_depth=depth).fit(X_train, y_train)
            # Log models
            mlflow.sklearn.log_model(lin_reg, name="lin_reg")
            mlflow.sklearn.log_model(tree_reg, name="tree_reg")
            mlflow.sklearn.log_model(forest_reg, name="forest_reg")
            # Evaluate performance
            y_pred_lin_reg = lin_reg.predict(X_test)
            lin_reg_rmse = root_mean_squared_error(y_test, y_pred_lin_reg)
            y_pred_tree_reg = tree_reg.predict(X_test)
            tree_reg_rmse = root_mean_squared_error(y_test, y_pred_tree_reg)
            y_pred_forest_reg = forest_reg.predict(X_test)
            forest_reg_rmse = root_mean_squared_error(y_test, y_pred_forest_reg)
            # Log performance
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("rmse_lr", lin_reg_rmse)
            mlflow.log_metric("rmse_tr", tree_reg_rmse)
            mlflow.log_metric("rmse_fr", forest_reg_rmse)
    return


@app.cell
def _(mlflow):
    # Search the runs for the experiment's results
    experiment_results = mlflow.search_runs()
    experiment_results
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
