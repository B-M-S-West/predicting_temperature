# London Weather Temperature Prediction

A machine learning project that predicts mean temperature using historical London weather data. This project implements and compares multiple regression models (Linear Regression, Decision Tree, and Random Forest) to forecast temperature based on various meteorological features.

## 🌟 Features

- **Data Analysis**: Comprehensive exploration of London weather patterns from 1979 onwards
- **Multiple ML Models**: Comparison of Linear Regression, Decision Tree, and Random Forest algorithms
- **MLflow Integration**: Experiment tracking and model management using MLflow
- **Interactive Analysis**: Built with Marimo for interactive data science workflows
- **Model Evaluation**: RMSE-based performance comparison across different model configurations

## 📊 Dataset

The project uses the `london_weather.csv` dataset containing daily weather observations with the following features:

- **Date**: Daily records from 1979 onwards
- **Weather Metrics**:
  - `cloud_cover`: Cloud coverage percentage
  - `sunshine`: Hours of sunshine
  - `global_radiation`: Solar radiation measurements
  - `max_temp`, `mean_temp`, `min_temp`: Daily temperature readings (°C)
  - `precipitation`: Rainfall amount
  - `pressure`: Atmospheric pressure
  - `snow_depth`: Snow depth measurements

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd predicting_temperature
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables** (optional):
   Create a `.env` file in the project root to configure MLflow settings:
   ```env
    DATABRICKS_TOKEN=<yourdatabrickstoken>
    DATABRICKS_HOST=<yourdatabrickhosturl>
    MLFLOW_TRACKING_URI=<yourdatabricksmlflowtrackinguri>
    MLFLOW_REGISTRY_URI=<yourdatabricksmlflowregistryuri>
    MLFLOW_EXPERIMENT_ID=<yourexperimentid>
    MLFLOW_LOCK_MODEL_DEPENDENCIES=true #Trying to use uv for dependencies
   ```

## 🚀 Usage

### Running the Analysis

1. **Start the Marimo notebook**:
   ```bash
   uv run marimo run main.py
   ```

2. **Access the interactive interface**: 
   Open your browser to the provided localhost URL to interact with the analysis

3. **Run all cells sequentially** to:
   - Load and explore the weather data
   - Visualize temperature trends and correlations
   - Train multiple regression models
   - Compare model performance
   - Track experiments with MLflow

### Key Analysis Steps

1. **Data Preprocessing**:
   - Date parsing and feature extraction (year, month)
   - Missing value imputation using mean strategy
   - Feature standardization with StandardScaler

2. **Feature Engineering**:
   - Selected features: `month`, `cloud_cover`, `sunshine`, `precipitation`, `pressure`, `global_radiation`
   - Target variable: `mean_temp`
   - Monthly aggregation of weather metrics

3. **Model Training**:
   - **Linear Regression**: Baseline linear model
   - **Decision Tree**: Non-linear model with varying max_depth (1, 2, 10)
   - **Random Forest**: Ensemble method with varying max_depth (1, 2, 10)

4. **Evaluation**:
   - Root Mean Square Error (RMSE) for model comparison
   - MLflow tracking in databricks for experiment management

## 📈 Results

The project compares model performance using RMSE metrics across different configurations:

- **Linear Regression**: Provides baseline performance
- **Decision Tree**: Performance varies with tree depth
- **Random Forest**: Generally better performance due to ensemble approach

Results are automatically logged to MLflow for easy comparison and model selection.

## 🔧 Dependencies

Core dependencies include:

- **marimo** (≥0.16.0): Interactive notebook environment
- **mlflow[databricks]** (≥3.0.1): Experiment tracking and model management
- **python-dotenv** (≥1.1.1): Environment variable management
- **seaborn** (≥0.13.2): Statistical data visualization
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **numpy**: Numerical computing

## 📁 Project Structure

```
predicting_temperature/
├── main.py                 # Main Marimo notebook with analysis
├── london_weather.csv      # Weather dataset
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock               # Dependency lock file
├── README.md             # Project documentation
└── __marimo__/           # Marimo session data
    └── session/
        └── main.py.json
```

## 🔬 Model Performance

The project evaluates three model types across different hyperparameter configurations:

1. **Linear Regression**: Consistent baseline performance
2. **Decision Tree Regressor**: 
   - `max_depth=1`: Simple model, may underfit
   - `max_depth=2`: Moderate complexity
   - `max_depth=10`: Higher complexity, risk of overfitting
3. **Random Forest Regressor**: Same depth variations as Decision Tree but with ensemble benefits

All models are evaluated using train-test split (67%/33%) with RMSE as the primary metric.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Links

- [Marimo Documentation](https://docs.marimo.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

**Note**: This project is designed for educational and research purposes to demonstrate machine learning workflows for weather prediction tasks.
