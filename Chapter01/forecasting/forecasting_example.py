import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from prophet import Prophet
import kaggle

def download_kaggle_dataset(kaggle_dataset: str ="pratyushakar/rossmann-store-sales") -> None:
    api = kaggle.api
    print(api.get_config_value('username'))
    kaggle.api.dataset_download_files(kaggle_dataset, path="./", unzip=True, quiet=False)
    
def prep_store_data(df: pd.DataFrame, store_id: int = 4, store_open: int = 1) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns= {'Date': 'ds', 'Sales': 'y'}, inplace=True)
    df_store = df[
        (df['Store'] == store_id) &\
        (df['Open'] == store_open)
    ].reset_index(drop=True)
    return df_store.sort_values('ds', ascending=True)   
    
def plot_store_data(df: pd.DataFrame) -> None:
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(20,10))
    df.plot(x='ds', y='y', ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend(['Truth'])
    current_ytick_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_ytick_values])
    plt.savefig('store_data.png')
    

        
def train_predict(
    df: pd.DataFrame,
    train_fraction: float,
    seasonality: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:

    df['ds'] = pd.to_datetime(df['ds'])
    
    # grab split data
    train_index = int(train_fraction*df.shape[0])
    df_train = df.copy().iloc[0:train_index]
    df_test = df.copy().iloc[train_index:]

    # create Prophet model
    model=Prophet(
        yearly_seasonality=seasonality['yearly'],
        weekly_seasonality=seasonality['weekly'],
        daily_seasonality=seasonality['daily'],
        interval_width = 0.95
    )

    # train and predict
    model.fit(df_train)
    predicted = model.predict(df_test)
    return predicted, df_train, df_test, train_index

# Function to print and inspect data for debugging
def print_debug_info(df, name):
    print(f"Debug info for {name}:")
    print(df.head())
    print(df.dtypes)
    print(df.isna().sum())

def plot_forecast(df_train: pd.DataFrame, df_test: pd.DataFrame, predicted: pd.DataFrame) -> None:
    # Ensure correct dtypes
    df_train['ds'] = pd.to_datetime(df_train['ds'])
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')

    df_test['ds'] = pd.to_datetime(df_test['ds'])
    df_test['y'] = pd.to_numeric(df_test['y'], errors='coerce')

    predicted['ds'] = pd.to_datetime(predicted['ds'])
    predicted['yhat'] = pd.to_numeric(predicted['yhat'], errors='coerce')
    predicted['yhat_upper'] = pd.to_numeric(predicted['yhat_upper'], errors='coerce')
    predicted['yhat_lower'] = pd.to_numeric(predicted['yhat_lower'], errors='coerce')

    # Drop NaNs
    df_train.dropna(subset=['ds', 'y'], inplace=True)
    df_test.dropna(subset=['ds', 'y'], inplace=True)
    predicted.dropna(subset=['ds', 'yhat', 'yhat_upper', 'yhat_lower'], inplace=True)

    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot truth (test)
    df_test.plot(
        x='ds',
        y='y',
        ax=ax,
        label='Truth',
        linewidth=1,
        markersize=5,
        color='tab:blue',
        alpha=0.9,
        marker='o'
    )

    # Plot prediction line
    predicted.plot(
        x='ds',
        y='yhat',
        ax=ax,
        label='Prediction + 95% CI',
        linewidth=2,
        color='red'
    )

    # --- 🔐 Safe fill_between ---
    ds = pd.to_datetime(predicted['ds']).values
    y_upper = pd.to_numeric(predicted['yhat_upper'], errors='coerce').values
    y_lower = pd.to_numeric(predicted['yhat_lower'], errors='coerce').values

    # Create mask for finite values
    mask = np.isfinite(ds) & np.isfinite(y_upper) & np.isfinite(y_lower)
    if mask.sum() == 0:
        raise ValueError("No valid data points for fill_between after cleaning.")

    ax.fill_between(
        x=ds[mask],
        y1=y_upper[mask],
        y2=y_lower[mask],
        alpha=0.15,
        color='red'
    )
    # ----------------------------

    # Plot last 100 training points
    df_train_tail = df_train.tail(100)
    df_train_tail.plot(
        x='ds',
        y='y',
        ax=ax,
        color='tab:blue',
        label='_nolegend_',
        alpha=0.5,
        marker='o'
    )

    # Format labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    current_yticks = ax.get_yticks()
    ax.set_yticklabels(['{:,.0f}'.format(yt) for yt in current_yticks])

    plt.tight_layout()
    plt.savefig('store_data_forecast.png', dpi=150)
    plt.close()



if __name__ == "__main__":
    import os

    # If data present, read it in, otherwise, download it
    file_path = './train.csv'
    if os.path.exists(file_path):
        logging.info('Dataset found, reading into pandas dataframe.')
        df = pd.read_csv(file_path)
    else:
        logging.info('Dataset not found, downloading ...')
        download_kaggle_dataset()
        logging.info('Reading dataset into pandas dataframe.')
        df = pd.read_csv(file_path)

    # Transform dataset in preparation for feeding to Prophet
    df = prep_store_data(df)

    # Define main parameters for modelling
    seasonality = {
        'yearly': True,
        'weekly': True,
        'daily': False
    }

    # Calculate the relevant dataframes
    predicted, df_train, df_test, train_index = train_predict(
        df = df,
        train_fraction = 0.8,
        seasonality=seasonality
    )

    # Debugging
    # print(df_train.dtypes)
    # print(df_test.dtypes)
    # print(predicted.dtypes)

    # Plot the forecast
    plot_forecast(df_train, df_test, predicted)
        
    



