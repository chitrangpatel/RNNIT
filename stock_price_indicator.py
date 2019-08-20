#! /usr/bin/env python
# coding: utf-8

from yahoo_historical import Fetcher
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation


def extract_data(tickers):
    """
    Extract data from yahoo finance for any given ticker from 2009 to 2019.
    
    Parameters
    ----------
    tickers: list of ticker symbols
    
    Returns
    -------
    dfs: list of dataframes containing the Open, High, 
         Low, Close, Adj Close and Volume for each ticker
         for the duration mentioned above.
    """
    dfs = []
    for t in tickers:
        data = Fetcher(t, [2009,1,1], [2019,1,1])
        data = data.getHistorical()
        data["Date"] = pd.to_datetime(data["Date"])
        data = data[["Adj Close", "Date"]]
        data = data.set_index('Date').dropna(axis=1)
        dfs.append(data)
    return dfs


def get_features_target(df, days_in_future):
    """
    Split the data into Features and Target Dataframes.
    Target is the Adj Closing price -- x days in the future.
    
    Parameters
    ----------
    df: Data frame to split into target and features
    days_in_future: The number of days
                    in the future you want to predict prices for.
    Returns
    -------
    features: dataframe of features to be used to train the model.
    target: dataframe of the target values that the model wants to predict.
    """
    target = df[["Adj Close"]][days_in_future:]
    features = df
    features = features.shift(days_in_future).dropna()
    return features, target


def min_max_scale_features(input_features):
    """
    Apply Scikit Learn's Min-Max Scaler to the input data.
    Parameters
    ----------
    input_features: input dataframe to scale
    
    Returns
    -------
    output_features: Scaled dataframe 
    """
    min_max_scaler = MinMaxScaler()
    output_features = pd.DataFrame(min_max_scaler.fit_transform(input_features),
                            columns=input_features.columns,
                            index=input_features.index)
    return output_features

def mean_scale_features(input_features):
    """
    This scaling is a customized scaling where I:
    1) subtract the mean of the dataset.
    2) divide by the standard deviation of the mean subtracted dataset.
    Parameters
    ----------
    input_features: input dataframe to scale
    
    Returns
    -------
    scaled_input_features: Scaled dataframe
    scaled_means: Dataframe of the mean values that were subtracted
    scaled_stds: Dataframe of the standard deviation calculated for the input array.
                 The scaled_means and scaled_stds can be used to unscale the data.
    """
    scaled_means = input_features.mean(axis=0)
    scaled_input_features = input_features.subtract(input_features.mean(axis=0), axis=1)
    scaled_stds = scaled_input_features.std(axis=0)
    scaled_input_features = scaled_input_features.divide(scaled_input_features.std(axis=0), axis=1)
    return scaled_input_features, scaled_means, scaled_stds

def unscale_target(input_target, scaled_mean, scaled_stds):
    """
    Unscale the data to original levels using the standard deviation and mean
    statistics from the original dataset. The scaled_mean and scaled_stds will
    be used to unscale the data.

    Parameters
    ----------
    input_target: input array to unscale back to the original levels.
    scaled_mean: mean value that was subtracted from the original dataset
    scaled_stds: standard deviation calculated for the original dataset.

    Returns
    -------
    input_target: unscaled array
    """
    input_target = input_target*scaled_stds
    input_target += scaled_mean
    return input_target


def split_train_test(features, target, split_date='2017-05-31'):
    """
    Split the features and target into training and testing sets.

    Parameters
    ----------
    features: dataframe of feature values
    target: dataframe of target values
    split_date: the date about which we want to split the data.

    Returns
    -------
    features_train: training set of features
    features_test: testing set of features.
    target_train: training set of target values.
    target_test: testing set of target values.
    """
    features_train = features[features.index <= split_date]
    features_test = features[features.index > split_date]
    target_train = target[target.index <= split_date]
    target_test = target[target.index > split_date]
    return features_train, features_test, target_train, target_test


def make_features_array(df):
    """
    Generate an array from features dataframe to be used by the LSTM model.

    Parameters
    ----------
    df: dataframe to convert to array

    Yields
    ------
    data_array
    """
    data_array = df.values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-1), range(1, num_elements)):
        yield data_array[start:stop, :]


def make_target_array(df):
    """
    Generate an array from target dataframe.

    Parameters
    ----------
    df: dataframe to convert to array

    Returns
    ------
    data_array
    """
    data_array = df.values
    num_elements = data_array.shape[0]
    return data_array[1:num_elements]


def explore_data(dfs, tickers):
    """
    plot the Adj Closing prices of each ticker
    as a subplot.

    Parameters
    ----------
    dfs: list of dataframes to explore
    tickers: list of ticker labels.

    """
    figure = plt.figure(figsize=(15, 15))
    figure.suptitle("Adj Close Prices Exploration", fontsize=24)
    for i, df in enumerate(dfs):
        plt.subplot(len(dfs)//2+1, 2, i+1)
        df['Adj Close'].plot(color='black')
        plt.ylabel(tickers[i])
        plt.grid()
    plt.savefig("adj_closing_prices.png", dpi=150)


def plot_values(output):
    """
    plot the predicted and actual target values "Adj closing prices"
    as a subplot for each ticker.

    Parameters
    ----------
    output: dictionary of each ticker's predicted and actual target values.
    """
    figure = plt.figure(figsize=(15, 10))
    figure.suptitle("Comparing Predicted to Actual Adj Close Prices", fontsize=24)
    for i, ticker in enumerate(list(output.keys())):
        plt.subplot(len(output)//2+1, 2, i+1)
        plt.plot(output[ticker][0], label='Predicted Adj Close')
        plt.plot(output[ticker][1], label='Actual Adj Close')
        plt.title(ticker)
        plt.ylabel("Adj Close Price")
        plt.legend()
        plt.grid()
        plt.axis('tight')
    plt.savefig("prediction_actual_values.png", dpi=150)


def plot_history(histories):
    """
    Plot the learning history as a function of epoch for
    each ticker as a subplot.

    Parameters
    ----------
    histories: dictionary of learning history for each ticker.
    """
    figure = plt.figure(figsize=(15, 10))
    figure.suptitle("Learning History", fontsize=24)
    for i, ticker in enumerate(list(histories.keys())):
        plt.subplot(len(histories)//2+1, 2, i+1)
        plt.plot(histories[ticker].history['loss'], label='train')
        plt.plot(histories[ticker].history['val_loss'], label='valid')
        plt.ylabel(ticker)
        plt.xlabel('Epoch')
        plt.legend()
    plt.savefig("learning_history.png", dpi=150)


def evaluate(prediction_values, target_test_values):
    """
    Compute the R2 score and the square root of the Mean Squared Error.
    This is used to evaluate the performance of the models.

    Parameters
    ----------
    prediction_values: array of predicted values
    target_test_values: array of actual target values

    Returns
    -------
    r2_score and Mean squared Error
    """
    return r2_score(target_test_values,prediction_values),np.sqrt(mean_squared_error(target_test_values,prediction_values))


# Benchmark Model (Linear Regression)
def run_linear_regression_model(dfs, tickers):
    """
    Run the benchmark model: Linear Regression.
    """
    output = {}
    for i, df in enumerate(dfs):
        # Extract features and target varaiables
        features, target = get_features_target(df, days_in_future=1)
        # split into test and training sets
        features_train, features_test, target_train, target_test = split_train_test(features, target)
       
        target_train_array = target_train["Adj Close"].tolist()
        features_train_array = list(range(len(target_train_array)))
        target_test_array = target_test["Adj Close"].tolist()
        features_test_array = list(range(len(target_train_array), len(target_test_array)+len(target_train_array)))
        
        # convert dataframes into arrays
        features_train_array = np.reshape(features_train_array, (len(features_train_array), 1))
        target_train_array = np.reshape(target_train_array, (len(target_train_array), 1))
        features_test_array = np.reshape(features_test_array, (len(features_test_array), 1))
        target_test_array = np.reshape(target_test_array, (len(target_test_array), 1))
 
        # make Scikit Learn's Linear Regression Model
        linear_model = LinearRegression()
        # Fit the model to the training set
        linear_model.fit(features_train_array, target_train_array)
        # Use the model to predict the results in the testing set
        linear_prediction = linear_model.predict(features_test_array)
        # Evaluate the model
        r2, mse = evaluate(linear_prediction, target_test_array)
        print (tickers[i], ': R2 score: ', r2, ', Mean Square Error: ',mse)
        #save the output to a dictionary for plotting
        output[tickers[i]] = (linear_prediction, target_test_array)
    #plot the output
    plot_values(output)


# LSTM Model (RNN)
def lstm_model(num_features):
    """
    Make and compile the LSTM model that will be used to learn from
    the training set and applied to the testing set.
    The model consists of 1 LSTM layer with 128 units followed by a
    Dropout layer, a Dense Layer and an activation layer with a linear 
    activiation finction. The model is compiled using the mean squared 
    error as a loss function and nadam as the optimizer.

    Parameters
    ----------
    num_features: number of features in the features array
    
    Returns
    -------
    model: A compiled LSTM model that can be used for fitting and prediction.
    """
    model = Sequential()
    model.add(LSTM(
             input_shape=(1, num_features),
             units=128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='nadam')
    return model


def run_lstm_model(dfs, tickers):
    """
    Run the LSTM Model.
    """
    hp = {}
    hp['AAPL'] = {'epoch': 20, 'batch_size': 30}
    hp['GOOGL'] = {'epoch': 20, 'batch_size': 30}
    hp['AMZN'] = {'epoch': 20, 'batch_size': 30}
    hp['NVDA'] = {'epoch': 20, 'batch_size': 30}
    hp['MSFT'] = {'epoch': 20, 'batch_size': 30}
    output = {}
    histories = {}
    for i, df in enumerate(dfs):
        # Extract features and target varaiables
        features, target = get_features_target(df, days_in_future=1)
        # split into test and training sets
        features_train, features_test, target_train, target_test = split_train_test(features, target)
        # Scale the data both training and testing
        features_train, scaled_mean_features_train, scaled_stds_features_train = mean_scale_features(features_train)
        target_train, scaled_mean_target_train, scaled_stds_target_train = mean_scale_features(target_train)
        features_test, scaled_mean_features_test, scaled_stds_features_test = mean_scale_features(features_test)
        target_test, scaled_mean_target_test, scaled_stds_target_test = mean_scale_features(target_test)

        # convert dataframes into arrays
        features_train_array = np.array(list(make_features_array(features_train)))
        features_test_array = np.array(list(make_features_array(features_test)))
        target_train_array = np.array(list(make_target_array(target_train)))
        target_test_array = np.array(list(make_target_array(target_test)))
 
        # build the Deep Neural Network
        num_features = features_train_array.shape[2]
        model = lstm_model(num_features)
        if i == 0: model.summary()
        # Fit the data
        history = model.fit(features_train_array,
                            target_train_array,
                            epochs=hp[tickers[i]]['epoch'],
                            batch_size=hp[tickers[i]]['batch_size'],
                            validation_split=0.05,
                            verbose=0)
        #save the learning history over epochs
        histories[tickers[i]] = history
        # use the model to pedict the output for the training set
        lstm_prediction = model.predict(features_test_array, batch_size=hp[tickers[i]]['batch_size'])
        # since the output will be scaled, unscale it to the original levels
        lstm_prediction = unscale_target(lstm_prediction, scaled_mean_target_test['Adj Close'], scaled_stds_target_test['Adj Close'])
        target_test_array = unscale_target(target_test_array, scaled_mean_target_test['Adj Close'], scaled_stds_target_test['Adj Close'])

        #Evaluate the model
        r2, mse = evaluate(lstm_prediction, target_test_array)
        print (tickers[i], ': R2 score: ', r2, ', Mean Square Error: ',mse)
        output[tickers[i]] = (lstm_prediction, target_test_array)

    plot_history(histories)
    plot_values(output)


def main():
    tickers = ['AAPL', 'GOOGL', 'AMZN', 'NVDA', 'MSFT']
    dfs = extract_data(tickers)
    explore_data(dfs, tickers)
    run_linear_regression_model(dfs, tickers)
    run_lstm_model(dfs, tickers)

if __name__=="__main__":
    main()

