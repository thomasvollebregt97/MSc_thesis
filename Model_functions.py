import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import t
import collections
from keras.optimizers import Adam


############################################################# Functions for LSTM modeling #############################################################

def rv_calc(log_returns, h):
    
    """
    Calculates the realized volatility at each point in time after the time window h is surpassed. Balancing h is essential,
    more h is a more accurate description, but volatility changes over time, which means window must not get too big

    Arguments:
    log_returns -- 1D array of log_returns
    h -- integer that specifies the window size for volatilty estimation
    """
    
    rv = np.zeros(len(log_returns)-(h-1))

    for i in range(len(log_returns)-(h-1)):
        rv[i] = np.var(log_returns[i:(h+i)])  
    rv = np.sqrt(rv)
    return rv

def scaled_data(data):
    
    """
    Scales the input data to a specified range using the MinMaxScaler from sklearn.

    Arguments:
    data -- 1D numpy array or a list, the data to be scaled.
    """
    
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler([-1,1])
    scaler.fit(data)
    scaled_rv = scaler.transform(data)
    return scaled_rv

def make_supervised(scaled_rv, obs_test, d, h, start): 
    
    """Transforms a time series into a supervised learning problem.
    
    Arguments:
    scaled_rv -- 1D numpy array, the scaled realized volatility, time series data.
    d -- int, the batch size or the number of time steps in each input data sample.
    h -- int, the forecast horizon or number of time steps to predict in the future.
    
    Returns:
    X -- 2D numpy array, the input data for the supervised learning problem.
    y -- 2D numpy array, the target output data for the supervised learning problem.
    """

    rv_train = scaled_rv[start:(start+((len(scaled_rv)-obs_test)))]

    n = math.floor(len(rv_train)/d)
    rv_train = rv_train.reshape(-1,)
    X = np.zeros((n,d))
    y = np.zeros((np.shape(X)[0], h))

    for i in range(n):
        X[i,:] = rv_train[(i*d):((i*d)+d)]
    for j in range(np.shape(X)[0]):
        y[j,:] = rv_train[(j+1)*d:((j+1)*d)+h]

    #Add 3D component
    X_3D = np.zeros((X.shape[0], X.shape[1], 1))
    X_3D[:, :, 0] = X

    y_3D = np.zeros((y.shape[0], y.shape[1], 1))
    y_3D[:, :, 0] = y

    return X_3D, y_3D


def test_batch(scaled_rv, obs_test_total, d):
    
    ''' 
    Create obs_test_total amount of test batches to plug into the LSTM model, it will return obs_test_total amount of forecasts 
    to compare with the realized volatility.
    
    Arguments:
    scaled_rv -- 1D numpy array, the scaled realized volatility, time series data.
    obs_test_total -- int, the total amount of forecast sequences produced
    d -- int, batch size.
    
    Returns:
    rv_test_3D -- 3D numpy array, the test batch used to produce forecasts.
    '''

    rv_train = scaled_rv[0:((len(scaled_rv) - obs_test_total))]
    rv_test = scaled_rv[(len(scaled_rv) - obs_test_total):len(scaled_rv)]

    rv_1 = np.ravel(rv_train[(len(rv_train) - d):len(rv_train)])
    rv_2 = np.ravel(rv_test)
    rv_batch = np.concatenate((rv_1, rv_2))

    rv_test_batch = np.zeros((obs_test_total, d))

    for i in range(obs_test_total):
        rv_test_batch[i,:] = rv_batch[i:i+d]

    rv_test_3D = np.zeros((rv_test_batch.shape[0], rv_test_batch.shape[1], 1))
    rv_test_3D[:,:,0] = rv_test_batch

    return rv_test_3D

def lstm_model(X_3D, y_3D, verbose, epochs, d, units, nodes, LR, activation_lstm):
    """
    This function defines, compiles, and fits an LSTM model on the provided data.

    Arguments:
    X_3D -- 3D numpy array, the input data for the LSTM model. The first dimension is the sample index, the second dimension is the time step, and the third dimension is the feature.
    y_3D -- 2D numpy array, the target output data for the LSTM model. The first dimension is the sample index and the second dimension is the time step.
    verbose -- int, whether to print detailed information during the model training. 0 means silent, 1 means progress bar, and 2 means one line per epoch.
    epochs -- int, the number of epochs to train the model.
    d -- int, the batch size for the model training.
    units -- int, the number of LSTM units.
    nodes -- int, the number of nodes in the dense layer of the LSTM model.
    LR -- float, the learning rate of the LSTM model
    activation_lstm -- str, allows you to modify the activation function of the LSTM layer

    Returns:
    model -- Sequential, the trained LSTM model.
    """

    # Define parameters
    verbose, epochs, batch_size = verbose, epochs, d
    n_timesteps, n_features, n_outputs = X_3D.shape[1], X_3D.shape[2], y_3D.shape[1]

    # Define model
    model = Sequential()
    model.add(LSTM(units, activation=activation_lstm, input_shape=(n_timesteps, n_features))) # LSTM units
    model.add(Dense(nodes, activation='relu')) # nodes
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')

    # Define a custom Adam optimizer with a learning rate of LR
    custom_adam = Adam(learning_rate = LR)

    # fit network
    model.fit(X_3D, y_3D, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model

def generate_2_forecasts(rv, model_1, model_2, rv_test_3D_1, rv_test_3D_2, obs_test, d):
    """
    This function generates forecasts using two LSTM models.

    Arguments:
    rv -- 1D numpy array, the original time series data.
    model_1, model_2 -- keras.models.Sequential, the two LSTM models used for generating forecasts.
    rv_test_3D_1, rv_test_3D_2 -- 3D numpy arrays, the test data for the LSTM models.
    obs_test -- int, the number of test observations.
    d -- int, the batch size or the number of time steps in each input data sample.

    Returns:
    preds_lstm -- 1D numpy array, the concatenation of the forecasts generated by the two LSTM models.
    """
    
    #For inverse transform
    rv = rv.reshape(-1, 1)
    scaler = MinMaxScaler([-1,1])
    scaler.fit(rv)
    
    #Generate 2 sets for training
    test_3D_2f_1 = np.zeros((rv_test_3D_1.shape[0], rv_test_3D_1.shape[1], 1))
    X_3D_1 = np.squeeze(rv_test_3D_1)
    test_3D_2f_1[:,:,0] = X_3D_1
    
    test_3D_2f_2 = np.zeros((rv_test_3D_2.shape[0], rv_test_3D_2.shape[1], 1))
    X_3D_2 = np.squeeze(rv_test_3D_2)
    test_3D_2f_2[:,:,0] = X_3D_2
    
    test_3D_2f_1 = test_3D_2f_1.reshape((-1, d, 1)) #2 for 2 features
    pred_1 = model_1.predict(test_3D_2f_1) #Predicts 20.000 forecasts!
    pred_1 = scaler.inverse_transform(pred_1)
    
    test_3D_2f_2 = test_3D_2f_2.reshape((-1, d, 1)) #2 for 2 features
    pred_2 = model_2.predict(test_3D_2f_2) #Predicts 20.000 forecasts!
    pred_2 = scaler.inverse_transform(pred_2)
    
    pred_1 = np.squeeze(pred_1) 
    pred_2 = np.squeeze(pred_2)
    preds_lstm = np.concatenate((pred_1, pred_2))
    
    return preds_lstm

def univariate_lstm_preds(scaled_rv, obs_test, start, d, h, epochs, units, nodes, verbose, LR, activation_lstm, rv):
    """
    This function trains LSTM models on different parts of the time series and then generates forecasts using these models.

    Arguments:
    scaled_rv -- 1D numpy array, the scaled time series data.
    obs_test -- int, the number of test observations.
    d -- int, the batch size or the number of time steps in each input data sample.
    h -- int, the forecast horizon or number of time steps to predict in the future.
    epochs -- int, the number of epochs for training the LSTM models.
    units -- int, the number of units in the LSTM layer.
    nodes -- int, the number of nodes in the Dense layer.
    verbose -- int, the verbosity mode for training the LSTM models.
    rv -- 1D numpy array, the original time series data.

    Returns:
    preds_lst -- 1D numpy array, the forecasts generated by the LSTM models.
    """

    #Make 2 training sets
    X_3D_1 , y_3D_1 = make_supervised(scaled_rv, obs_test, d, h, 0)
    X_3D_2 , y_3D_2 = make_supervised(scaled_rv, obs_test, d, h, start)

    obs_test_total = 20000

    rv_test_3D = test_batch(scaled_rv, obs_test_total, d)
    rv_test_3D_1 = rv_test_3D[:int((len(rv_test_3D)/2))]
    rv_test_3D_2 = rv_test_3D[int((len(rv_test_3D)/2)):]

    model_1 = lstm_model(X_3D_1, y_3D_1, verbose, epochs, d, units, nodes, LR, activation_lstm)
    model_2 = lstm_model(X_3D_2, y_3D_2, verbose, epochs, d, units, nodes, LR, activation_lstm)

    preds_lst = generate_2_forecasts(rv, model_1, model_2, rv_test_3D_1, rv_test_3D_2, obs_test, d)
    
    return preds_lst


def generate_2_forecasts_hybrid(rv_data, filt_data, obs_test, start, d, epochs, units, nodes, LR, activation_lstm, rv, h):
    
    """
    This function generates forecasts for time series data using a hybrid approach of realized and filtered volatilities.
    The data is transformed into supervised learning format and fed into LSTM models for forecasting.
    
    Arguments:
    rv_data -- 1D numpy array, the realized volatilities data.
    filt_data -- 1D numpy array, the filtered volatilities data.
    obs_test -- int, the number of observations for the test set.
    start -- int, the start index for the second training set.
    d -- int, the number of time steps in each input data sample.
    epochs -- int, the number of epochs for the LSTM model training.
    units -- int, the number of LSTM units.
    nodes -- int, the number of nodes in the dense layer of the LSTM model.
    rv -- 1D numpy array, the realized volatilities data used for scaling the forecasts.
    h -- int, the forecast horizon or number of time steps to predict in the future.
    
    Returns:
    pred -- 2D numpy array, the concatenated forecasts from the two LSTM models.
    """
        
    scaled_filt = filt_data
    h = h

    X_3D_1 , y_3D_1 = make_supervised(rv_data, obs_test, d, h, 0)
    X_3D_2 , y_3D_2 = make_supervised(rv_data, obs_test, d, h, start)
    
    X_3D_filt_1, y_3D_filt_1 = make_supervised(scaled_filt, obs_test, d, h, start = 0) #y_3D_filt_1 is not used because the filtered volatility ouputs are not used in training
    X_3D_filt_2, y_3D_filt_2 = make_supervised(scaled_filt, obs_test, d, h, start = start) #y_3D_filt_2 is not used because the filtered volatility ouputs are not used in training
    
    #Training set 1
    features = 2
    X_3D_2f_1 = np.zeros((X_3D_filt_1.shape[0], X_3D_filt_1.shape[1], features))
    X_3D_1 = np.squeeze(X_3D_1)
    X_3D_filt_1 = np.squeeze(X_3D_filt_1)
    X_3D_2f_1[:,:,0] = X_3D_1
    X_3D_2f_1[:,:,1] = X_3D_filt_1
    
    #Training set 2
    X_3D_2f_2 = np.zeros((X_3D_filt_2.shape[0], X_3D_filt_2.shape[1], features))
    X_3D_2 = np.squeeze(X_3D_2)
    X_3D_filt_2 = np.squeeze(X_3D_filt_2)
    X_3D_2f_2[:,:,0] = X_3D_2
    X_3D_2f_2[:,:,1] = X_3D_filt_2
    
    verbose = 0

    model_1 = lstm_model(X_3D_2f_1, y_3D_1, verbose, epochs, d, units, nodes, LR, activation_lstm)
    model_2 = lstm_model(X_3D_2f_2, y_3D_2, verbose, epochs, d, units, nodes, LR, activation_lstm)
    
    #Generate 2 sets for training
    obs_test_total = obs_test

    #1e tensor laag: realized volatilities
    rv_test_3D = test_batch(rv_data, obs_test_total, d)
    rv_test_3D_1 = rv_test_3D[:int((len(rv_test_3D)/2))]
    rv_test_3D_2 = rv_test_3D[int((len(rv_test_3D)/2)):]

    #2de tensor laag: filtered/predicted volatilities
    filt_test_3D = test_batch(scaled_filt, obs_test_total, d)
    filt_test_3D_1 = filt_test_3D[:int((len(rv_test_3D)/2))]
    filt_test_3D_2 = filt_test_3D[int((len(rv_test_3D)/2)):] 
    
    # Test set Nr. 1 & 2
    testset_1 = np.zeros((rv_test_3D_1.shape[0], rv_test_3D_1.shape[1], features))
    testset_2 = np.zeros((rv_test_3D_2.shape[0], rv_test_3D_2.shape[1], features))

    testset_1[:,:,0] = np.squeeze(rv_test_3D_1) 
    testset_1[:,:,1] = np.squeeze(filt_test_3D_1) 

    testset_2[:,:,0] = np.squeeze(rv_test_3D_2) 
    testset_2[:,:,1] = np.squeeze(filt_test_3D_2) 
    
    #Define rv for inverse transformation
    rv = rv.reshape(-1, 1)
    scaler = MinMaxScaler([-1,1])
    scaler.fit(rv)
    test_len = obs_test
    
    # Generate forecasts for set 1
    testset_1 = testset_1.reshape((-1, d, 2))
    pred_1 = model_1.predict(testset_1) #Predicts 10.000 forecasts!
    pred_1 = scaler.inverse_transform(pred_1)
    
    # Generate forecasts for set 2
    testset_2 = testset_2.reshape((-1, d, 2))
    pred_2 = model_2.predict(testset_2) #Predicts 10.000 forecasts!
    pred_2 = scaler.inverse_transform(pred_2)
    
    pred = np.concatenate((pred_1, pred_2))
    return pred


def generate_forecasts_hybrid_no_rf(rv_data, filt_data, obs_test, start, d, epochs, units, nodes, LR, activation_lstm, rv):
    
    h = 1

    X_3D, y_3D = make_supervised(rv_data, obs_test, d, h, 0)
    X_3D_filt, y_3D_filt = make_supervised(filt_data, obs_test, d, h, start = 0)

    features = 2
    X_3D_2f = np.zeros((X_3D.shape[0], X_3D.shape[1], features))

    X_3D = np.squeeze(X_3D)
    X_3D_filt = np.squeeze(X_3D_filt)

    X_3D_2f[:,:,0] = X_3D
    X_3D_2f[:,:,1] = X_3D_filt

    obs_test_total = 20000
    epochs = 70
    units = 300
    nodes = 100
    verbose = 0

    model = lstm_model(X_3D_2f, y_3D, verbose, epochs, d, units, nodes, LR, activation_lstm)

    X_pred_test_3D = test_batch(filt_data, obs_test, d)
    rv_test_3D = test_batch(rv_data, obs_test_total, d)

    test_3D_2f = np.zeros((X_pred_test_3D.shape[0], X_pred_test_3D.shape[1], features))
    X_3D = np.squeeze(rv_test_3D)
    X_3D_filt = np.squeeze(X_pred_test_3D)

    test_3D_2f[:,:,0] = X_3D
    test_3D_2f[:,:,1] = X_3D_filt

    rv = rv.reshape(-1, 1)
    scaler = MinMaxScaler([-1,1])
    scaler.fit(rv)
    test_len = obs_test_total
    
    # Generate forecasts

    test_3D_2f = test_3D_2f.reshape((-1, d, 2))

    pred = model.predict(test_3D_2f) #Predicts 20.000 forecasts!
    pred = scaler.inverse_transform(pred)
    return pred

def create_h_step_forecasts(pred_matrix, h, obs_test):
    """
    This function creates h-step forecasts from the provided prediction matrix.
    It does this by collecting every h-th forecast in the matrix.

    Arguments:
    - pred_matrix (numpy.array): A 2D numpy array containing the predictions. 24 steps for each sequence, 
    We retrieve every 24th sequence.
    - h (int): The step size for the forecast. For example, if h=3, the function will collect every 3rd prediction.
    - obs_test (int): The total number of observations in the test data.

    Returns:
    - my_vector (numpy.array): A 1D numpy array containing the h-step forecasts.
    """

    prediction_list = []
    
    for j in range(int(np.round(obs_test/h))):
        prediction_list.append(pred_matrix[j*h,:].tolist())
    
    prediction_array = np.array(prediction_list)
    prediction_vector = prediction_array.flatten()
    
    return prediction_vector

################################################ Functions that allow for a rolling window supervises learning approach #########################################

def make_supervised_shift(scaled_rv, obs_test, d, h, start, shift):
    
    """Transforms a time series into a supervised learning problem. BUT: This function allows you to produce a rolling window training. 
    You can shift the window to your preference, when you shift by 20, after each 20 observations you create a new training set, with d
    observations and h associated output values. This creates a much larger training sample, but with a higher correlation between the 
    samples, and you can't use the last ~shift observations as they do not fit in your training set anymore.
    
    Arguments:
    scaled_rv -- 1D numpy array, the scaled realized volatility, time series data.
    d -- int, the batch size or the number of time steps in each input data sample.
    h -- int, the forecast horizon or number of time steps to predict in the future.
    SHIFT(!) -- int, the number of time steps to shift each input data sample.
    
    Returns:
    X -- 2D numpy array, the input data for the supervised learning problem.
    y -- 2D numpy array, the target output data for the supervised learning problem.
    """
    
    scaled_rv_train = scaled_rv[start:(start+((len(scaled_rv)-obs_test)))]
    
    num_samples = (len(scaled_rv_train) - d - h + 1) // shift
    
    X = np.empty((num_samples, d))
    y = np.empty((num_samples, h))
    
    for i in range(num_samples):
        X[i] = scaled_rv_train[i * shift : i * shift + d]
        y[i] = scaled_rv_train[i * shift + d : i * shift + d + h]  # predict next h steps
        
    #Add 3D component
    X_3D = np.zeros((X.shape[0], X.shape[1], 1))
    X_3D[:, :, 0] = X

    y_3D = np.zeros((y.shape[0], y.shape[1], 1))
    y_3D[:, :, 0] = y

    return X_3D, y_3D


def univariate_lstm_preds_shift(scaled_rv, obs_test, start, d, h, epochs, units, nodes, verbose, LR, activation_lstm, rv, shift):

    """ Same function as before, but now trained using rolling window approach, 
    you can adjust rolling window using the shift argument"""


    #Make 2 training sets
    X_3D_1 , y_3D_1 = make_supervised_shift(scaled_rv, obs_test, d, h, 0, shift)
    X_3D_2 , y_3D_2 = make_supervised_shift(scaled_rv, obs_test, d, h, start, shift)

    obs_test_total = 20000  # Before edits

    rv_test_3D = test_batch(scaled_rv, obs_test, d)
    rv_test_3D_1 = rv_test_3D[:int((len(rv_test_3D)/2))]
    rv_test_3D_2 = rv_test_3D[int((len(rv_test_3D)/2)):]

    model_1 = lstm_model(X_3D_1, y_3D_1, verbose, epochs, d, units, nodes, LR, activation_lstm)
    model_2 = lstm_model(X_3D_2, y_3D_2, verbose, epochs, d, units, nodes, LR, activation_lstm)

    preds_lst = generate_2_forecasts(rv, model_1, model_2, rv_test_3D_1, rv_test_3D_2, obs_test, d)
    
    return preds_lst


def generate_2_forecasts_hybrid_shift(rv_data, filt_data, obs_test, start, d, epochs, units, nodes, LR, activation_lstm, rv, h, shift):
    
    """ Same function as before, but now trained using rolling window approach, 
    you can adjust rolling window using the shift argument"""

    scaled_filt = filt_data

    X_3D_1 , y_3D_1 = make_supervised_shift(rv_data, obs_test, d, h, 0, shift)
    X_3D_2 , y_3D_2 = make_supervised_shift(rv_data, obs_test, d, h, start, shift)
    
    X_3D_filt_1, y_3D_filt_1 = make_supervised_shift(scaled_filt, obs_test, d, h, 0, shift)
    X_3D_filt_2, y_3D_filt_2 = make_supervised_shift(scaled_filt, obs_test, d, h, start, shift)
    
    #Training set 1
    features = 2
    X_3D_2f_1 = np.zeros((X_3D_filt_1.shape[0], X_3D_filt_1.shape[1], features))
    X_3D_1 = np.squeeze(X_3D_1)
    X_3D_filt_1 = np.squeeze(X_3D_filt_1)
    X_3D_2f_1[:,:,0] = X_3D_1
    X_3D_2f_1[:,:,1] = X_3D_filt_1
    
    #Training set 2
    X_3D_2f_2 = np.zeros((X_3D_filt_2.shape[0], X_3D_filt_2.shape[1], features))
    X_3D_2 = np.squeeze(X_3D_2)
    X_3D_filt_2 = np.squeeze(X_3D_filt_2)
    X_3D_2f_2[:,:,0] = X_3D_2
    X_3D_2f_2[:,:,1] = X_3D_filt_2
    
    verbose = 0

    model_1 = lstm_model(X_3D_2f_1, y_3D_1, verbose, epochs, d, units, nodes, LR, activation_lstm)
    model_2 = lstm_model(X_3D_2f_2, y_3D_2, verbose, epochs, d, units, nodes, LR, activation_lstm)
    
    #Generate 2 sets for training
    obs_test_total = obs_test

    #1e tensor laag: realized volatilities
    rv_test_3D = test_batch(rv_data, obs_test_total, d)
    rv_test_3D_1 = rv_test_3D[:int((len(rv_test_3D)/2))]
    rv_test_3D_2 = rv_test_3D[int((len(rv_test_3D)/2)):]

    #2de tensor laag: filtered/predicted volatilities
    filt_test_3D = test_batch(scaled_filt, obs_test_total, d)
    filt_test_3D_1 = filt_test_3D[:int((len(rv_test_3D)/2))]
    filt_test_3D_2 = filt_test_3D[int((len(rv_test_3D)/2)):] 
    
    # Test set Nr. 1 & 2
    testset_1 = np.zeros((rv_test_3D_1.shape[0], rv_test_3D_1.shape[1], features))
    testset_2 = np.zeros((rv_test_3D_2.shape[0], rv_test_3D_2.shape[1], features))

    testset_1[:,:,0] = np.squeeze(rv_test_3D_1) 
    testset_1[:,:,1] = np.squeeze(filt_test_3D_1) 

    testset_2[:,:,0] = np.squeeze(rv_test_3D_2) 
    testset_2[:,:,1] = np.squeeze(filt_test_3D_2) 
    
    #Define rv for inverse transformation
    rv = rv.reshape(-1, 1)
    scaler = MinMaxScaler([-1,1])
    scaler.fit(rv)
    test_len = obs_test
    
    # Generate forecasts for set 1
    testset_1 = testset_1.reshape((-1, d, 2))
    pred_1 = model_1.predict(testset_1) #Predicts 10.000 forecasts!
    pred_1 = scaler.inverse_transform(pred_1)
    
    # Generate forecasts for set 2
    testset_2 = testset_2.reshape((-1, d, 2))
    pred_2 = model_2.predict(testset_2) #Predicts 10.000 forecasts!
    pred_2 = scaler.inverse_transform(pred_2)
    
    pred = np.concatenate((pred_1, pred_2))
    return pred


############################################################ Garch-type models error analysis ############################################################ 

def calc_errors(rv, obs_test, preds_norm, preds_std, preds_ged):
    
    MSE_norm = mse(rv[(len(rv)-obs_test):len(rv)], preds_norm)
    MAE_norm = mae(rv[(len(rv)-obs_test):len(rv)], preds_norm)
    MSE_std = mse(rv[(len(rv)-obs_test):len(rv)], preds_std)
    MAE_std = mae(rv[(len(rv)-obs_test):len(rv)], preds_std)
    MSE_ged = mse(rv[(len(rv)-obs_test):len(rv)], preds_ged)
    MAE_ged = mae(rv[(len(rv)-obs_test):len(rv)], preds_ged)
    
    return MSE_norm, MAE_norm, MSE_std, MAE_std, MSE_ged, MAE_ged

def print_errors(MSE_norm, MAE_norm, MSE_std, MAE_std, MSE_ged, MAE_ged, model):
    print('Errors: ' + model +' Normal')
    print(MSE_norm)
    print(MAE_norm)
    print('Errors: '+ model +' Std')
    print(MSE_std)
    print(MAE_std)
    print('Errors: '+ model +' GED')
    print(MSE_ged)
    print(MAE_ged)

def calc_errors_ms(rv, obs_test, preds_std, preds_ged):
    
    MSE_std = mse(rv[(len(rv)-obs_test):len(rv)], preds_std)
    MAE_std = mae(rv[(len(rv)-obs_test):len(rv)], preds_std)
    MSE_ged = mse(rv[(len(rv)-obs_test):len(rv)], preds_ged)
    MAE_ged = mae(rv[(len(rv)-obs_test):len(rv)], preds_ged)
    
    return MSE_std, MAE_std, MSE_ged, MAE_ged


def calc_errors_lstm(rv, obs_test, preds_lstm, preds_lstm_egarch_std, preds_lstm_egarch_ged, preds_lstm_msgarch_std, preds_lstm_msgarch_ged):
    
    MSE_std_1 = mse(rv[(len(rv)-obs_test):len(rv)], preds_lstm)
    MAE_std_1 = mae(rv[(len(rv)-obs_test):len(rv)], preds_lstm)
    MSE_e_std_2 = mse(rv[(len(rv)-obs_test):len(rv)], preds_lstm_egarch_std)
    MAE_e_std_2 = mae(rv[(len(rv)-obs_test):len(rv)], preds_lstm_egarch_std)
    MSE_e_ged_3 = mse(rv[(len(rv)-obs_test):len(rv)], preds_lstm_egarch_ged)
    MAE_e_ged_3 = mae(rv[(len(rv)-obs_test):len(rv)], preds_lstm_egarch_ged)
    MSE_ms_std_4 = mse(rv[(len(rv)-obs_test):len(rv)], preds_lstm_msgarch_std)
    MAE_ms_std_4 = mae(rv[(len(rv)-obs_test):len(rv)], preds_lstm_msgarch_std)
    MSE_ms_ged_5 = mse(rv[(len(rv)-obs_test):len(rv)], preds_lstm_msgarch_ged)
    MAE_ms_ged_5 = mae(rv[(len(rv)-obs_test):len(rv)], preds_lstm_msgarch_ged)
    
    return MSE_std_1, MAE_std_1, MSE_e_std_2, MAE_e_std_2, MSE_e_ged_3, MAE_e_ged_3, MSE_ms_std_4, MAE_ms_std_4, MSE_ms_ged_5, MAE_ms_ged_5

def print_errors_ms(MSE_std, MAE_std, MSE_ged, MAE_ged, model):
    print('Errors: '+ model +' Std')
    print(MSE_std)
    print(MAE_std)
    print('Errors: '+ model +' GED')
    print(MSE_ged)
    print(MAE_ged)


############################################################ Error calculations: simulations (WIP!) ############################################################ 

# Function h-step ahead

def produce_h_step_forecasting_errors(scaled_rv, pred_matrix, obs_test, start, d, p, h, epochs, units, nodes, verbose, LR, activation_lstm, rv, shift, benchmark_vol):
    
    # Unpack prediction matrix
    scaled_pred_arch = scaled_data(pred_matrix[:, 0]).flatten()
    scaled_pred_garch = scaled_data(pred_matrix[:, 1]).flatten()
    scaled_pred_egarch = scaled_data(pred_matrix[:, 2]).flatten()
    scaled_pred_gjrgarch = scaled_data(pred_matrix[:, 3]).flatten()
    
    if np.shape(pred_matrix)[1]==5:
        scaled_pred_msgarch = scaled_data(pred_matrix[:, 4]).flatten()

    diff =  len(scaled_rv) - len(scaled_pred_arch) 

    # Generate predictions (p = window size realized volatility)
    lstm_preds = univariate_lstm_preds_shift(scaled_rv, obs_test, start, d, h, epochs, units, nodes, verbose, LR, activation_lstm, rv, shift)
    lstm_preds_arch = generate_2_forecasts_hybrid_shift(scaled_rv[diff:], scaled_pred_arch, obs_test, start, d, epochs, units, nodes, LR, activation_lstm, rv, h, shift)
    lstm_preds_garch = generate_2_forecasts_hybrid_shift(scaled_rv[diff:], scaled_pred_garch, obs_test, start, d, epochs, units, nodes, LR, activation_lstm, rv, h, shift)
    lstm_preds_egarch = generate_2_forecasts_hybrid_shift(scaled_rv[diff:], scaled_pred_egarch, obs_test, start, d, epochs, units, nodes, LR, activation_lstm, rv, h, shift)
    lstm_preds_gjrgarch = generate_2_forecasts_hybrid_shift(scaled_rv[diff:], scaled_pred_gjrgarch, obs_test, start, d, epochs, units, nodes, LR, activation_lstm, rv, h, shift)
    lstm_preds_msgarch = generate_2_forecasts_hybrid_shift(scaled_rv[diff:], scaled_pred_msgarch, obs_test, start, d, epochs, units, nodes, LR, activation_lstm, rv, h, shift)

    # Execute if h>1: Retrieving every h-th sequence, for h-th step ahead prediction evaluation
    if h!=1:        
        lstm_preds = create_h_step_forecasts(lstm_preds, h, obs_test)
        lstm_preds_arch = create_h_step_forecasts(lstm_preds_arch, h, obs_test)
        lstm_preds_garch = create_h_step_forecasts(lstm_preds_garch, h, obs_test)
        lstm_preds_egarch = create_h_step_forecasts(lstm_preds_egarch, h, obs_test)
        lstm_preds_gjrgarch = create_h_step_forecasts(lstm_preds_gjrgarch, h, obs_test)
        lstm_preds_msgarch = create_h_step_forecasts(lstm_preds_msgarch, h, obs_test)

    # Error calculations (MSE)
    print(f"MSE: LSTM & hybrid models {h}-step ahead forecast")
    print(mse(lstm_preds[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))])) # -(h+1) to ensure same length outputs
    print(mse(lstm_preds_arch[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))]))
    print(mse(lstm_preds_garch[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))]))
    print(mse(lstm_preds_egarch[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))]))
    print(mse(lstm_preds_gjrgarch[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))]))
    print(mse(lstm_preds_msgarch[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))]))

    # Error calculations (MAE)
    print(f"MAE: LSTM & hybrid models {h}-step ahead forecast")
    print(mae(lstm_preds[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))])) # -(h+1) to ensure same length outputs
    print(mae(lstm_preds_arch[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))]))
    print(mae(lstm_preds_garch[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))]))
    print(mae(lstm_preds_egarch[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))]))
    print(mae(lstm_preds_gjrgarch[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))]))
    print(mae(lstm_preds_msgarch[:obs_test-(h+1)], benchmark_vol[(len(benchmark_vol) - obs_test):(len(benchmark_vol)-(h+1))]))

    # Realized variance errors (MSE & MAE)
    print(f"MSE & MAE: realized volatility {h}-step ahead forecast")
    print(mse(rv[(len(rv) - obs_test - h):(len(rv) - h)], benchmark_vol[(len(benchmark_vol)-obs_test):len(benchmark_vol)]))
    print(mae(rv[(len(rv) - obs_test - h):(len(rv) - h)], benchmark_vol[(len(benchmark_vol)-obs_test):len(benchmark_vol)]))
    
    # Flatten predictions to put them into 1 matrix
    lstm_preds_arch = lstm_preds_arch.flatten()
    lstm_preds_garch = lstm_preds_garch.flatten()
    lstm_preds_egarch = lstm_preds_egarch.flatten()
    lstm_preds_gjrgarch = lstm_preds_gjrgarch.flatten()
    lstm_preds_msgarch = lstm_preds_msgarch.flatten()
    
    prediction_matrix_lstm = np.hstack((lstm_preds, lstm_preds_arch, lstm_preds_garch, lstm_preds_egarch, lstm_preds_gjrgarch, lstm_preds_msgarch))
    
    return prediction_matrix_lstm

########################################################################## Diebold-Mariano test ##################################################################


def dm_test_2(real_values, pred1, pred2, h=1, harvey_adj=True):
    """
    Implements the Diebold-Mariano (DM) test to compare the accuracy of two sets of predictions.

    Parameters:
    real_values (list): A list of actual (observed) values.
    pred1 (list): A list of the first set of predicted values.
    pred2 (list): A list of the second set of predicted values.
    h (int, optional): The forecast horizon. Defaults to 1.
    harvey_adj (bool, optional): Whether to apply Harvey's adjustment for small samples. Defaults to True.

    Returns:
    result (namedtuple): A namedtuple with two fields: 'DM' and 'p_value'. 'DM' is the DM test statistic, and 'p_value' is the associated p-value of the test.
    """

    e1_lst = []
    e2_lst = []
    d_lst = []

    real_values = pd.Series(real_values).apply(lambda x: float(x)).tolist()
    pred1 = pd.Series(pred1).apply(lambda x: float(x)).tolist()
    pred2 = pd.Series(pred2).apply(lambda x: float(x)).tolist()

    # Length of forecasts
    T = float(len(real_values))

    # Construct loss differential according to error criterion (MSE)
    for real, p1, p2 in zip(real_values, pred1, pred2):
        e1_lst.append((real - p1)**2)
        e2_lst.append((real - p2)**2)
    for e1, e2 in zip(e1_lst, e2_lst):
        d_lst.append(e1 - e2)

    # Mean of loss differential
    mean_d = pd.Series(d_lst).mean()

    # Calculate autocovariance 
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
            autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    
    # Calculate the denominator of DM stat
    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))  # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    
    # Calculate DM stat
    DM_stat = V_d**(-0.5)*mean_d

    # Calculate and apply Harvey adjustement
    # It applies a correction for small sample
    if harvey_adj is True:
        harvey_adj = ((T+1-2*h+h*(h-1)/T)/T)**(0.5)
        DM_stat = harvey_adj*DM_stat 

    # Calculate p-value
    p_value = 2*t.cdf(-abs(DM_stat), df=T - 1)

    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    result = dm_return(DM=DM_stat, p_value=p_value)

    return result