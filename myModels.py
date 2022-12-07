import numpy as np
import tensorflow as tf
import itertools
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError


# %% Dense model.
def get_dense_model(nFirstUnits, nHiddenUnits, nHiddenLayers, input_dim,
                    output_dim, L2_p, dropout_p, learning_r, loss_f,
                    marker_weights):
    
    # For reproducibility.
    np.random.seed(1)
    tf.random.set_seed(1)

    # Model.
    model = Sequential()
    # First layer.
    if L2_p > 0:
        model.add(Dense(nFirstUnits, input_shape=(input_dim,), 
                        kernel_initializer=glorot_normal(seed=None), 
                        activation='relu',
                        activity_regularizer=L2(L2_p)))
    else:
        model.add(Dense(nFirstUnits, input_shape=(input_dim,), 
                        kernel_initializer=glorot_normal(seed=None), 
                        activation='relu'))
    # Hidden layers.
    if nHiddenLayers > 0:
        for i in range(nHiddenLayers):
            if dropout_p > 0:
                model.add(Dropout(dropout_p))
            if L2_p > 0:
                model.add(Dense(nHiddenUnits, 
                                kernel_initializer=glorot_normal(seed=None), 
                                activation='relu',
                                kernel_regularizer=L2(L2_p)))            
            else:
                model.add(Dense(nHiddenUnits, 
                                kernel_initializer=glorot_normal(seed=None), 
                                activation='relu'))
    if dropout_p > 0:
        model.add(Dropout(dropout_p))
    # Last layer.
    model.add(Dense(output_dim, kernel_initializer=glorot_normal(seed=None), 
                    activation='linear'))
    
    # Optimizer.
    opt=Adam(learning_rate=learning_r)
    
    # Loss function.
    if loss_f == "weighted_mean_squared_error":
        model.compile(
            optimizer=opt,
            loss=weighted_mean_squared_error(marker_weights),
            metrics=[MeanSquaredError(), RootMeanSquaredError()])    
    else:
        model.compile(
            optimizer=opt,
            loss=loss_f,
            metrics=[MeanSquaredError(), RootMeanSquaredError()])
    
    return model

# %% LSTM model.
def get_lstm_model(input_dim, output_dim, nHiddenLayers, nHUnits, learning_r,
                   loss_f, batch_size,desired_nFrames, bidirectional=False, length_constraints=None, angular_constraints=None, lambda_1 = 1, lambda_2 = 1, lambda_3 = 1):
    
    # For reproducibility.
    np.random.seed(1)
    tf.random.set_seed(1)

    # Model.
    model = Sequential()
    # First layer.
    if bidirectional:
        model.add(Bidirectional(LSTM(units=nHUnits, 
                                     input_shape=(None, input_dim),
                                     return_sequences=True)))
    else:
        model.add(LSTM(units=nHUnits, input_shape = (None, input_dim),
                       return_sequences=True))
    # Hidden layers.
    if nHiddenLayers > 0:
        for i in range(nHiddenLayers):
            if bidirectional:
                model.add(Bidirectional(LSTM(units=nHUnits, 
                                             return_sequences=True)))
            else:
                model.add(LSTM(units=nHUnits, return_sequences=True))
    # Last layer.    
    model.add(TimeDistributed(Dense(output_dim, activation='linear')))
    
    # Optimizer.
    opt=Adam(learning_rate=learning_r)

    loss_f = output_constrained_loss(length_constraints, angular_constraints, batch_size, desired_nFrames, lambda_1, lambda_2, lambda_3)


    # Loss function.
    model.compile(
            optimizer=opt,
            loss=loss_f,
            metrics=[MeanSquaredError(), RootMeanSquaredError()])
    
    return model

# %% Helper functions.
def get_callback():
    callback =  tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=1e-08, patience=10, verbose=0,
        mode='min', baseline=None, restore_best_weights=True )
    
    return callback

def weighted_mean_squared_error(weights):
    def loss(y_true, y_pred):      
        squared_difference = tf.square(y_true - y_pred)        
        weighted_squared_difference = weights * squared_difference  
        return tf.reduce_mean(weighted_squared_difference, axis=-1)
    return loss

def output_constrained_loss(length_constraints, angular_constraints, batch_size, desired_nFrames, lambda_1, lambda_2, lambda_3):
    ang_const_loss = output_angular_constr_loss(angular_constraints, batch_size,desired_nFrames, lambda_2, lambda_3)
    len_const_loss = output_len_constr_loss(length_constraints, lambda_1)
    def loss(y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis = [1, 2])
        return mse_loss + ang_const_loss(y_true, y_pred)  + len_const_loss(y_true, y_pred)
    return loss

def output_len_constr_loss(constraints, lambda_1):
    #reduce list of list of tuples to list of tuples (should move to constraint calc function)
    reduced_constraints = list(itertools.chain.from_iterable(constraints))
    def loss(_, y_pred):
        constrained_pairwise_markers_coords = tf.gather(y_pred, indices=reduced_constraints, axis = -1)
        constrained_pairwise_markers_dist = tf.reduce_mean(tf.square(constrained_pairwise_markers_coords[:, :, :, 1, :] - constrained_pairwise_markers_coords[:, :, :, 0, :]), axis=-1)
        one_time_step_dist_diff_mean = tf.reduce_sum(tf.square(constrained_pairwise_markers_dist[:, 1:, :] - constrained_pairwise_markers_dist[:, :-1, :]), axis = -1)
        total_constraint_violation = tf.reduce_sum(one_time_step_dist_diff_mean, axis = -1)

        return lambda_1 * total_constraint_violation

    return loss

def output_angular_constr_loss(constraints,batchSize,desired_nFrames, lambda_2, lambda_3):

    def loss(_, y_pred):
        #add extra dimension to y_pred
        #print(y_pred.shape)
        padding = np.empty((batchSize,desired_nFrames,1)) 
        padding[:] = np.nan
        padding = tf.convert_to_tensor(padding,dtype=tf.float32)
        y_pred_pad = tf.concat([y_pred,padding],axis=-1)

        #retrieve segment coordinate values
        markers_values_segment1 = tf.gather(y_pred_pad, indices=[a[0] for a in constraints], axis=-1)
        markers_values_segment2 = tf.gather(y_pred_pad, indices=[a[1] for a in constraints], axis=-1)
        markers_values_reference = tf.gather(y_pred_pad, indices=[a[2] for a in constraints], axis=-1)

        #calculate centroid of the segments
        centroid_segment1 =tf.experimental.numpy.nanmean(markers_values_segment1, axis=-2, keepdims=True)
        centroid_segment2 = tf.experimental.numpy.nanmean(markers_values_segment2, axis=-2, keepdims=True)
        centroid_reference = tf.experimental.numpy.nanmean(markers_values_reference, axis=-2, keepdims=True)

        #Calculate vectors of each segment (segment-reference)
        vector_segment1 = tf.subtract(centroid_segment1, centroid_reference)
        vector_segment2 = tf.subtract(centroid_segment2, centroid_reference)
        
        #Calculate cosine
        num = tf.reduce_sum(tf.math.multiply(vector_segment1, vector_segment2),axis=-1,keepdims=True)
        norm_vec1 = tf.norm(vector_segment1, ord='euclidean',axis=-1,keepdims=True)
        norm_vec2 = tf.norm(vector_segment2, ord='euclidean',axis=-1,keepdims=True)
        cosine = num/(tf.multiply(norm_vec1, norm_vec2))
        
        #Calculate cosine of ranges
        #ranges = tf.convert_to_tensor([a[3] for a in constraints],dtype=tf.double)
        #cosine_ranges = tf.math.cos(ranges)
        ranges =[a[3] for a in constraints]
        min_ranges = tf.convert_to_tensor([a[0] for a in ranges])
        max_ranges = tf.convert_to_tensor([a[1] for a in ranges])
        min_ranges_cosines = tf.math.cos(min_ranges)
        max_ranges_cosines = tf.math.cos(max_ranges)
        
        #calculate difference between cosine of segments and min range
        min_ranges = tf.reshape(min_ranges_cosines,shape=(6,1,1))
        diff_min_range = tf.subtract(min_ranges,cosine)
        min_range_loss = tf.keras.activations.relu(diff_min_range)
        
        #calculate difference between cosine of segments and max range
        max_ranges = tf.reshape(max_ranges_cosines,shape=(6,1,1))
        diff_max_range = tf.subtract(cosine,max_ranges)
        max_range_loss = tf.keras.activations.relu(diff_max_range)
        
        #reduce dimensions and add to loss term
        min_angle_loss =tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(min_range_loss,axis=-1),axis=-1),axis=-1),axis=-1)
        max_angle_loss =tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(max_range_loss,axis=-1),axis=-1),axis=-1),axis=-1)

        return lambda_2 * min_angle_loss + lambda_3 * max_angle_loss

    return loss
