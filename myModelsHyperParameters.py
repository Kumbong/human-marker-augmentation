import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

from myModels import output_constrained_loss
from keras_tuner.engine.hypermodel import HyperModel

# %% LSTM model (unidirectional) - hyperparameters tuning.
class get_lstm_model(HyperModel):

    def __init__(self, input_dim, output_dim, loss_f,
                 learning_r, units_h, layer_h, batch_size, desired_nFrames, length_constraints = None, angular_constraints = None, lambda_1 = 1, lambda_2 =1, lambda_3 = 1, bidirectional = False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_f = loss_f
        
        self.learning_r = learning_r
        self.units_h = units_h
        self.layer_h = layer_h
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.length_constraints = length_constraints
        self.angular_constraints = angular_constraints
        self.desired_nFrames = desired_nFrames
        self.batch_size = batch_size
        self.bidirectional = bidirectional

    def build(self, hp):
        
        np.random.seed(1)
        tf.random.set_seed(1)
        
        learning_r = hp.Float(self.learning_r["name"], 
                              min_value=self.learning_r["min"],
                              max_value=self.learning_r["max"],
                              sampling=self.learning_r["sampling"], 
                              default=self.learning_r["default"])

        lambda_1 = hp.Float(self.lambda_1["name"], 
                        min_value=self.lambda_1["min"],
                        max_value=self.lambda_1["max"],
                        sampling=self.lambda_1["sampling"],
                        default=self.lambda_1["default"])

        lambda_2 = hp.Float(self.lambda_2["name"], 
                min_value=self.lambda_2["min"],
                max_value=self.lambda_2["max"],
                sampling=self.lambda_2["sampling"],
                default=self.lambda_2["default"])
        
        lambda_3 = hp.Float(self.lambda_3["name"], 
                min_value=self.lambda_3["min"],
                max_value=self.lambda_3["max"],
                sampling=self.lambda_3["sampling"],
                default=self.lambda_3["default"])

        units_h = self.units_h

        
        layers_h = self.layer_h
        model = Sequential()
        
        # First layer
        if self.bidirectional:
            model.add(Bidirectional(LSTM(units=units_h, return_sequences=True), 
                        input_shape=(None, self.input_dim)))  
        else:    
            model.add(LSTM(units = units_h, 
                       input_shape = (None, self.input_dim),
                       return_sequences=True))  
        
        # Hidden layer(s)
        if layers_h > 0:
            if self.bidirectional:
                for _ in range(layers_h):
                    model.add(Bidirectional(LSTM(units=units_h,
                        return_sequences=True)))
            else:
                for _ in range(layers_h):
                    model.add(LSTM(units=units_h, return_sequences=True))
                
        # Last layer
        model.add(TimeDistributed(Dense(self.output_dim, activation='linear'))) 
        
        opt=Adam(learning_rate=learning_r)

        self.loss_f = output_constrained_loss(self.length_constraints, self.angular_constraints, self.batch_size, self.desired_nFrames, lambda_1, lambda_2, lambda_3)

        model.compile(
            optimizer=opt,
            loss=self.loss_f,
            metrics=[MeanSquaredError(), RootMeanSquaredError()]
        )
        
        return model