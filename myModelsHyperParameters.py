import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

from myModels import output_len_constr_loss
from keras_tuner.engine.hypermodel import HyperModel

# %% LSTM model (unidirectional) - hyperparameters tuning.
class get_unidirectional_lstm_model(HyperModel):

    def __init__(self, input_dim, output_dim, loss_f,
                 learning_r, units_h, layer_h, constraints = None, lambda_1 = 1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_f = loss_f
        
        self.learning_r = learning_r
        self.units_h = units_h
        self.layer_h = layer_h
        self.lambda_1 = lambda_1
        self.constraints = constraints

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
                        default=self.lambda_1["default"])

        units_h = self.units_h

        
        layers_h = self.layer_h
        model = Sequential()
        
        # First layer
        model.add(LSTM(units = units_h, 
                       input_shape = (None, self.input_dim),
                       return_sequences=True))        
        
        # Hidden layer(s)
        if layers_h > 0:
            for i in range(layers_h):
                model.add(LSTM(units=units_h, return_sequences=True))
                
        # Last layer
        model.add(TimeDistributed(Dense(self.output_dim, activation='linear'))) 
        
        opt=Adam(learning_rate=learning_r)

        if self.loss_f == 'output_length_constr':
            self.loss_f = output_len_constr_loss(self.constraints, lambda_1)

        model.compile(
            optimizer=opt,
            loss=self.loss_f,
            metrics=[MeanSquaredError(), RootMeanSquaredError()]
        )
        
        return model
    
# %% LSTM model (bidirectional) - hyperparameters tuning.
class get_bidirectional_lstm_model(HyperModel):

    def __init__(self, input_dim, output_dim, loss_f,
                 learning_r, units_h, layer_h, constraints = None, lambda_1 = 1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_f = loss_f        
        self.learning_r = learning_r
        self.units_h = units_h
        self.layer_h = layer_h
        self.constraints = constraints
        self.lambda_1 = lambda_1

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
                default=self.lambda_1["default"])
        
        units_h = self.units_h
        
        layers_h = self.layer_h
        
        model = Sequential()        
        # First layer
        model.add(Bidirectional(LSTM(units=units_h, return_sequences=True), 
                                input_shape=(None, self.input_dim)))        
        # Hidden layer(s)
        if layers_h > 0:
            for i in range(layers_h):
                model.add(Bidirectional(LSTM(units=units_h,
                                             return_sequences=True)))                
        # Last layer
        model.add(TimeDistributed(Dense(self.output_dim, activation='linear'))) 
        # Optimizer
        opt=Adam(learning_rate=learning_r)   

        if self.loss_f == 'output_length_constr':
            self.loss_f = output_len_constr_loss(self.constraints, lambda_1)

        model.compile(
            optimizer=opt,
            loss=self.loss_f,
            metrics=[MeanSquaredError(), RootMeanSquaredError()]
        )
        
        return model