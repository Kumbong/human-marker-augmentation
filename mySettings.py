# Dense.
def get_dense_settings(a):

    settings = {"0":
                {"idxDatasets": [idx for idx in range(9)],
                 "nFirstUnits": 1000,
                 "nHUnits": 0,
                 "nHLayers": 0,
                 "nEpochs": 20,
                 "batchSize": 64,
                 "idxFold": 0,
                 "dropout_p": 1.39e-6,
                 "L2_p": 1.56e-6,
                 "learning_r": 6.76e-5,
                 "mean_subtraction": True,
                 "std_normalization": True,
                 "noise_magnitude": 0.018,
                 "noise_type": "per_subject_per_marker"}}

    return settings[str(a)]

# LSTM.


def get_lstm_settings(a):

    settings = {
        "reference":
            {'augmenter_type': 'lowerExtremity',
             "poseDetector": 'OpenPose',
             "mean_subtraction": True,
             "std_normalization": True, },
        "0":
            {'augmenter_type': 'lowerExtremity',
             "poseDetector": 'OpenPose',
             "idxDatasets": [idx for idx in range(0, 1)],
             "scaleFactors": [0.9, 0.95, 1., 1.05, 1.1],
             "nHUnits": 96,
             "nHLayers": 2,
             "nEpochs": 50,
             "batchSize": 64,
             "idxFold": 0,
             'learning_r': 5e-05,
             "mean_subtraction": True,
             "std_normalization": True,
             "noise_magnitude": 0.018,
             "noise_type": "per_timestep",
             'nRotations': 1,
             'bidirectional': False},
        "1":
            {'augmenter_type': 'lowerExtremity',
             "poseDetector": 'OpenPose',
             "idxDatasets": [idx for idx in range(0, 1)],
             "scaleFactors": [0.9, 0.95, 1., 1.05, 1.1],
             "nHUnits": 96,
             "nHLayers": 2,
             "nEpochs": 50,
             "batchSize": 64,
             "idxFold": 0,
             "loss_f":  "output_constr",
             'learning_r': 5e-05,
             "lambda_1": 1,
             "lambda_2": 1,
             "lambda_3": 1,
             "mean_subtraction": True,
             "std_normalization": True,
             "noise_magnitude": 0.018,
             "noise_type": "per_timestep",
             'nRotations': 1,
             'bidirectional': False}}

    return settings[str(a)]

# LSTM - Hyperparameters tuning


def get_lstm_tuner_settings(a):

    settings = {
        "0": # Only output length constraints
        {'augmenter_type': 'lowerExtremity',
         "poseDetector": 'OpenPose',
         "idxDatasets": [idx for idx in range(0, 1)],
         "scaleFactors": [0.9, 0.95, 1., 1.05, 1.1],
         "nEpochs": 10,
         "batchSize": 64,
         "idxFold": 0,
         "mean_subtraction": True,
         "std_normalization": True,
         "max_trials": 10,
         "executions_per_trial": 1,
         "nEpochsBest": 15,
         "units_h": 96,
         "layer_h": 2,
         "loss_f": "output_len_constr",
         "lambda_1": {
             "name": 'lambda_1', "min": .01, "max": 10,
             "sampling": "LOG", "default": 1},
         "lambda_2": {
             "name": 'lambda_2', "min": 1e-8, "max": 1e-8,
             "sampling": "LOG", "default": 1e-8},
         "lambda_3": {
             "name": 'lambda_3', "min": 1e-8, "max": 1e-8,
             "sampling": "LOG", "default": 1e-8},
         "learning_r": {
             "name": 'learning_r', "min": 1e-5, "max": 1e-4,
             "sampling": 'LOG', "default": 5e-5},
         "noise_magnitude": 0.018,
         "noise_type": "per_timestep",
         'nRotations': 8,
         'bidirectional': False},

        "1": # Only angular constraints
        {'augmenter_type': 'lowerExtremity',
         "poseDetector": 'OpenPose',
         "idxDatasets": [idx for idx in range(0, 1)],
         "scaleFactors": [0.9, 0.95, 1., 1.05, 1.1],
         "nEpochs": 10,
         "batchSize": 64,
         "idxFold": 0,
         "mean_subtraction": True,
         "std_normalization": True,
         "max_trials": 10,
         "executions_per_trial": 1,
         "nEpochsBest": 15,
         "units_h": 96,
         "layer_h": 2,
         "loss_f": "output_angular_constr",
         "lambda_1": {
             "name": 'lambda_1', "min": 1e-8, "max": 1e-8,
             "sampling": "LOG", "default": 1e-8},
         "lambda_2": {
             "name": 'lambda_2', "min": 0.01, "max": 10.0,
             "sampling": "LOG", "default": 1.0},
         "lambda_3": {
             "name": 'lambda_3', "min": 0.01, "max": 10.0,
             "sampling": "LOG", "default": 1.0},
         "learning_r": {
             "name": 'learning_r', "min": 1e-5, "max": 1e-4,
             "sampling": 'LOG', "default": 5e-5},
         "noise_magnitude": 0.018,
         "noise_type": "per_timestep",
         'nRotations': 8,
         'bidirectional': False},
         
         "2": # output length + angular constraints
        {'augmenter_type': 'lowerExtremity',
         "poseDetector": 'OpenPose',
         "idxDatasets": [idx for idx in range(0, 1)],
         "scaleFactors": [0.9, 0.95, 1., 1.05, 1.1],
         "nEpochs": 10,
         "batchSize": 64,
         "idxFold": 0,
         "mean_subtraction": True,
         "std_normalization": True,
         "max_trials": 5,
         "executions_per_trial": 1,
         "nEpochsBest": 15,
         "units_h": 96,
         "layer_h": 2,
         "loss_f": "output_len_ang_constr",
         "lambda_1": {
             "name": 'lambda_1', "min": 0.01, "max": 10.0,
             "sampling": "LOG", "default": 1.0},
         "lambda_2": {
             "name": 'lambda_2', "min": 0.01, "max": 10.0,
             "sampling": "LOG", "default": 1.0},
         "lambda_3": {
             "name": 'lambda_3', "min": 0.01, "max": 10.0,
             "sampling": "LOG", "default": 1.0},
         "learning_r": {
             "name": 'learning_r', "min": 1e-5, "max": 1e-4,
             "sampling": 'LOG', "default": 5e-5},
         "noise_magnitude": 0.018,
         "noise_type": "per_timestep",
         'nRotations': 8,
         'bidirectional': False}}

    return settings[str(a)]
