# CS-230: Human body marker augmentation

### Install requirements
- TODO
1. Install [Anaconda](https://www.anaconda.com/)
2. Fork and clone the repository to your machine.
3. Open the Anaconda command prompt and create a conda environment: `conda create -n cs-230-project`
4. Activate the environment: `conda activate cs-230-project`
5. Install (tensorflow with GPU support)[https://www.tensorflow.org/install/pip]
6. Install other dependencies. Navigate to the local directory where the repository is cloned, then: `python -m pip install -r requirements.txt`

### Overview files
**Main files**:
- `trainLSTM.py`: script to train LSTM model.
- `evaluateLSTM.py`: script to evaluate LSTM model.
- `myModels.py`: script describing the model architecture.
- `myDataGenerator.py`: script describing the data generator.
- `mySettings.py`: script with some tunable model settings.
**Other files (you should not need to interact with these files)**:
- `splitData.py`: script to split the data into different sets.
- `infoDatasets.py`: some details about the dataset.
- `utilities.py`: various utilities.
