python testLSTM.py --case OLC --fail_case drop_jump --tuned
python testLSTM.py --case OAC --fail_case drop_jump --tuned
python testLSTM.py --case OLC_OAC --fail_case drop_jump --tuned
# Note the IOC model had to be tuned using the trainLSTM script due to increased runtimes when using functional API
# Hence the different format and location
python testLSTM.py --case io_best --fail_case drop_jump