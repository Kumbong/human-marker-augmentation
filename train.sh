#!/bin/bash
python tuneHyperParametersLSTM.py --case 1 >> log1.txt
python tuneHyperParametersLSTM.py --case 2 >> log2.txt
python trainLSTM.py --case "io_best" >> log_io_best.txt