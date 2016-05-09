# volatility
A model for forecasting stock volatility.

This software constructs an LTSM based RNN to forecast stock volatility
and tests it against some benchmarks. This is not production code, but
part of a research project.

It requires Python 3.4, TensorFlow 0.8, Sqilite3 and arch 3.0 (python library)

This directory does not include the CRSP US Stock Database CSV files
required to build the SQL database. This data is only available under licence.
