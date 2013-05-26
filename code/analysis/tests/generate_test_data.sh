#!/bin/bash

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=social --mouse-number=0001

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=novel --mouse-number=0001

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=social --mouse-number=0002

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=novel --mouse-number=0002

cd ..
python preprocessing.py /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/  --generate-hdf5 -f tests/test_analysis_filenames.txt --out-path /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/test_data_raw.h5
