#!/bin/bash

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=social --mouse-number=0001 --mouse-type=GC5 --exp-date=20130524

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=novel --mouse-number=0001 --mouse-type=GC5 --exp-date=20130524

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=social --mouse-number=0002 --mouse-type=GC5 --exp-date=20130524 --ts-type='point_process'

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=novel --mouse-number=0002 --mouse-type=GC5 --exp-date=20130524 --ts-type='point_process'


python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=novel --mouse-number=0003 --mouse-type=GC5_NAcprojection --exp-date=20130523 --ts-type='point_process'

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=social --mouse-number=0003 --mouse-type=GC5_NAcprojection --exp-date=20130523 --ts-type='point_process'

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=novel --mouse-number=0004 --mouse-type=GC5_NAcprojection --exp-date=20130523 --ts-type='point_process'

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=social --mouse-number=0004 --mouse-type=GC5_NAcprojection --exp-date=20130523 --ts-type='point_process'

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=novel --mouse-number=0005 --mouse-type=GC5 --exp-date=20130524 --ts-type='simple' --tail=2

python generate_test_data.py -o /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/ --exp-type=social --mouse-number=0005 --mouse-type=GC5 --exp-date=20130524 --ts-type='simple' --tail=2

cd ..
python preprocessing.py /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/  --generate-hdf5 -f tests/test_analysis_filenames.txt --out-path /Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/analysis/tests/test_data/test_data_raw.h5
