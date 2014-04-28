"""
load_configuration.py

This file loads hardcoded filepaths
from a json (such as 'tests/test_configuration.json')
into a python dict such that they can be accessed in
various modules.
"""

import json


def load_configuration():
    config_file = 'tests/test_configuration.json'
    config_dict = json.loads(open(config_file).read())
    config_dict = dict((str(k), str(v) if isinstance(v, unicode) else v) for k, v in config_dict.items())
    print config_dict
    return config_dict

    
    ##If json doesnt seem like a good idea,
    ##then this loop allows reading from a 
    ##custom variable=filepath
    ##format

    # cfg = {};
    # with open('tests/test_configuration.txt', 'rb') as f:
    #     reader = csv.reader(f, delimiter='=', quoting=csv.QUOTE_NONE)
    #     for row in reader:
    #         cfg[str(row[0].strip())] = row[1].strip()
    #return cfg