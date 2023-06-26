

def check_dict_for_keys(dict, keys):
    for key in keys:
        _tmpdict = dict
        for p_key in key.split('.'):
            if p_key not in _tmpdict:
                return False
            _tmpdict = _tmpdict[p_key]
    return True
