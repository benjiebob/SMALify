import numpy as np
import pickle as pkl

model_path = "data/SMALST/smpl_models/my_smpl_00781_4_all.pkl"

with open(model_path, 'rb') as f:
    u = pkl._Unpickler(f)
    u.encoding = 'latin1'
    dd = u.load()
    print (dd['f'])