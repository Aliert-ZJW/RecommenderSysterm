import numpy as np
import pandas as pd
data = pd.DataFrame({"ID":[1001,1002,1003,1004],
                    "q":["0","1","-1","-1"],
                     "w":["0","1","-1","-1"],
                     "e":["0","1","-1","-1"],
                     "r":["0","1","-1","-1"],
                     "o":["0","1","-1","-1"],})
print(data)
print('----------')
print(data[:, :1])