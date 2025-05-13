import os
import pandas as pd
if os.path.exists("muse_v3.csv"):
    df = pd.read_csv("muse_v3.csv")
    print("file found")
else:
    print('file not found')    