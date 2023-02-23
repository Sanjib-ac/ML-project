import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

fileName = 'STK_WME0925570TMOD.xls'
df = pd.read_excel(fileName, sheet_name='800 Ton Test')
print('Columns name: ', df.columns)

# Using Pearson Correlation
plt.figure(figsize=(14, 14))
cor = df.corr(numeric_only=True)
print(cor)
# sns.heatmap(cor)
# plt.show()
