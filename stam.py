import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

e = pd.read_excel('Summary_data_BRV_ISO.xlsx', sheet_name='Young vs. Old Max')

sns.set_theme(color_codes=True)
tips = sns.load_dataset("tips")
sns.lmplot(x="total_bill", y="tip", hue="smoker",
           col="time", row="sex", data=tips)
plt.show()
a=1