#%%
import pandas as pd

dados: pd.DataFrame = pd.read_csv("results/SA_only.csv")
print(dados.describe().iloc[1:3, :].T)