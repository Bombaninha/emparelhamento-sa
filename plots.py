#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

for i in range(0, 12, 4):
    print(i)
    plt.figure(figsize=(10,5))
    dados: pd.DataFrame = pd.read_csv("results/results_cmp.csv")
    df = pd.melt(dados[i:i+4], id_vars='Instância', var_name='Métodos', value_name="Arestas na Solução")
    g = sns.barplot(x='Instância', y="Arestas na Solução", hue="Métodos", data=df)
    plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
    plt.tight_layout()
    plt.savefig(fname=f"results/Bar{i+1}_{i+4}.png", dpi=150)
