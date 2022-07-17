# %%
# LIBRARY
import pandas as pd
import umap
import umap.plot

# %%
# READ DATA
df_fish = pd.read_csv('Fish.csv')
df_fish = df_fish[df_fish['Weight'] != 0]

# %%
mapper = (umap
          .UMAP(min_dist=0, n_neighbors=10)
          .fit(df_fish[['Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']].values)
          )
umap.plot.points(mapper, labels=df_fish['Species'])
