import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


g = nx.karate_club_graph()
fig, ax = plt.subplots(1, 1, figsize=(8, 6));
nx.draw_networkx(g, ax=ax)
plt.show()

from IPython import embed

a = "I will be accessible in IPython shell!"

embed()