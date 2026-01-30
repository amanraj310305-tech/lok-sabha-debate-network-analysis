import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import re
import glob
import os
from nltk.corpus import stopwords
from collections import Counter
import community.community_louvain as community_louvain

nltk.download("stopwords")

DATASET_FOLDER = r"C:\Users\Hp\sample_project\archive\Loksabha_debate"  # ALL YEARS
TOP_WORDS = 60
WINDOW_SIZE = 5
MIN_EDGE_WEIGHT = 12
OUTPUT_PREFIX = "output/lok_sabha_all_years"


print("Loading all debate files...")

files = glob.glob(DATASET_FOLDER + "/**/*", recursive=True)

texts = []
for f in files:
    if os.path.isfile(f):
        with open(f, "r", errors="ignore") as file:
            texts.append(file.read())

texts = pd.Series(texts)
print("Debate files loaded:", len(texts))

stop = set(stopwords.words("english"))

extra_stop = {
    "shri","hon","sir","madam","member","members","sabha","house",
    "singh","kumar","laid","address","constituency","minister",
    "government","bill","act","shall","said","also","like","need",
    "first","year","india","indian","provide","brought","many","much",
    "take","make","one","two","three","four","five","mantri","pradhan",
    "yojana","scheme","schemes","sector","come","going","very","state",
    "states","central","committee","statement","funds","project"
}

stop = stop.union(extra_stop)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    return [w for w in words if w not in stop and len(w) > 3]

documents = texts.apply(clean_text)

all_words = []
for d in documents:
    all_words.extend(d)

vocab = set([w for w,_ in Counter(all_words).most_common(TOP_WORDS)])
print("Vocabulary size:", len(vocab))

G = nx.Graph()

for doc in documents:
    words = [w for w in doc if w in vocab]
    for i in range(len(words)):
        for j in range(i+1, min(i+WINDOW_SIZE, len(words))):
            w1,w2 = words[i], words[j]
            if G.has_edge(w1,w2):
                G[w1][w2]["weight"] += 1
            else:
                G.add_edge(w1,w2, weight=1)

# Filter weak edges
for u,v,w in list(G.edges(data="weight")):
    if w < MIN_EDGE_WEIGHT:
        G.remove_edge(u,v)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


partition = community_louvain.best_partition(G, weight="weight")
modularity = community_louvain.modularity(partition, G)
print("\nPolarization (Modularity):", round(modularity,3))

bet = nx.betweenness_centrality(G)

top = sorted(bet.items(), key=lambda x:x[1], reverse=True)[:10]

print("\nTop bridging terms:")
for w,s in top:
    print(w, round(s,4))


plt.figure(figsize=(12,10))
pos = nx.spring_layout(G, k=0.5)

nx.draw_networkx_nodes(G, pos,
                       node_color=list(partition.values()),
                       node_size=180,
                       cmap=plt.cm.tab10)

nx.draw_networkx_edges(G, pos, alpha=0.2)

top_nodes = sorted(bet, key=bet.get, reverse=True)[:15]
labels = {n:n for n in top_nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

plt.title("Lok Sabha Debate Network (All Years)")
plt.axis("off")
plt.savefig(f"{OUTPUT_PREFIX}_network.png", dpi=300)
plt.show()

sizes = Counter(partition.values())

plt.figure()
plt.bar(sizes.keys(), sizes.values())
plt.title("Community Sizes")
plt.savefig(f"{OUTPUT_PREFIX}_communities.png", dpi=300)
plt.show()



pd.DataFrame({
    "word": list(bet.keys()),
    "betweenness": list(bet.values()),
    "community": [partition[w] for w in bet]
}).to_csv(f"{OUTPUT_PREFIX}_metrics.csv", index=False)

print("\nSaved:")
print(f"{OUTPUT_PREFIX}_network.png")
print(f"{OUTPUT_PREFIX}_communities.png")
print(f"{OUTPUT_PREFIX}_metrics.csv")

print("\nPIPELINE COMPLETE.")
