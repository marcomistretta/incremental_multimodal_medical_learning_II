# import pandas as pd
#
# # Leggi il file CSV in un dataframe
# df = pd.read_csv("C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\test_labels.csv")
#
# # Conta il numero di "0" e "1" per ogni cc specificata
# columns = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
# counts = {}
# for cc in columns:
#     counts[cc] = df[cc].value_counts()
#
# # Stampa i risultati
# print("len:", len(df))
# for cc in columns:
#     print(f"Counts for {cc}:")
#     print(counts[cc])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\test_labels.csv")
df = pd.read_csv("C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\train_visualCheXbert.csv")
# Conta il numero di "0" e "1" per ogni cc specificata

columns = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
df = df[columns]

# df.to_csv("faq-patterns/count_train_patterns.csv", index=False)
# count the frequencies of each pattern in the rows

counts = df.apply(lambda x: ''.join(x.astype(str)), axis=1).value_counts()
for cc in columns:
    print(np.sum(df[cc]) / len(df))
# create a bar plot of the pattern frequencies
fig, ax = plt.subplots(figsize=(8, 6))

counts.plot(kind='bar', ax=ax)

# set the plot title and axes labels
ax.set_title('Pattern Frequencies')
ax.set_xlabel('Pattern')
ax.set_ylabel('Frequency')

# adjust the margins of the plot
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
# show the plot
plt.show()
