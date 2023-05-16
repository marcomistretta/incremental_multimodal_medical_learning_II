import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

columns = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

# df = pd.read_csv("C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\train_visualCheXbert.csv")
# df = pd.read_csv("C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\val_labels.csv")
df = pd.read_csv("C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\test_labels.csv")

if False:
    # Select columns of interest
    df = df[columns]

    # Define condition labels for each column
    condition_labels = {"Atelectasis": "ATEL", "Cardiomegaly": "CARD", "Consolidation": "CONS", "Edema": "EDEMA",
                        "Pleural Effusion": "PLEU"}

    # Create a new column with the combined condition labels for each row
    df["condition_labels"] = "SS" + df.apply(lambda row: '+'.join([condition_labels[column] if row[column] == 1 else "" for column in columns]), axis=1) + "EE"


    # Count the frequencies of each pattern in the rows
    pattern_counts = df["condition_labels"].value_counts() / len(df)

    # Create a bar plot of the pattern frequencies
    fig, ax = plt.subplots(figsize=(8, 6))
    pattern_counts.plot(kind='bar', ax=ax)

    # Set the plot title and axes labels
    ax.set_title('Test Pattern Frequencies')
    ax.set_xlabel('Condition Combinations')
    ax.set_ylabel('Frequency')

    # Adjust the x-axis tick labels to show the presence of each condition
    tick_labels = [
        label.replace("++++", "+").replace("+++", "+").replace("++", "+").replace("SS+", "").replace("+EE", "").replace("SS", "").replace("EE", "") for label in pattern_counts.index]
    ax.set_xticklabels(tick_labels)

    # Adjust the margins of the plot
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.5)

    plt.tight_layout()
    plt.savefig("faq-patterns/test_patterns.png")
    # Show the plot
    plt.show()

    df.to_csv("faq-patterns/count_test_patterns.csv", index=False)

for cc in columns:
    print(cc, np.sum(df[cc]), np.sum(df[cc]) / len(df), len(df)-np.sum(df[cc]), (len(df)-np.sum(df[cc]))/len(df))