import pandas as pd

# input_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\chexpert\\train.csv"
# output_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\chexpert\\train2.csv"
# #
# # Read in the CSV file using pandas
# df = pd.read_csv(input_dir)
#
# # Drop rows with any NaN values
# df.dropna(subset=['Pleural Effusion', 'Pneumothorax', 'Atelectasis', 'Pneumonia', 'Consolidation'], inplace=True)
# # Save the cleaned dataframe to a new CSV file
# df.to_csv(output_dir, index=False)


# input_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\chexpert\\valid2.csv"
# output_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\chexpert\\valid2-frontal.csv"
# input_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\val_labels.csv"
# output_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\val_labels_frontal.csv"
#
# # Read the CSV file
# df = pd.read_csv(input_dir)
#
# # Filter the rows based on the value of "Frontal/Lateral" column
# df = df[df['Frontal/Lateral'] == 'Frontal']
#
# # Save the filtered data to a new CSV file
# df.to_csv(output_dir, index=False)

# input_dir = "D:\\CheXpert-v1.0\\CheXpert-v1.0\\train_visualCheXbert_fixed.csv"
# output_dir = "D:\\CheXpert-v1.0\\CheXpert-v1.0\\train_visualCheXbert_fixed_frontal.csv"
# # leggi il file CSV
# df = pd.read_csv(input_dir)
#
# # seleziona solo le righe che contengono la sottostringa "frontal" nella prima colonna
# df_frontal = df[df['Path'].str.contains('frontal')]
#
# # salva il nuovo CSV contenente solo le righe selezionate
# df_frontal.to_csv(output_dir, index=False)

import pandas as pd
input_csv = "D:\\CheXpert-v1.0\\CheXpert-v1.0\\train_visualCheXbert_fixed.csv"
out_csv = "D:\\CheXpert-v1.0\\CheXpert-v1.0\\train_visualCheXbert_fixed_SOLO_LABEL_CHEX.csv"
# Read the input CSV file
df = pd.read_csv(input_csv)
# Select the desired columns
# columns_to_keep = ["Pleural Effusion", "Pneumothorax", "Atelectasis", "Pneumonia", "Consolidation"]
columns_to_keep = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

df = df[columns_to_keep]
# Save the output CSV file
df.to_csv(out_csv, index=False)