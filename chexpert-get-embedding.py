import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.utils.data as data_utils

from DataRetrieval import DataRetrieval
from health_multimodal.image import get_biovil_resnet

CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # todo swap with right device
    print("running on:", device)

    resnet_path = "C:\\Users\\mistr\\OneDrive\\Desktop\\mcs_only_local\\hi-ml\\hi-ml-multimodal\\src\\biovil_image_resnet50_proj_size_128.pt"
    # out_path = "embeddingDataset\\512-not-normalize\\dataset.pt"
    resnet50 = get_biovil_resnet(pretrained=resnet_path)
    resnet50.train(mode=False, my_freeze=True)
    resnet50.eval()
    if not resnet50.training:
        print("Res-Net is in eval mode")
    resnet50.to(device)

    img_normalize = False
    img_size = 512
    batch_size = 1
    perc_dataset = 1
    img_dir = "D:\\CheXpert-v1.0\\CheXpert-v1.0\\"
    labels_dir = "D:\\CheXpert-v1.0\\CheXpert-v1.0\\train_visualCheXbert_fixed_frontal.csv"
    # img_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\"
    # labels_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\val_labels_frontal.csv"
    # labels_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\val_labels_frontal.csv"

    chexpert_loader = DataRetrieval(dataset="chexpert", labels_dir=labels_dir, img_dir=img_dir, batch_size=batch_size,
                                    perc_dataset=perc_dataset, size=img_size, s=0, verbose=True, num_workers=4).loader

    # create an empty list to store the embeddings and labels
    embeddings_list = []
    labels_list = []
    # create a counter for the number of batches processed
    batch_counter = 0
    # set the checkpoint interval (in number of batches)
    checkpoint_interval = 5000  # saranno pezzetti da 10'000 immagini

    with torch.no_grad():
        for images, labels in tqdm(chexpert_loader, desc="Creating Embedding Chexpert"):

            # encode the embs
            images = images.to(device)
            image_embeddings = resnet50(images)
            if img_normalize:
                image_embeddings = F.normalize(image_embeddings, dim=-1)

            # append the embeddings and labels to the respective lists
            embeddings_list.append(image_embeddings.cpu())
            labels_list.append(labels.cpu())

            # increment the batch counter
            batch_counter += 1

            # save the embeddings and labels periodically as checkpoints
            if batch_counter % checkpoint_interval == 0:
                # stack the embeddings and labels into tensors
                embeddings_tensor = torch.cat(embeddings_list, dim=0)
                labels_tensor = torch.cat(labels_list, dim=0)

                # create a torch dataset that contains the embeddings and labels
                dataset = data_utils.TensorDataset(embeddings_tensor, labels_tensor)

                # save the dataset as a torch tensor file
                checkpoint_filename = "embeddingDataset\\train\\512-chex-not-normalize-frontal\\" + f"embeddings_dataset_{batch_counter}.pt"
                torch.save(dataset, checkpoint_filename)

                # clear the embeddings and labels lists to save memory
                embeddings_list.clear()
                labels_list.clear()

                # print a message to indicate that a checkpoint has been saved
                print(f"Saved checkpoint {batch_counter}: {checkpoint_filename}")

        # stack any remaining embeddings and labels into tensors and save them as a final checkpoint
        if len(embeddings_list) > 0 and len(labels_list) > 0:
            # stack the embeddings and labels into tensors
            embeddings_tensor = torch.cat(embeddings_list, dim=0)
            labels_tensor = torch.cat(labels_list, dim=0)

            # create a torch dataset that contains the embeddings and labels
            dataset = data_utils.TensorDataset(embeddings_tensor, labels_tensor)

            # save the dataset as a torch tensor file
            checkpoint_filename = "embeddingDataset\\train\\512-chex-not-normalize-frontal\\" + f"embeddings_dataset_final.pt"
            torch.save(dataset, checkpoint_filename)

            embeddings_list.clear()
            labels_list.clear()
            # print a message to indicate that the final checkpoint has been saved
            print(f"Saved final checkpoint: {checkpoint_filename}")


    # create a data loader for the embeddings dataset
    # xxx batch_size = 16
    # xxx loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # xxx for embeddings, labels in loader:
    '''
    # Load the dataset from a file
    dataset = torch.load("all_embeddings.pt")
    # Create a torch loader from the dataset
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # Iterate through the loader to get batches of embeddings
    for embeddings_batch in loader:
        # Do something with the embeddings batch
        pass
    '''
