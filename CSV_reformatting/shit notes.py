    #
    #
    #
    # img_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\"
    # labels_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\test_labels.csv"
    #
    # chexpert_loader = DataRetrieval(dataset="chexpert", labels_dir=labels_dir, img_dir=img_dir, batch_size=batch_size,
    #                                 perc_dataset=perc_dataset, size=img_size, s=0, verbose=True, num_workers=4).loader
    #
    # # create an empty list to store the embeddings and labels
    # embeddings_list = torch.empty(0, 128)
    # labels_list = torch.empty(0, 5)
    # # create a counter for the number of batches processed
    # batch_counter = 0
    # # set the checkpoint interval (in number of batches)
    # checkpoint_interval = 5000
    #
    # with torch.no_grad():
    #     for images, labels in tqdm(chexpert_loader, desc="Creating Embedding Chexpert"):
    #
    #
    #         images = images.to(device)
    #         image_embeddings = resnet50(images)
    #                  embeddings_list = torch.cat([embeddings_list, image_embeddings.cpu()], dim=0)
    #         labels_list = torch.cat([labels_list, labels.cpu()], dim=0)
    #
    #         # increment the batch counter
    #         batch_counter += 1
    #
    #         # save the embeddings and labels periodically as checkpoints
    #         if batch_counter % checkpoint_interval == 0:
    #
    #             # create a torch dataset that contains the embeddings and labels
    #             dataset = data_utils.TensorDataset(embeddings_list, labels_list)
    #
    #             # save the dataset as a torch tensor file
    #             checkpoint_filename = "new_embeddingDataset\\test\\"f"embeddings_dataset_{batch_counter}.pt"
    #             torch.save(dataset, checkpoint_filename)
    #
    #             # clear the embeddings and labels lists to save memory
    #             embeddings_list = torch.empty(0, 128)  # []
    #             labels_list = torch.empty(0, 5)  # []
    #
    #             # print a message to indicate that a checkpoint has been saved
    #             print(f"Saved checkpoint {batch_counter}: {checkpoint_filename}")
    #
    #     # stack any remaining embeddings and labels into tensors and save them as a final checkpoint
    #     if len(embeddings_list) > 0 and len(labels_list) > 0:
    #
    #         # create a torch dataset that contains the embeddings and labels
    #         dataset = data_utils.TensorDataset(embeddings_list, labels_list)
    #
    #         # save the dataset as a torch tensor file
    #         checkpoint_filename = "new_embeddingDataset\\train\\embeddings_dataset_final.pt"
    #         torch.save(dataset, checkpoint_filename)
    #
    #         embeddings_list = torch.empty(0, 128)
    #         labels_list = torch.empty(0, 5)
    #         # print a message to indicate that the final checkpoint has been saved
    #         print(f"Saved final checkpoint: {checkpoint_filename}")
    #
torch.max(emb_1 - emb_2)
Out[5]: tensor(3.1292e-07)
torch.sum(emb_1 - emb_2)
Out[6]: tensor(-7.6951e-06)


torch.max(emb_1 - emb_2)
Out[8]: tensor(3.1292e-07)
torch.sum(emb_1 - emb_2)
Out[9]: tensor(-7.6951e-06)