import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_curve
from torch import nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from DataRetrieval import DataRetrieval, basic_create_prompts, create_prompts
from health_multimodal.image import get_biovil_resnet
from health_multimodal.text.utils import get_cxr_bert, get_cxr_bert_inference

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # todo swap with right device
    print("running on:", device)

    resnet_path = "C:\\Users\\mistr\\OneDrive\\Desktop\\mcs_only_local\\hi-ml\\hi-ml-multimodal\\src\\biovil_image_resnet50_proj_size_128.pt"

    resnet50 = get_biovil_resnet(pretrained=resnet_path)
    resnet50.train(mode=False, my_freeze=True)
    resnet50.eval()
    if not resnet50.training:
        print("Res-Net is in eval mode")
    resnet50.to(device)

    cxr_bert = get_cxr_bert_inference()
    if cxr_bert.is_in_eval():
        print("Bert is in eval mode")

    class_names = ["Pleural Effusion", "Pneumothorax", "Atelectasis", "Pneumonia", "Consolidation"]
    prompt_names = ["pleural effusion", "pneumothorax", "atelectasis", "pneumonia", "consolidation"]
    img_size = 512
    perc_dataset = 1
    dataset = "real"  # "small" / "real" xxx se small o real
    split = "test"  # "val" "test" "train" "val-test" xxx main cartella del writer
    batch_size = 2  # 8  / 224:16 320:8 512:2 768:1

    img_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\"
    labels_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\test_labels_NO_maiusc.csv"
    basic_prompts = False
    if basic_prompts:
        str_basic = "NO"
    else:
        str_basic = "mean"
    do_writer = True
    if do_writer:  # NO or mean
        ll = "mcs-"+str_basic+"-prompt-NO-maiusc"
        additional_info = ""  # "-prompt-eng-mcs-with-mean"  # xxx pezzettino aggiuntivo in fondo
        name_summary = str(
            "runs-fix/chex-" + str(dataset) + "/" + str(split) + "/" + str(img_size) + "-" + str(ll) + str(
                additional_info))
        print("name summary", name_summary)
        writer = SummaryWriter(name_summary)

    chexpert_loader = DataRetrieval(dataset="chexpert", labels_dir=labels_dir, img_dir=img_dir, batch_size=batch_size,
                                    perc_dataset=perc_dataset, size=img_size, s=0, verbose=True, num_workers=4).loader
    y_true = []
    y_pred = []
    if basic_prompts:
        prompts = basic_create_prompts(prompt_names)
    else:
        prompts = create_prompts(prompt_names)

    print("img size:", img_size)
    print("single_prompts:", basic_prompts)
    print("do_writer:", do_writer)
    with torch.no_grad():
        for images, labels in tqdm(chexpert_loader, desc="Evaluating Zero-shot on chexpert"):

            # image0_to_plt = embs[0]
            # image0_to_plt = image0_to_plt.permute(1, 2, 0)
            # # Plot the RGB tensor
            # plt.imshow(image0_to_plt)
            # plt.show()

            images = images.to(device)
            labels = labels.to(device)
            image_embeddings = resnet50(images)
            image_embeddings = F.normalize(image_embeddings, dim=-1)

            predicted_labels = torch.zeros(labels.shape[0], 5).to(device)

            # Loop through each label
            for i, label_name in enumerate(chexpert_loader.dataset.label_names):
                # Get the positive and negative prompts for the label
                # pos_prompt = prompts[label_name]["positive"]
                # neg_prompt = prompts[label_name]["negative"]
                pos_prompt = prompts[prompt_names[i]]["positive"]
                neg_prompt = prompts[prompt_names[i]]["negative"]

                # pos_prompt = pos_prompt.to(device)
                # neg_prompt = neg_prompt.to(device)
                # Encode the positive and negative prompts
                pos_prompt_embedding = cxr_bert.get_embeddings_from_prompt(pos_prompt, normalize=False)
                assert pos_prompt_embedding.shape[0] == len(pos_prompt)
                if not basic_prompts:
                    pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
                pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(device)

                neg_prompt_embedding = cxr_bert.get_embeddings_from_prompt(neg_prompt, normalize=False)
                assert neg_prompt_embedding.shape[0] == len(neg_prompt)
                if not basic_prompts:
                    neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
                neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(device)

                # Calculate the similarities between the image and the positive and negative prompts
                pos_similarities = torch.matmul(image_embeddings, pos_prompt_embedding.T)
                neg_similarities = torch.matmul(image_embeddings, neg_prompt_embedding.T)

                pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
                neg_similarities = neg_similarities.reshape(-1, 1)

                # Take the maximum similarity as the predicted label
                predicted_labels[:, i] = torch.argmax(torch.cat([neg_similarities, pos_similarities], dim=1), dim=1)

            # Convert the predicted labels to a numpy array
            predicted_labels_np = predicted_labels.cpu().numpy()

            # Append the true and predicted labels to the lists
            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted_labels_np)

    # Concatenate the true and predicted labels
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Calculate the metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    auroc = roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    # Calculate precision-recall curve for each class
    precision_curve = []
    recall_curve = []
    for i in range(5):
        precision_i, recall_i, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
        precision_curve.append(precision_i)
        recall_curve.append(recall_i)

    # Print the results
    if do_writer:
        writer.add_scalar("Accuracy", accuracy)
        writer.add_scalar("F1 score", f1)
        writer.add_scalar("AUROC", auroc)
        for i in range(5):
            writer.add_scalar("Class {} Accuracy".format(i), accuracy_score(y_true[:, i], y_pred[:, i]))
            writer.add_scalar("Class {} Precision".format(i),
                              precision_score(y_true[:, i], y_pred[:, i], average="weighted"))
            writer.add_scalar("Class {} Recall".format(i), recall_score(y_true[:, i], y_pred[:, i], average="weighted"))

        # add the precision-recall curves to the SummaryWriter
        for i in range(5):
            fig = plt.figure()
            plt.plot(recall_curve[i], precision_curve[i], label='Precision-Recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve for Class ' + str(i))
            plt.legend(loc="lower left")
            writer.add_figure('Precision-Recall Curve for Class ' + str(i), fig)
            # add the precision and recall values as time series to the SummaryWriter
        # for i in range(5):
        #     for j in range(len(precision_curve[i])):
        #         writer.add_scalar('Precision/Class_' + str(i), precision_curve[i][j], j)
        #         writer.add_scalar('Recall/Class_' + str(i), recall_curve[i][j], j)
        writer.close()

    print("Accuracy: {:.4f}".format(accuracy))
    print("F1 score: {:.4f}".format(f1))
    print("AUROC: {:.4f}".format(auroc))
