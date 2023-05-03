import numpy as np
import torch
import playsound
from torch import nn

from Trainer import Trainer

# xxx SET REPRODUCIBILITY
seed_value = 27
torch.manual_seed(seed_value)
import random

random.seed(seed_value)
np.random.seed(seed_value)
# xxx

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on:", device)

    batch_size = 6144  # 4096, 6144 8192, 8192 (val e test sono settati di default a 1024 per avere dei plot meno rumorosi)
    lr = 0.0001  # 0.1  # 0.0001, 1, 30
    epochs = 10  # 10
    single_prompt = False  # False-->multiple True-->single
    chex_competition = True  # True, False
    xrays_position = "all"  # "all", "frontal", "lateral"
    loss_name = "standard"  # "bce-only-pp"  # standard, opzione2, opzione2variant cosine
    writer, class_names, train_loader, val_loader, test_loader, prompts = Trainer.preprocessing(chex_competition,
                                                                                                xrays_position,
                                                                                                single_prompt,
                                                                                                batch_size, lr, epochs,
                                                                                                loss_name)

    if loss_name == "standard":
        print("*** BCEWithLogitsLoss ***")
        # define the percentages of positive examples for each class
        criterion = nn.BCEWithLogitsLoss()  # nn.BCEWithLogitsLoss() nn.CrossEntropyLoss
    elif loss_name == "ce":
        print("*** 5 CrossEntropyLoss ***")
        criterion = [nn.CrossEntropyLoss() for i in range(5)]
    # elif loss_name == "5-bce-only-pp":
    #     print("*** 5 BCEWithLogitsLoss SOLO PROMPTS POSITIVI ***")
    #     criterion = [nn.BCEWithLogitsLoss() for i in range(5)]
    elif loss_name == "bce-only-pp":
        print("*** BCEWithLogitsLoss SOLO PROMPTS POSITIVI ***")
        criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(single_prompt, prompts, class_names, loss_name, lr, device, writer)

    # XXX run
    CONTINUAL_LEARNING = None  # "myCL", "profCL"
    threshold = 0.5

    try:
        if CONTINUAL_LEARNING is not None:
            print("**** CONTINUAL LEARNING ****")
            print("--->", CONTINUAL_LEARNING)
        else:
            if epochs == 0:
                print("**** zero-shot ****")
            if epochs > 0:
                print("**** joint-train ****")
        if epochs > 0:
            for epoch in range(1, epochs + 1):
                trainer.train(train_loader, criterion, epoch, CONTINUAL_LEARNING, threshold)
                trainer.val(val_loader, criterion, epoch, epochs, mode="joint", tasks_order=None)
                trainer.test(test_loader, criterion, epoch, epochs, mode="joint", tasks_order=None)
        else:
            trainer.val(val_loader, criterion, 0, 0, mode="zero", tasks_order=None)
            trainer.test(test_loader, criterion, 0, 0, mode="zero", tasks_order=None)
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        # Play a sound to notify the end of the execution
        playsound.playsound("mixkit-correct-answer-tone-2870.wav")
