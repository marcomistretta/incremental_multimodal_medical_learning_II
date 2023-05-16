import numpy as np
import playsound
import torch
from torch import nn

from Trainer import Trainer

# SET REPRODUCIBILITY
seed_value = 27
torch.manual_seed(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on:", device)

    batch_size = 6144  # 4096, 6144 8192, 8192 (val e test sono settati di default a 1024 per avere dei plot meno rumorosi)
    lr = 0.0001  # 0.001  # 0.1  # 0.0001, 1, 30
    parts = 20  # 5, 10, 20
    epochs = 10  # 10
    single_prompt = False  # False-->multiple True-->single
    chex_competition = True  # True, False
    xrays_position = "frontal"  # "all", "frontal", "lateral"
    loss_name = "standard"  # "standard", "opzione2" "opzione2variant", "bce"

    CONTINUAL_LEARNING = None  # "profCL"  # "myCL"  # "myCL", "profCL", None
    threshold = 0.01
    ratio = True
    adder = 0.001
    threshold_scheduling = True

    mode = "data-inc"

    writer, class_names, train_loader, val_loader, test_loader, prompts, plot_tsne_array = Trainer.preprocessing_data_incremental(
        chex_competition, xrays_position, single_prompt, batch_size, lr,
        parts, epochs, loss_name, mode, CONTINUAL_LEARNING, ratio,
        threshold, threshold_scheduling, adder)

    criterion = nn.BCEWithLogitsLoss()
    trainer = Trainer(single_prompt, prompts, class_names, loss_name, lr, device, writer)

    count = 0
    try:
        for part in range(1, parts + 1):
            for epoch in range(1, epochs + 1):
                count += 1
                threshold = threshold + adder
                if threshold_scheduling and CONTINUAL_LEARNING is not None:
                    writer.add_scalar("monitor-resets/threshold-scheduling", threshold, count)
                if CONTINUAL_LEARNING == "profCL":
                    trainer.model_copy()
                trainer.train(train_loader[part - 1], criterion, epoch,
                              CONTINUAL_LEARNING, threshold, part=part, epochs=epochs, actual_task=part)
                if CONTINUAL_LEARNING == "profCL":
                    trainer.profIncremental(epoch, epochs, part, threshold)
                torch.cuda.empty_cache()
            train_loader[part - 1] = None
            trainer.val(val_loader, criterion, part, parts, mode="data-inc", tasks_order=part)
            trainer.test(test_loader, criterion, part, parts, mode="data-inc", tasks_order=part, plot_tsne_array=plot_tsne_array)
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        # Play a sound to notify the end of the execution
        if epochs > 0:
            trainer.save()
        playsound.playsound("mixkit-correct-answer-tone-2870.wav")
