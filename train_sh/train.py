import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
import logging.handlers
import matplotlib.pyplot as plt
from resources.utils import *
from resources import get_model
import eval as ev
import datetime
import random
import resources.utils.rgb_dataset as dsl
import resources.utils.pose_dataset as pose_dsl

# Save information during training
os.makedirs("logs", exist_ok=True)
log_filename = datetime.datetime.now().strftime("sc_%d_%m_%H_%M_%S.log")
log_filepath = os.path.join("logs", log_filename)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            log_filepath, maxBytes=(1048576 * 5), backupCount=7
        ),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)


# Print something when error occurs
def handle_exception(exc_type, exc_value, exc_traceback):
    # Custom exception handling logic here
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # Call the default handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_exception


def evaluate(
    model, model_name, dataloader, loss_fn, steps, class_info, device="cpu", pbar=True
):
    model.eval()
    logits = []
    current_loss = 0

    with torch.no_grad():
        data_iter = tqdm(dataloader, desc="Evaluating") if pbar else dataloader
        for data in data_iter:
            if model_name != "lstm":
                inputs, labels = data
                inputs = inputs.to(device)
            else:
                x_time, x_spatial, labels = data
                inputs = (x_time.to(device), x_spatial.to(device))

            labels = labels.to(device)
            per_frame_logits = model(inputs)
            current_loss += loss_fn(per_frame_logits, labels).cpu().item()

            logit = per_frame_logits.max(-1)[1].cpu().numpy()
            labels = (labels.to(device).cpu().max(1)[1]).numpy()

            for i in range(len(logit)):
                logits.append([logit[i], labels[i]])

        elog.evaluate("val", steps, logits, class_info)

    current_loss = current_loss / (len(dataloader))
    model.train()
    return current_loss


def run(
    model_name,
    init_lr,
    max_steps,
    device,
    root,
    batch_size,
    n_frames,
    num_workers,
    evaluate_frequently,
    num_gradient_per_update,
    pretrained_path,
    learnig_scheduler_gammar,
    learnig_scheduler_step,
    seed=42,
    cache=None,
    elog=None,
    name="i3d-rgb",
    num_keypoints=None,
    **kwargs,
):
    loss_fn = nn.CrossEntropyLoss()
    HEIGHT = 224
    WIDTH = 224

    if model_name != "lstm":
        dataset = dsl.DSL(
            root,
            height=HEIGHT,
            width=WIDTH,
            n_frames=n_frames,
            random_seed=seed,
            cache_folder=cache,
        )
    else:
        dataset = pose_dsl.DSL(
            root, n_frames=n_frames, random_seed=seed, cache_folder=cache
        )

    class_info = dataset.get_classes()
    person_list = dataset.get_persons()
    random.seed(seed)
    random.shuffle(person_list)
    val_index = int(len(person_list) * 0.7)
    test_index = int(len(person_list) * 0.8)
    train_persons = person_list[:val_index]
    val_persons = person_list[val_index:test_index]
    test_persons = person_list[test_index:]

    save_model = elog.get_path() + f"/{name}_"

    print(
        f"Train: {len(train_persons)}",
        f" Val: {len(val_persons)}",
        f" Test: {len(test_persons)}",
    )

    train_filter = dataset.filter(persons=train_persons)
    val_filter = dataset.filter(persons=val_persons)
    test_filter = dataset.filter(persons=test_persons)

    print(f"Train set size: {len(train_filter['class'])}")
    print(f"Validation set size: {len(val_filter['class'])}")
    print(f"Test set size: {len(test_filter['class'])}")

    with open(args.a_config, "r") as f:
        try:
            spatial_augment = yaml.safe_load(f).get("augment")
        except:
            spatial_augment = None

    train_ds = dataset.get_generator(
        train_filter, mode="train", spatial_augment=spatial_augment, **kwargs
    )

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_ds = dataset.get_generator(val_filter, mode="valid", **kwargs)
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    test_ds = dataset.get_generator(test_filter, mode="valid", **kwargs)
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    dataloaders = {"train": train_dl, "val": val_dl, "test": test_dl}
    num_classes = len(dataset.get_classes())

    model = get_model(
        model_name,
        num_classes,
        finetuning=True,
        num_keypoints=num_keypoints,
        n_frames=n_frames,
        **kwargs,
    )

    model.to(device)

    if len(pretrained_path) > 0:
        model_state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(model_state_dict)

    model = nn.DataParallel(model)

    print(f"Train on {device}")
    print(f"Model name {model_name} ")

    lr = init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0000001)

    steps = 0
    train_loss = []
    valid_loss = []
    best_valid_loss = float("inf")

    for epoch in range(max_steps):
        for phase in ["train", "val"]:
            if phase == "train":
                lr = optimizer.param_groups[0]["lr"]
                optimizer.zero_grad()
                model.train()
                total_loss = 0.0
                num_iter = 0

                pbar = tqdm(
                    enumerate(dataloaders[phase]), total=len(dataloaders[phase])
                )

                for index, data in pbar:
                    num_iter += 1

                    if model_name != "lstm":
                        inputs, labels = data
                        inputs = inputs.to(device)
                    else:
                        x_time, x_spatial, labels = data
                        inputs = (x_time.to(device), x_spatial.to(device))

                    labels = labels.to(device)
                    per_frame_logits = model(inputs)
                    loss = loss_fn(per_frame_logits, labels) / num_gradient_per_update
                    loss.backward()

                    total_loss += loss.data.item()

                    if num_iter == num_gradient_per_update:
                        optimizer.step()
                        optimizer.zero_grad()

                        info = (
                            f"{epoch}/{max_steps}, lr: {lr}, train loss: {total_loss}"
                        )
                        pbar.set_description(info)
                        pbar.set_postfix()

                        train_loss.append(total_loss)
                        num_iter = 0
                        total_loss = 0

                        if (index + 1) % evaluate_frequently == 0:
                            current_valid_loss = evaluate(
                                model,
                                model_name,
                                dataloaders["val"],
                                loss_fn,
                                steps,
                                class_info,
                                device=device,
                                pbar=False,
                            )

                            valid_loss.append(current_valid_loss)
                            pbar.set_postfix_str(
                                f"Valid loss: {round(current_valid_loss, 2)}"
                            )

                            if current_valid_loss < best_valid_loss:
                                torch.save(
                                    model.module.state_dict(), save_model + "best.pt"
                                )
                                best_valid_loss = current_valid_loss

                torch.save(model.module.state_dict(), save_model + "last.pt")

            if phase == "val":
                current_valid_loss = evaluate(
                    model,
                    model_name,
                    dataloaders["val"],
                    loss_fn,
                    steps,
                    class_info,
                    device=device,
                )

                valid_loss.append(current_valid_loss)
                print(f"Val loss: ", round(current_valid_loss, 2))

                if current_valid_loss < best_valid_loss:
                    torch.save(model.module.state_dict(), save_model + "best.pt")
                    best_valid_loss = current_valid_loss

    plt.figure(clear=True)
    plt.plot(train_loss)
    plt.ylabel("Loss")
    plt.xlabel("Batch")
    plt.title("Train loss")
    plt.savefig(save_model + "train_loss.png")
    plt.close()

    plt.figure(clear=True)
    plt.plot(valid_loss)
    plt.ylabel("Loss")
    plt.xlabel("Batch")
    plt.title("Validation loss")
    plt.savefig(save_model + "val_loss.png")
    plt.close()


# Ensure this
# Ensure this script is being run standalone
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models.")
    parser.add_argument(
        "-c",
        "--a_config",
        default="cgp_training.yaml",
        type=str,
        help="yaml file containing parameters",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        type=str,
        help="comma separated list of GPU(s) to use.",
    )
    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="batch size per GPU",
    )
    parser.add_argument(
        "--n-frames",
        default=20,
        type=int,
        help="number of frames per clip",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--evaluate-frequently",
        default=100,
        type=int,
        help="evaluate every n batches",
    )
    parser.add_argument(
        "--num-gradient-per-update",
        default=1,
        type=int,
        help="number of gradient accumulation steps",
    )
    parser.add_argument(
        "--pretrained-path",
        default="",
        type=str,
        help="/kaggle/working/test_data",
    )
    parser.add_argument(
        "--learnig-scheduler-gammar",
        default=0.1,
        type=float,
        help="gammar of learning scheduler",
    )
    parser.add_argument(
        "--learnig-scheduler-step",
        default=1000,
        type=int,
        help="step of learning scheduler",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="seed for random generators",
    )
    parser.add_argument(
        "--cache",
        default=None,
        type=str,
        help="Cache folder",
    )
    parser.add_argument(
        "--elog",
        default=None,
        type=str,
        help="path for saving training information",
    )
    parser.add_argument(
        "--name",
        default="i3d-rgb",
        type=str,
        help="model name",
    )
    parser.add_argument(
        "--num-keypoints",
        default=None,
        type=int,
        help="number of keypoints for the pose estimation",
    )
    parser.add_argument(
        "--model-name",
        default="rgb",
        type=str,
        help="name of the model to train",
    )
    parser.add_argument(
        "--init-lr",
        default=0.01,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--max-steps",
        default=50000,
        type=int,
        help="number of training steps",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device to run the model",
    )
    parser.add_argument(
        "--root",
        default="datasets/data1",
        type=str,
        help="directory containing the dataset",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run(
        args.model_name,
        args.init_lr,
        args.max_steps,
        args.device,
        args.root,
        args.batch_size,
        args.n_frames,
        args.num_workers,
        args.evaluate_frequently,
        args.num_gradient_per_update,
        args.pretrained_path,
        args.learnig_scheduler_gammar,
        args.learnig_scheduler_step,
        args.seed,
        args.cache,
        args.elog,
        args.name,
        args.num_keypoints,
        a_config=args.a_config,
    )
