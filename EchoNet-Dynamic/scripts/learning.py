from CustomCnn import CustomCnn
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch import nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score


def train(model: CustomCnn, epoch: int, batch_size: int, training_set: Dataset, device):
    """
    Train the model.

    :param model: model to be trained.
    :param epoch: training's epoch
    :param batch_size: batch size to use for the forward pass
    :param training_set: dataset used as training data. Includes the
        name of the video's file and its corresponding label.
    """

    model.train()

    trainLoader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

    # Define the loss function and the optimizer used.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    running_loss = 0.0
    total = 0

    for batch_idx, batch_data in enumerate(trainLoader):

        video, labels = batch_data

        video, labels = video.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        predictions = model(video)
        loss = criterion(predictions, labels)
        running_loss += loss.item()
        total += 1

        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1}, Batch: {total}, loss = {loss.item()}")

    avg_loss = running_loss / total

    print(f"Average loss of epoch {epoch+1}, using {total} batches: {avg_loss}")

    return avg_loss


def validation(
    model: CustomCnn,
    epoch: int,
    batch_size: int,
    validation_set: Dataset,
    logger,
    device,
):
    """
    Validation of the model.

    :param model: valuated model.
    :param epoch: training's epoch
    :param batch_size: batch size to use for the forward pass
    :param validation_set: dataset used as validation data.
    :param device: processing device
    :param logger: logger object
    :return: average loss and accruacy of the validation.
    """

    # Create a weighted sampler.
    target = validation_set.file_list["Label"].tolist()
    class_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1.0 / class_count
    samples_weight = np.array([weight[t] for t in target])

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=False
    )

    model.eval()

    valLoader = DataLoader(validation_set, batch_size=batch_size, sampler=sampler)

    criterion = nn.CrossEntropyLoss()
    labels_list, predictions_list = [], []
    running_vloss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for video, label in valLoader:

            video, label = video.to(device), label.to(device)

            # forward
            output = model(video)
            predicted_label = torch.argmax(output, dim=1)

            loss = criterion(output, label)
            running_vloss += loss.item()

            # prediction
            labels_list.extend(label.tolist())
            predictions_list.extend(predicted_label.tolist())
            total += 1
            correct += sum(predicted_label == label).item()

        # accuracy, precision, sensitivity and average validation loss
        accuracy = accuracy_score(labels_list, predictions_list)
        precision = precision_score(labels_list, predictions_list)
        sensitivity = recall_score(labels_list, predictions_list)
        avg_loss = running_vloss / float(total)

        logger.info(
            f"Epoch: {epoch}, accuracy: {accuracy}, precision: {precision}, sensitivity: {sensitivity}, avg_loss: {avg_loss}"
        )

        print(f"Metrics of the network on {total} images")
        print(f"Accuracy: {accuracy*100:.2f}")
        print(f"Precision: {precision*100:.2f}")
        print(f"Sensitivity: {sensitivity*100:.2f}")
        print(
            f"Average validation loss of epoch {epoch+1}, using {total} images: {avg_loss}"
        )

        return avg_loss, accuracy


def test(
    model: CustomCnn,
    log_metrics: int,
    batch_size: int,
    test_set: Dataset,
    logger,
    device,
):
    """
    Test of the model.

    :param model: tested model.
    :param batch_size: batch size to use for the forward pass
    :param test_set: dataset used as test data.
    :param device: processing device
    :param logger: logger object
    """

    model.eval()

    testLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    labels_list, predictions_list = [], []
    total = 0
    correct = 0

    with torch.no_grad():
        for video, label in testLoader:

            video, label = video.to(device), label.to(device)

            # forward
            output = model(video)
            predicted_label = torch.argmax(output, dim=1)

            # prediction
            labels_list.extend(label.tolist())
            predictions_list.extend(predicted_label.tolist())
            total += 1
            correct += sum(predicted_label == label).item()

            print(
                f"Image n.{total}, true label: {label.item()}, predicted label: {predicted_label.item()}"
            )

            logger.info(
                f"Image n.{total}, true label: {label.item()}, predicted label: {predicted_label.item()}"
            )

            if total % log_metrics == 0:

                # accuracy, precision, sensitivity
                accuracy = accuracy_score(labels_list, predictions_list)
                precision = precision_score(labels_list, predictions_list)
                sensitivity = recall_score(labels_list, predictions_list)

                logger.info(
                    f"Data tested: {total}, accuracy: {accuracy}, precision: {precision}, sensitivity: {sensitivity}"
                )

                print(f"Metrics of the network on {total} videos")
                print(f"Accuracy: {accuracy*100:.2f}")
                print(f"Precision: {precision*100:.2f}")
                print(f"Sensitivity: {sensitivity*100:.2f}")

        # Final value of the metrics
        accuracy = accuracy_score(labels_list, predictions_list)
        precision = precision_score(labels_list, predictions_list)
        sensitivity = recall_score(labels_list, predictions_list)
        auc = roc_auc_score(labels_list, predictions_list)

        logger.info(
            f"Test completed: Total: {total}, accuracy: {accuracy}, precision: {precision}, sensitivity: {sensitivity}, AUC: {auc}"
        )

        total_class_0 = sum(1 for x in labels_list if x == 0)
        total_class_1 = sum(1 for x in labels_list if x == 1)

        labels_list = np.array(labels_list)
        predictions_list = np.array(predictions_list)

        accuracy_class_0 = accuracy_score(
            labels_list[np.where(labels_list == 0)],
            predictions_list[np.where(labels_list == 0)],
        )
        accuracy_class_1 = accuracy_score(
            labels_list[np.where(labels_list == 1)],
            predictions_list[np.where(labels_list == 1)],
        )

        logger.info(
            f"Out of {total_class_0} class 0 videos, accuracy scored is: {accuracy_class_0*100}"
        )
        logger.info(
            f"Out of {total_class_1} class 1 videos, accuracy scored is: {accuracy_class_1*100}"
        )


def model_saving(model: CustomCnn, epoch: int, **kwargs):
    """
    Save the model with the best validation score.

    :param model: model to be saved
    :param fold: training's fold
    :param epoch: training's epoch
    """

    model_path = Path.cwd().joinpath("checkpoints", f"model_epoch{epoch}.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            **kwargs,
        },
        model_path,
    )


def model_loading(epoch_to_load: int):
    """
    Load the model with the best validation score.

    :param model: model to be saved
    :param fold: training's fold
    :param epoch: training's epoch
    """

    model_path = Path.cwd().joinpath("checkpoints", f"model_epoch{epoch_to_load}.pt")
    try:
        checkpoint = torch.load(model_path, weights_only=True)
    except Exception:
        print("The given epoch does not exist")

    model = CustomCnn()
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
