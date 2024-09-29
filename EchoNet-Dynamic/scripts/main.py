from CustomDataset import CustomSequenceDataset
from CustomCnn import CustomCnn
from learning import *
from preprocessing import *
from pathlib import Path
import torch
import logging
import matplotlib.pyplot as plt


def main():

    epochs, train_batch_size, val_batch_size = 500, 32, 16
    best_loss = 1e6

    logging.basicConfig(
        filename="training.log", 
        format="%(asctime)s %(message)s", 
        filemode="w",
        level=logging.INFO
    )

    # Create logger object
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_list = csv_with_labels()
    split_data(file_list)

    train_data = Path.cwd() / "train_data"
    val_data = Path.cwd() / "validation_data"
    test_data = Path.cwd() / "test_data"
    videos_dir = Path.cwd() / "Videos"

    train_dataset = CustomSequenceDataset("train_files.csv", str(train_data))
    validation_dataset = CustomSequenceDataset("val_files.csv", str(val_data))
    test_dataset = CustomSequenceDataset("test_files.csv", str(test_data))

    # Define model.
    model = CustomCnn()
    model.to(device)
    
    avg_loss_train = []
    avg_loss_val = []
    accuracy_vec = []

    for epoch in range(epochs):

        # Training step
        train_loss = train(model, epoch, train_batch_size, train_dataset, device)

        # Validation step
        val_loss, accuracy = validation(
            model, epoch, val_batch_size, validation_dataset, logger, device
        )

        avg_loss_train.append(train_loss)
        avg_loss_val.append(val_loss)

        # Save the best model.
        if val_loss < best_loss:
            best_loss = model_saving(
                model,
                epoch,
                **{"val_loss": val_loss, "accuracy": accuracy},
            )
            best_loss = val_loss

        print(f"Epoch {epoch} is completed")

    print("Training is completed!")
    
    plt.figure()
    plt.plot(range(1, epochs+1), avg_loss_train, 'b-', label='Training Loss')
    plt.plot(range(1, epochs+1), avg_loss_val, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig('learning_curve.png', dpi=300)

    plt.figure()
    plt.plot(range(1, epochs+1), accuracy_vec)
    plt.title('Accuracy plot of validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy score')
    
    plt.savefig('accuracy_plot.png', dpi=300)
    
if __name__ == "__main__":
    main()
