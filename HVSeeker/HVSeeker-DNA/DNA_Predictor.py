import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def predict(input_data, labels, batch_size=32):
    # Load the model
    model = torch.load('model_best_acc2_test_model.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Move your model to the GPU
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    labels = np.array(labels, dtype=np.float32)
    input_data = torch.tensor(input_data).float().to(device)
    labels = torch.tensor(labels).long().to(device)  # Changed to long for classification

    data_loader = DataLoader(TensorDataset(input_data, labels), batch_size=batch_size)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch, outputs in data_loader:
            batch = batch.to(device)
            outputs = outputs.to(device)
            predictions = model(batch)

            # If your model outputs raw logits, apply softmax and then argmax
            # Otherwise, if it already outputs probabilities, just use argmax
            predictions = torch.argmax(predictions, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(outputs.cpu().numpy())

        # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Set the context to be more readable
    sns.set_context('talk', font_scale=0.8)  # Adjust the font_scale to make it more readable

    # Plot the confusion matrix
    # Set an appropriate figure size based on your specific matrix size
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as necessary

    # Plot the heatmap
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax, cbar=False)

    # Labels, title and ticks
    label_font = {'size':'14'}  # Adjust the size as needed
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('True labels', fontdict=label_font)
    ax.set_title('Confusion Matrix', fontdict=label_font)
    ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust the label size as needed

    # Make sure the text is well-fitted
    plt.tight_layout()

    # Save the confusion matrix to a file
    plt.savefig('confusion_matrix.png', dpi=300)  # Save with high resolution
    plt.close()  # Close the plot to avoid displaying it in notebooks or environments

    # Calculate overall accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    # Save predictions and labels to a CSV file
    df = pd.DataFrame({
        'Predictions': all_predictions,
        'Labels': all_labels
    })
    df['Accuracy'] = accuracy  # Add the accuracy to the DataFrame
    df.to_csv('predictions_and_accuracy.csv', index=False)

    return accuracy


