import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add this function to calculate and visualize metrics
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get predicted class labels
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return all_preds, all_targets

# Visualization function for metrics
def visualize_metrics(all_preds, all_targets, num_classes=19):
    # Classification report
    report = classification_report(all_targets, all_preds, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(12, 5))

    # Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Precision, Recall, F1-Score
    metrics = ['precision', 'recall', 'f1-score']
    for idx, metric in enumerate(metrics):
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(num_classes), [report[str(i)][metric] for i in range(num_classes)], label=metric)

    plt.title('Precision, Recall, F1-Score per Class')
    plt.xticks(np.arange(num_classes), np.arange(num_classes))
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Main evaluation and visualization
if __name__ == "__main__":
      
    # After training, evaluate the model on validation data
    all_preds, all_targets = evaluate_model(model, val_loader)
    
    # Visualize metrics
    visualize_metrics(all_preds, all_targets)
