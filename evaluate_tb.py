import argparse, os, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import pandas as pd

# -----------------------------
# 1) Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2) Model loader
# -----------------------------
def load_model(weights_path):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)  # 0=Normal, 1=TB

    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# -----------------------------
# 3) Image transform
# -----------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# -----------------------------
# 4) Infer label from folder name
# -----------------------------
def label_from_path(path):
    parent = os.path.basename(os.path.dirname(path)).lower()
    if parent.startswith("normal"):
        return 0
    if "tb" in parent or "tuber" in parent:
        return 1
    return None  # unknown

# -----------------------------
# 5) Evaluate
# -----------------------------
def evaluate(data_dir, weights, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(weights)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    if not image_paths:
        print(f"No images found under: {data_dir}")
        return

    X_scores = []      # probability of TB
    y_true = []        # 0 or 1
    y_pred = []        # 0 or 1

    print(f"Found {len(image_paths)} images. Evaluating...")

    for p in tqdm(image_paths, desc="Evaluating images"):
        y = label_from_path(p)
        if y is None:
            continue

        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            score_tb = float(probs[1])  # probability of TB

        y_true.append(y)
        y_pred.append(pred)
        X_scores.append(score_tb)

    if len(y_true) == 0:
        print("No labeled images recognized (ensure subfolders are named 'Normal' and 'TB').")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    X_scores = np.array(X_scores)

    # -----------------------------
    # Console output: accuracy & classification report
    # -----------------------------
    overall_acc = (y_true == y_pred).mean()
    class_names = ["Normal", "TB"]

    print(f"\nOverall Accuracy: {overall_acc*100:.2f}%\n")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=2)
    print(report)

    cm_df = pd.DataFrame(confusion_matrix(y_true, y_pred), index=class_names, columns=class_names)
    print("Confusion Matrix:")
    print(cm_df)

    # -----------------------------
    # Confusion matrix & accuracy for plots
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # -----------------------------
    # Per-class accuracy plot
    # -----------------------------
    plt.figure(figsize=(6,4))
    plt.bar(class_names, per_class_acc * 100)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-class Accuracy")
    acc_bar_path = os.path.join(out_dir, "per_class_accuracy.png")
    plt.tight_layout()
    plt.savefig(acc_bar_path, dpi=150)
    plt.close()

    # -----------------------------
    # Confusion matrix plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)

    # -----------------------------
    # ROC curve
    # -----------------------------
    roc_path = None
    try:
        fpr, tpr, _ = roc_curve(y_true, X_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (TB vs Normal)")
        plt.legend(loc="lower right")
        roc_path = os.path.join(out_dir, "roc_curve.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150)
        plt.close()
    except Exception:
        pass

    # -----------------------------
    # Save PDF report (inside function)
    # -----------------------------
    pdf_path = os.path.join(out_dir, "accuracy_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # 1) Plots
        for path in [acc_bar_path, cm_path, roc_path]:
            if path is None:
                continue
            img = plt.imread(path)
            plt.figure(figsize=(6,4))
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig()
            plt.close()

        # 2) Text summary page
        plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        summary_lines = [
            f"Overall Accuracy: {overall_acc*100:.2f}%",
            "",
            "Classification Report:",
            report,
            "",
            "Confusion Matrix (rows=true, cols=pred):",
            str(cm)
        ]
        plt.text(0.05, 0.95, "\n".join(summary_lines), va="top", family="monospace")
        pdf.savefig()
        plt.close()

    print("\nSaved PDF at:", pdf_path)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TB model & plot accuracy.")
    parser.add_argument("--data_dir", required=True, help="Path to test set root (Normal/ and TB/ subfolders)")
    parser.add_argument("--weights", default="best_densenet_tb.pth", help="Path to model weights")
    parser.add_argument("--out_dir", default="metrics_out", help="Where to save plots/PDF")
    args = parser.parse_args()

    evaluate(args.data_dir, args.weights, args.out_dir)
