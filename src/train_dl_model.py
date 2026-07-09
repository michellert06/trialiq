import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sqlalchemy import create_engine
import os

DATABASE_URL = "postgresql://localhost/trialiq"

# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_data():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT title, conditions, status FROM trials", engine)
    valid = ["COMPLETED", "TERMINATED", "WITHDRAWN"]
    df = df[df["status"].isin(valid)].copy()
    df["failed"] = df["status"].isin(["TERMINATED", "WITHDRAWN"]).astype(int)

    # Combine title + conditions into one text input
    df["text"] = (
        df["title"].fillna("") + " [SEP] " + df["conditions"].fillna("")
    )
    return df

# ── 2. Dataset class ──────────────────────────────────────────────────────────
class TrialDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

# ── 3. Train ──────────────────────────────────────────────────────────────────
def train(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)  # use weighted loss instead
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = outputs.logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=["completed", "failed"]))
    print(f"ROC AUC: {roc_auc_score(all_labels, all_probs):.3f}")

# ── 5. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_data()
    print(f"Total trials: {len(df)} | Failure rate: {df['failed'].mean():.1%}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["failed"]
    )

    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    model.to(device)

    # Calculate class weights to handle imbalance
    failure_rate = train_df["failed"].mean()
    weight_for_failed = (1 - failure_rate) / failure_rate  # ~5.8x more weight on failures
    class_weights = torch.tensor([1.0, weight_for_failed], dtype=torch.float).to(device)
    loss_fn = CrossEntropyLoss(weight=class_weights)
    print(f"Class weight for 'failed': {weight_for_failed:.2f}x")

    train_dataset = TrialDataset(train_df["text"].tolist(), train_df["failed"], tokenizer)
    test_dataset = TrialDataset(test_df["text"].tolist(), test_df["failed"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    print("\nTraining for 3 epochs...")
    for epoch in range(3):
        loss = train(model, train_loader, optimizer, device, loss_fn)
        print(f"Epoch {epoch+1}/3 — Loss: {loss:.4f}")

    print("\n--- DistilBERT Evaluation ---")
    evaluate(model, test_loader, device)

    # Save model
    os.makedirs("models/distilbert_trial", exist_ok=True)
    model.save_pretrained("models/distilbert_trial")
    tokenizer.save_pretrained("models/distilbert_trial")
    print("\nModel saved to models/distilbert_trial/")