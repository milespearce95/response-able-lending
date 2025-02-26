import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
import pandas as pd

# Custom dataset class for email classification
class EmailDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Removing unnecessary dimensions
        item_dict = {key: encoding[key].squeeze(0) for key in encoding}
        item_dict['labels'] = torch.tensor(label)
        
        return item_dict

def train_model():
    # Example dataset (you can replace this with your actual dataset)
    data = {
        "text": [
            "I want to know the status of my loan application.",
            "Can you please help me with my payment difficulty? I lost my job.",
            "When is my next payment due?",
            "I need a copy of my tax statement.",
            "I am wondering about the interest rate on my loan.",
            "Can you send me information on early repayment options?"
        ],
        "label": [2, 3, 0, 7, 1, 4]  # Corresponding category labels (for example)
    }

    # Convert data into a pandas DataFrame
    df = pd.DataFrame(data)

    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # Load BERT tokenizer and model
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=9)
    
    # Prepare datasets
    train_dataset = EmailDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = EmailDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    
    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    # Train the model
    trainer.train()
    
    # Save the trained model
    model.save_pretrained('./trained_model')
    tokenizer.save_pretrained('./trained_model')

    print("Training complete. Model saved to './trained_model'.")

if __name__ == "__main__":
    train_model()

