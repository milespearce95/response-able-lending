import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset as TorchDataset
import pandas as pd

# Custom dataset class for email classification
class EmailDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Removing unnecessary dimensions
        item_dict = {key: encoding[key].squeeze(0) for key in encoding}
        return item_dict


class EmailResponder:
    def __init__(self, model_path: str, categories: list):
        # Load the trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.categories = categories

        # Ensure the model is in evaluation mode
        self.model.eval()

    def classify_email(self, email_text: str) -> str:
        """
        Classify the email into one of the predefined categories.

        Args:
            email_text (str): The email text to classify.

        Returns:
            str: The predicted category (one of the predefined categories).
        """
        # Tokenizing the input
        inputs = self.tokenizer(
            email_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # Making prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # Get the category name based on the predicted class index
        predicted_category = self.categories[predicted_class]
        return predicted_category

    def generate_response(self, predicted_category: str) -> str:
        """
        Generate a response based on the predicted category.

        Args:
            predicted_category (str): The predicted category of the email.

        Returns:
            str: The generated response text.
        """
        # Response templates for each category (customize these as per your requirement)
        responses = {
            "payment_question": "Thank you for your email. The next payment due date is [Insert Date]. Please let us know if you need any further assistance.",
            "interest_rate_question": "Thank you for reaching out. The current interest rate for your loan is [Insert Interest Rate]. Feel free to ask if you need more details.",
            "application_status": "Thank you for your inquiry. Your loan application is currently under review. We will notify you as soon as there is an update.",
            "payment_difficulty": "We are sorry to hear about your difficulties. Please let us know if you'd like to discuss a hardship program or payment extension options.",
            "early_repayment": "Thank you for your inquiry. If you're interested in early repayment, we would be happy to provide more information on how that process works.",
            "loan_extension": "We understand that sometimes circumstances change. We can discuss extending your loan. Please contact us for further details.",
            "complaint": "We're sorry to hear that you're having issues. We take your feedback seriously and will make sure to address your concerns as soon as possible.",
            "document_request": "Thank you for your request. We can provide you with the necessary documents. Please specify which document you need.",
            "general_inquiry": "Thank you for reaching out. Please feel free to let us know if you have any other questions or need further assistance."
        }

        # Get the appropriate response template based on the predicted category
        return responses.get(predicted_category, "Thank you for your email. How can we assist you further?")

    def respond_to_email(self, email_text: str) -> str:
        """
        Classify the email and generate an appropriate response.

        Args:
            email_text (str): The email text to classify and generate a response for.

        Returns:
            str: The generated response text.
        """
        predicted_category = self.classify_email(email_text)
        response = self.generate_response(predicted_category)
        return response


# Example Usage
if __name__ == "__main__":
    # Define the categories (make sure these match the categories used during training)
    categories = [
        "payment_question",
        "interest_rate_question",
        "application_status",
        "payment_difficulty",
        "early_repayment",
        "loan_extension",
        "complaint",
        "document_request",
        "general_inquiry"
    ]

    # Initialize the EmailResponder with the model path and categories
    model_path = './trained_model'  # Replace this with the path to your trained model
    email_responder = EmailResponder(model_path=model_path, categories=categories)

    # Example email text
    email_text = "Can you tell me when my next payment is due?"

    # Get the response to the email
    response = email_responder.respond_to_email(email_text)
    print("Response:", response)
