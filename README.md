Lending Email Classification Model

This project implements an email classification model specifically designed for categorizing customer emails in the lending industry. The model classifies customer inquiries into different categories such as payment questions, application status, and document requests, based on the content of their emails.

The model can be extended to work with advanced NLP models like BERT for more accurate classifications and deeper insights.
Features

    Classifies customer emails into lending-specific categories:
        Payment questions
        Application status
        Payment difficulties
        Document requests
        Interest rate inquiries
        Early repayment requests
        Loan extension inquiries
        Complaints
        General inquiries
    The model can be trained on a dataset of labeled customer emails.
    Can be integrated with pre-trained models like BERT to further enhance classification accuracy.

Requirements

This project uses Poetry for managing dependencies and virtual environments. To get started, follow these steps:

    Install Poetry if you haven't already:

curl -sSL https://install.python-poetry.org | python3 -

    Install the project dependencies:

poetry install

This will create a virtual environment and install all necessary dependencies from the pyproject.toml file.
Project Structure

.
├── data/                # Folder containing training data (CSV, etc.)
├── model/               # Folder for saving the trained model
├── train.py             # Script for training the model
├── classify.py          # Script for classifying emails
└── README.md            # Project documentation

Categories

The emails will be classified into the following categories:

    Payment questions - Inquiries regarding payment due dates, payment amounts, etc.
    Application status - Questions regarding the status of loan applications.
    Payment difficulties - Inquiries about financial hardship or the inability to make payments.
    Document requests - Requests for documents such as statements or tax forms.
    Interest rate inquiries - Questions about the loan interest rate or APR.
    Early repayment requests - Questions about repaying the loan early or prepayment options.
    Loan extension inquiries - Requests for extending loan terms or restructuring.
    Complaints - Customer complaints or dissatisfaction with loan services.
    General inquiries - General questions not fitting into any of the above categories.

Usage
1. Train the Model

To train the model, run the train_model() function from the train.py script:

poetry run python train.py

This will:

    Load the training dataset.
    Preprocess the data and train the model.
    Save the trained model to the model/ directory.

2. Classify New Emails

Once the model is trained, you can use the classify.py script to classify new email texts. Example usage:

poetry run python classify.py "I need help understanding my next payment due date."

This will classify the email and print the predicted category with confidence scores.
