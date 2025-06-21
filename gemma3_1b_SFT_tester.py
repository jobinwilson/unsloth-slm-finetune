#!/usr/bin/env python
# coding: utf-8

# This script evaluates the finetuned small language model (SLM) <b>gemma3-1b-it</b> for sentiment analysis on the financial news dataset.
# make sure gemma3_1b_SFT_trainer.py is run before running this script


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from datasets import Dataset
from unsloth import FastModel,FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats
from trl import SFTTrainer, SFTConfig
import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


# ### 1. load test dataset
test_df = pd.read_csv("sentiment_test_split.csv")
## view datasample
print('data sample:',test_df.head(2))

## check class distribution
print('class distribution:',test_df.sentiment.value_counts())


# ### Convert to Unsloth chat format

def format_as_conversations(df, test=False):
    prompt_template = (
        "Analyze the sentiment of the text between brackets. "
        "Return one of: positive, neutral, negative.\n\n"
        "[{text}] ="
    )
    prompt_template_new = (
        "Analyze the sentiment of the following news article text. "
        "Return one of: positive, neutral, negative.\n\n"
        "####Text: {text} ="
    )
    conversations = []
    for i in range(len(df)):
        user_msg = prompt_template_new.format(text=df.iloc[i]['text'])
        convo = [{"role": "user", "content": user_msg}]
        if not test:
            convo.append({"role": "model", "content": df.iloc[i]['sentiment']})
        conversations.append(convo)
    return {"conversations": conversations}


test_ds  = Dataset.from_dict(format_as_conversations(test_df, test=True))  # no labels here

###  Load finetuned model and tokenizer from local machine
model, tokenizer = FastModel.from_pretrained(
    model_name = "gemma3-1b-sentiment-ft",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    #token=hf_token,
)


# ### 5. Standardize & Apply chat templates for gemma3

tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
test_ds  = standardize_data_formats(test_ds, aliases_for_assistant=["model"])

def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
        for conv in examples["conversations"]
    ]
    return {"text": texts}

test_ds  = test_ds.map(formatting_prompts_func, batched=True)
print(test_ds[0])


print(f'sample inference prompt:\n{test_ds[0]["text"]}')


# Define prediction and evaluation function

def predict(test_dataset, model, tokenizer, device ="cuda", max_new_tokens = 5,
            temperature = 1.0):
    """
    Predict the sentiment of all texts in the passed dataframe
    using the model and tokenizer. 
    """
    y_preds = [] # List to store predicted sentiment labels
    model.eval() # Set model to evaluation mode
    print(f'total inference samples={len(test_dataset)}')
    # Iterate through each text in the test set
    for i in tqdm(range(len(test_dataset)), desc="Inference Loop"):
        prompt = test_dataset[i]['text']
        # Tokenize the prompt and move tensors to the correct device
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        # Generate output from the model
        torch.manual_seed(42) ## for reproduciability
        with torch.no_grad(): # Disable gradient calculations for inference
             outputs = model.generate(**input_ids,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature,
                                      # Recommended Gemma-3 settings!
                                       top_p = 0.95, top_k = 64,
                                       pad_token_id=tokenizer.eos_token_id) # Avoid warning
        # Decode the generated tokens (excluding the input prompt)
        generated_text = tokenizer.decode(
            outputs[0][input_ids["input_ids"].shape[1]:],  # Slice off the input tokens
            skip_special_tokens=True
        ).strip().lower()
       
        if "positive" in generated_text:
            y_preds.append("positive")
        elif "negative" in generated_text:
            y_preds.append("negative")
        elif "neutral" in generated_text:
            y_preds.append("neutral")
        else:
            y_preds.append("none")
            
    return y_preds

def evaluate_predictions(y_true, y_pred):
    """Evaluate model's performance on sentiment analysis task"""

    # map labels to int values for calculating scikit-learn metrics
    label_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    # map 'none' predictions to neurtral
    y_true_labels = np.array([label_mapping.get(label, 0) for label in y_true])
    y_pred_labels = np.array([label_mapping.get(label, 0) for label in y_pred])

    # Calculate overall accuracy
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    print(f'Overall Accuracy: {accuracy:.3f}')

    # Compute accuracy for each sentiment label
    unique_labels = np.unique(y_true_labels) # Get unique numeric labels

    # Map numeric back to string for printing
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # print accuracy for each class
    for label_num in unique_labels:
        label_mask = y_true_labels == label_num # filter for current class
        label_accuracy = accuracy_score(y_true_labels[label_mask], y_pred_labels[label_mask])
        print(f'Accuracy for class {label_num} ({reverse_label_mapping[label_num]}): {label_accuracy:.3f}')

    # Generate classification report using string labels for clarity
    class_report = classification_report(y_true, y_pred, labels=["negative", "neutral", "positive"], zero_division=0)
    print('\nClassification Report:\n', class_report)

    # Compute and display confusion matrix (using numeric labels)
    # Ensure labels are ordered correctly: negative(-1), neutral(0), positive(1)
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=[-1, 0, 1])
    print('\nConfusion Matrix (Rows: Ground Truth, Cols: Predictions)\
    \n[Negetive, Neutral, Positive]:\n', conf_matrix)


print("Evaluation of fine-tuned model:")
y_true = test_df['sentiment'].tolist()
y_pred = predict(test_ds, model, tokenizer)

evaluate_predictions(y_true,y_pred)

