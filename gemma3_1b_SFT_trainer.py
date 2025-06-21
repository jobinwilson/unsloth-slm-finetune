#!/usr/bin/env python
# coding: utf-8

# This script pre-trains a small language model (SLM) <b>gemma3-1b-it</b> for sentiment analysis. The basic idea is to frame the
# "pre-training" as a supervised fine-tuning (SFT) problem, where examples of prompt-completion pairs are used to fine-tune the 
# SLM using PEFT technique. The user provides an instruction along with the text on which the model is to perform sentiment analysis, 
# and the model returns its response. We compare the model’s performance on this task before and after fine-tuning to observe the gains.
# 
# <b>Note:</b> We are deliberately formulating this as a generative problem, as we will subsequently be fine-tuning the SLM for multiple NLP tasks. 
# If the model were to be used only for a single classification task, a better approach would have been to attach a classification head to 
# the SLM and then fine-tune it. We use <a href="https://unsloth.ai/">Unsloth</a>, as it abstracts away a lot of complexities and enables 
#fine-tuning on limited hardware (e.g., this script runs fine on an RTX 3060 with just 12GB of VRAM).
# 
# To run this script locally, it is recommended to create a separate conda enviornment for unsloth using the following commands from your terminal
# 
# conda create -n unsloth_env python=3.10 -y
# conda activate unsloth_env
# pip install python-dotenv unsloth scikit-learn matplotlib pandas datasets trl
#
# Optional Step: we upgrade unsloth to latest version without updating the dependencies
# pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git 
# 


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
from dotenv import load_dotenv
import os
warnings.filterwarnings("ignore")



# ### 1. load dataset and analyze
## dataset source: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
full_data = pd.read_csv("all-data.csv",names=["sentiment", "text"],encoding="utf-8", encoding_errors="replace")
## view datasample
print('data sample:',full_data.head(2))

## check class distribution
print('class distribution:',full_data.sentiment.value_counts())


# ### 2. Balance the dataset and create splits

min_count = full_data.sentiment.value_counts().min()
# round min_count to a multiple of 100
min_count = min_count - min_count%100
print("Balancing to min class count:", min_count)
df_splits = []
for sentiment in full_data.sentiment.unique():
    df_splits.append(full_data[full_data.sentiment == sentiment].sample(n = min_count, random_state=42))
balanced_df = pd.concat(df_splits,axis=0)

# === Shuffle and split === #
train_df, test_df = train_test_split(balanced_df, test_size=300, stratify=balanced_df['sentiment'], random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=150, stratify=train_df['sentiment'], random_state=42)

print(f'Creating splits:train={len(train_df)},vaidation={len(val_df)},test={len(test_df)}')
# You can print value counts to confirm balance
print("Train:", train_df['sentiment'].value_counts().to_dict())
print("Val:", val_df['sentiment'].value_counts().to_dict())
print("Test:", test_df['sentiment'].value_counts().to_dict())

## save the splits to run experiments from other notebooks/scripts
train_df.to_csv("sentiment_train_split.csv",index=False)
val_df.to_csv("sentiment_validation_split.csv",index=False)
test_df.to_csv("sentiment_test_split.csv",index=False)

# ### 3. Convert to Unsloth chat format

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

train_ds = Dataset.from_dict(format_as_conversations(train_df))
val_ds   = Dataset.from_dict(format_as_conversations(val_df))
test_ds  = Dataset.from_dict(format_as_conversations(test_df, test=True))  # no labels here

## load huggingface token from .env file
load_dotenv()
hf_token=os.getenv("HF_TOKEN")
## add your huggingface token as some models may be gated


# ##### optional: force download of a model snapshot to the local cache
# 
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="unsloth/gemma-3-1b-it",token=hf_token)
#     

### 4. Load model and tokenizer
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-1b-it",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    token=hf_token,
)


# ### 5. Standardize & Apply chat templates for gemma3

tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")


train_ds = standardize_data_formats(train_ds, aliases_for_assistant=["model"])
val_ds   = standardize_data_formats(val_ds, aliases_for_assistant=["model"])
test_ds  = standardize_data_formats(test_ds, aliases_for_assistant=["model"])

def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
        for conv in examples["conversations"]
    ]
    return {"text": texts}

train_ds = train_ds.map(formatting_prompts_func, batched=True)
val_ds   = val_ds.map(formatting_prompts_func, batched=True)
test_ds  = test_ds.map(formatting_prompts_func, batched=True)
print(train_ds[0])


print(f'sample inference prompt:\n{test_ds[0]["text"]}')


# ### 6. Define prediction and evaluation function

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


print("Model Evaluation BEFORE fine-tuning:")
y_true = test_df['sentiment'].tolist()
y_pred = predict(test_ds, model, tokenizer)

evaluate_predictions(y_true,y_pred)


# ### 7. Prepare model for PEFT fine-tuning

### prepare model adapter using unsloth's FastModel wrapper
model = FastModel.get_peft_model(
    model,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8, ## LORA rank, keeping small value as task complexity is relatively low
    lora_alpha=16, ## keep r*2
    lora_dropout=0.05, ## for regularization
    random_state = 3407,
)

## prepare trainer uisng HF TRL
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        eval_steps=25,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=25,
        save_total_limit = 3, ## only keep best 3 models
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        report_to="none"
    ),
)


## check the tokens being passed by trainer
#tokenizer.decode(trainer.train_dataset[0]['input_ids'])


trainer.train()


print("Model Evaluation AFTER fine-tuning:")
y_pred_ft = predict(test_ds, model, tokenizer)
evaluate_predictions(y_true,y_pred_ft)


### 8. Analyze learning curves / training and validation losses 

evaluation_freq = 25 ## no. of steps in which evals happen
train_logs = trainer.state.log_history
train_losses = [log for log in train_logs if "loss" in log]
eval_losses = [log for log in train_logs if  "eval_loss" in log]
## extract losses as dataframes by filtering for steps for which both "loss" and "eval_loss" exist
train_loss_df = pd.DataFrame(train_losses)
train_loss_df = train_loss_df[train_loss_df.step%evaluation_freq==0]
eval_losses_df = pd.DataFrame(eval_losses)
eval_losses_df = eval_losses_df[eval_losses_df.step%evaluation_freq==0]


## plot the losses
plt.figure(figsize=(10, 4))
plt.plot(train_loss_df['step'], train_loss_df['loss'], label='Training')
plt.plot(eval_losses_df['step'], eval_losses_df['eval_loss'], label='Validation')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)


## save final model as LoRA adapters
model.save_pretrained("gemma3-1b-sentiment-ft-adapter")  # Local saving
tokenizer.save_pretrained("gemma3-1b-sentiment-ft-adapter")


## SAVE FINETUNED MODEL LOCALLY (After merging PEFT weights) ##
model.save_pretrained_merged("gemma3-1b-sentiment-ft", tokenizer,token=hf_token)
