#!/usr/bin/env python
# coding: utf-8

# # <font color = 'indianred'>**Emotion Detection - MultiLabel** </font>
# 
# **Plan**
# 
# 1. Set Environment
# 2. Load Dataset
# 3. Load Pre-trained Tokenizer
# 4. Train Model
#      1. Compute Metric Function <br>
#      2. Training Arguments <br>
#      3. Specify Model
#      4. Instantiate Trainer <br>
#      5. Setup WandB <br>
#      6. Training and Validation
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# # <font color = 'indianred'> **1. Setting up the Environment** </font>
# 
# 

# In[ ]:


# If in Colab, then import the drive module from google.colab
if 'google.colab' in str(get_ipython()):
  from google.colab import drive
  # Mount the Google Drive to access files stored there
  drive.mount('/content/drive')

  # Install the latest version of torchtext library quietly without showing output
  # !pip install torchtext -qq
  get_ipython().system('pip install transformers evaluate wandb datasets accelerate peft bitsandbytes -U -qq ## NEW LINES ##')
  basepath = '/content/drive/MyDrive/hw6'
else:
  basepath = '/home/harpreet/Insync/google_drive_shaannoor/data'


# <font color = 'indianred'> *Load Libraries* </font>

# In[ ]:


# standard data science librraies for data handling and v isualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import BitsAndBytesConfig

import wandb
import evaluate


# In[ ]:


import peft
from peft import prepare_model_for_kbit_training
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
)


# # <font color = 'indianred'> **2. Load_Dataset** </font>

# In[ ]:


df = pd.read_csv("/content/drive/MyDrive/Natural Language Processing Datasets/train.csv")
test = pd.read_csv("/content/drive/MyDrive/Natural Language Processing Datasets/test.csv")


# In[ ]:





# In[ ]:


Text = df['Tweet'].apply(lambda x: x.lower())
Labels = df.drop(['ID','Tweet'],axis=1)
X =Text.values
test = test['Tweet'].apply(lambda x: x.lower())
testset = test.values


# In[ ]:


y = Labels.iloc[:,:].values.astype(float)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


from datasets import Dataset,DatasetDict
trainset = Dataset.from_dict({
    'text': X_train,
    'label': y_train
})

validset = Dataset.from_dict({
    'text': X_val,
    'label': y_val
})
testset = Dataset.from_dict({
    'text':testset

})
train_val = DatasetDict(
    {"train": trainset, "valid": validset})


# In[ ]:


emotion_data = train_val


# In[ ]:


emotion_data


# In[ ]:


## task 1 part a


# In[ ]:


checkpoint = "google/gemma-1.1-2b-it"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, token = '')


# In[ ]:


def tokenize_fn(batch):
    return tokenizer(text = batch["text"], truncation=True)


# <font color = 'indianred'> *Use map function to apply tokenization to all splits*

# In[ ]:


tokenized_dataset= emotion_data.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(
    ['text']
)
# tokenized_dataset.set_format(type='torch')


# In[ ]:


tokenized_dataset


# #  <font color = 'indianred'> **4. Model Training**

# ##  <font color = 'indianred'> **4.1. compute_metrics function** </font>
# 
# 

# In[ ]:


accuracy_metric = evaluate.load('accuracy', 'multilabel')
f1 = evaluate.load('f1','multilabel')

def compute_metrics(eval_pred):
    pred, labels = eval_pred
    # print(logits.shape)
    preds = (pred > 0).astype(int)
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1_micro = f1.compute(predictions=preds, references=labels, average='micro')
    f1_macro = f1.compute(predictions=preds, references=labels, average='macro')
    return {'f1_micro':f1_micro['f1'],
            'f1_macro':f1_macro['f1'],
            'accuracy':accuracy['accuracy'],
            }


# ## <font color = 'indianred'> **4.2. Training Arguments**</font>
# 
# 
# 
# 
# 
# 

# In[ ]:


# Define the directory where model checkpoints will be saved
run_name = "google/gemma"
base_folder = Path(basepath)
model_folder = base_folder / "models"/run_name
# Create the directory if it doesn't exist
model_folder.mkdir(exist_ok=True, parents=True)

# Configure training parameters
training_args = TrainingArguments(
    # Training-specific configurations
    num_train_epochs=10,  # Total number of training epochs
    # Number of samples per training batch for each device
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    # gradient_accumulation_steps=8,

    weight_decay=0.01,  # Apply L2 regularization to prevent overfitting
    learning_rate=1e-4,  # Step size for the optimizer during training
    lr_scheduler_type='linear',
    warmup_steps=0,  # Number of warmup steps for the learning rate scheduler
    optim='adamw_torch',  # Optimizer,
    max_grad_norm = 1.0,

    # Checkpoint saving and model evaluation settings
    output_dir=str(model_folder),  # Directory to save model checkpoints
    evaluation_strategy='steps',  # Evaluate model at specified step intervals
    eval_steps=100,  # Perform evaluation every 10 training steps
    save_strategy="steps",  # Save model checkpoint at specified step intervals
    save_steps=100,  # Save a model checkpoint every 10 training steps
    load_best_model_at_end=True,  # Reload the best model at the end of training
    save_total_limit=2,  # Retain only the best and the most recent model checkpoints
    # Use 'accuracy' as the metric to determine the best model
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,  # A model is 'better' if its accuracy is higher


    # Experiment logging configurations (commented out in this example)
    logging_strategy='steps',
    logging_steps=100,
    report_to='wandb',  # Log metrics and results to Weights & Biases platform
    run_name=run_name,  # Experiment name for Weights & Biases


)


# ## <font color = 'indianred'> **4.3. Specify Model**</font>

# In[ ]:


model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=11,
                                                           problem_type="multi_label_classification", token = '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' )

config = AutoConfig.from_pretrained(checkpoint)
model.config = config


# In[ ]:


model


# ## <font color = 'indianred'> **4.4. LORA Setup**</font>

# In[ ]:


import re
model_modules = str(model.modules)
pattern = r'\((\w+)\): Linear'
linear_layer_names = re.findall(pattern, model_modules)

names = []
# Print the names of the Linear layers
for name in linear_layer_names:
    names.append(name)
target_modules = list(set(names))
target_modules


# In[ ]:


gemma_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=64,
    lora_alpha=128,
    lora_dropout=0.01,
    bias="lora_only",
    modules_to_save = ['score', 'norm'],
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",'down_proj', 'gate_proj', 'up_proj'])
gemma_peft_model = get_peft_model(model, gemma_peft_config)
gemma_peft_model.print_trainable_parameters()


# In[ ]:





# In[ ]:





# In[ ]:





# ##  <font color = 'indianred'> **4.4 Custom Trainer**</font>
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:


pos_weights= torch.tensor([2., 3., 2., 2., 2., 3., 2., 3., 2., 4., 4.])


# In[ ]:


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()  # Ensure labels are float for BCE loss
        outputs = model(**inputs)
        logits = outputs.get("logits")

        device = next(model.parameters()).device

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# In[ ]:


trainer = CustomTrainer(
    model=gemma_peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,

)


# ## <font color = 'indianred'> **4.5 Setup WandB**</font>

# In[ ]:


wandb.login()
get_ipython().run_line_magic('env', 'WANDB_PROJECT = emotions_kaggle_S2024')


# ## <font color = 'indianred'> **4.6. Start Training**</font>

# In[ ]:


trainer.train()  # start training


# ## <font color = 'indianred'> **4.7. Validation**</font>
# 

# In[ ]:


eval_results = trainer.evaluate(emotion_dataset['valid'])
eval_results


# In[ ]:


wandb.log({"eval_accuracy": eval_results["eval_accuracy"], "eval_loss": eval_results["eval_loss"],
"eval_f1_micro": eval_results["eval_f1_micro"], "eval_f1_macro": eval_results["eval_f1_macro"]})


# In[ ]:


trainer.save_model('/content/drive/MyDrive/hw6/saved_models/google-gemma')


# In[ ]:


## Prediction on test  set for kaggle


# In[ ]:


from datasets import Dataset,DatasetDict
test = pd.read_csv("/content/drive/MyDrive/Natural Language Processing Datasets/test.csv")
testset = Dataset.from_dict({'text':test['Tweet']})
tokenized_testset= testset.map(tokenize_fn, batched=True)


# In[ ]:



tokenized_testset = tokenized_testset.remove_columns(
    ['text']
)
pred =trainer.predict(tokenized_testset)
pred = (pred[0] >0).astype(int)
prediction = pd.DataFrame(pred)
path = "/content/drive/MyDrive/prediction1.xlsx"
prediction.to_excel(path, index=False, header=False)


# In[ ]:





# ###  <font color = 'indianred'> **Check Confusion Matrix**</font>
# 
# 
# 

# In[ ]:


# Use the trainer to generate predictions on the tokenized validation dataset.
# The resulting object, valid_output, will contain the model's logits (raw prediction scores) for each input in the validation set.
valid_output = trainer.predict(tokenized_dataset["valid"])


# In[ ]:


predictions_valid = (valid_output.predictions[0] > 0).astype(int)
labels_valid = valid_output.label_ids.astype(int)


# In[ ]:


y_true = labels_valid
y_pred = predictions_valid
class_names = labels

mcm = multilabel_confusion_matrix(y_true, y_pred,)

# 1. Individual Heatmaps
for idx, matrix in enumerate(mcm):
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['True Negative', 'True Positive'])
    plt.title(f'Confusion Matrix for {class_names[idx]}')
    plt.show()

# 2. Aggregate Metrics Heatmap
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)
f1_per_class = f1_score(y_true, y_pred, average=None)

metrics_df = pd.DataFrame({
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-Score': f1_per_class
}, index=class_names)

plt.figure(figsize=(10, 8))
# sns.heatmap(metrics_df, annot=True, cmap='Blues')
# plt.title('Metrics for each class')
# plt.show()

ax = sns.heatmap(metrics_df, annot=True, cmap='Blues')
plt.title('Metrics for each class')
plt.tight_layout()  # Adjust layout to not cut off edges

# Log the heatmap to wandb
wandb.log({"Metrics Heatmap": wandb.Image(ax.get_figure())})
plt.show()

# 3. Histogram of Metrics
metrics_df.plot(kind='bar', figsize=(12, 7))
plt.ylabel('Score')
plt.title('Precision, Recall, and F1-Score for Each Class')
plt.show()



# In[ ]:


wandb.finish()


# In[ ]:


from huggingface_hub import notebook_login
notebook_login()


# In[ ]:


flant5_peft_model.push_to_hub("harpreetmann/flant5_peft_model_emotion_detection")


# # Test Set Predictions

# In[ ]:


import torch

from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

peft_model_id = "harpreetmann/flant5_peft_model_emotion_detection"
config = PeftConfig.from_pretrained(peft_model_id)
config.base_model_name_or_path
base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,
                                                                num_labels=11,
                                                                problem_type="multi_label_classification")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)


# In[ ]:


base_model


# In[ ]:


# Load the Lora model
inference_model = PeftModel.from_pretrained(base_model, peft_model_id)


# In[ ]:


inference_model


# In[ ]:


## peft using IA3


# In[ ]:





# In[ ]:


checkpoint = "google/gemma-1.1-2b-it"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, token = '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')


# In[ ]:


def tokenize_fn(batch):
    return tokenizer(text = batch["text"], truncation=True)


# In[ ]:


tokenized_dataset= emotion_data.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(
    ['text']
)
# tokenized_dataset.set_format(type='torch')


# In[ ]:


accuracy_metric = evaluate.load('accuracy', 'multilabel')
f1 = evaluate.load('f1','multilabel')

def compute_metrics(eval_pred):
    pred, labels = eval_pred
    # print(logits.shape)
    preds = (pred > 0).astype(int)
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1_micro = f1.compute(predictions=preds, references=labels, average='micro')
    f1_macro = f1.compute(predictions=preds, references=labels, average='macro')
    return {'f1_micro':f1_micro['f1'],
            'f1_macro':f1_macro['f1'],
            'accuracy':accuracy['accuracy'],
            }


# In[ ]:


# Define the directory where model checkpoints will be saved
run_name = "google/gemma"
base_folder = Path(basepath)
model_folder = base_folder / "models"/run_name
# Create the directory if it doesn't exist
model_folder.mkdir(exist_ok=True, parents=True)

# Configure training parameters
training_args = TrainingArguments(
    # Training-specific configurations
    num_train_epochs=10,  # Total number of training epochs
    # Number of samples per training batch for each device
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    # gradient_accumulation_steps=8,

    weight_decay=0.1,  # Apply L2 regularization to prevent overfitting
    learning_rate=1e-4,  # Step size for the optimizer during training
    lr_scheduler_type='linear',
    warmup_steps=0,  # Number of warmup steps for the learning rate scheduler
    optim='adamw_torch',  # Optimizer,
    max_grad_norm = 1.0,

    # Checkpoint saving and model evaluation settings
    output_dir=str(model_folder),  # Directory to save model checkpoints
    evaluation_strategy='steps',  # Evaluate model at specified step intervals
    eval_steps=100,  # Perform evaluation every 10 training steps
    save_strategy="steps",  # Save model checkpoint at specified step intervals
    save_steps=100,  # Save a model checkpoint every 10 training steps
    load_best_model_at_end=True,  # Reload the best model at the end of training
    save_total_limit=2,  # Retain only the best and the most recent model checkpoints
    # Use 'accuracy' as the metric to determine the best model
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,  # A model is 'better' if its accuracy is higher


    # Experiment logging configurations (commented out in this example)
    logging_strategy='steps',
    logging_steps=100,
    report_to='wandb',  # Log metrics and results to Weights & Biases platform
    run_name=run_name,  # Experiment name for Weights & Biases

    # fp16=False,
    # bf16=False,
    # tf32= False
)


# In[ ]:



from transformers import BitsAndBytesConfig


# In[ ]:





# In[ ]:


bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=11,
                                                           problem_type="multi_label_classification",
                                                           token = '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',
                                                           quantization_config = bnb_config  )
model = prepare_model_for_kbit_training(model)
config = AutoConfig.from_pretrained(checkpoint)
model.config = config


# In[ ]:





# In[ ]:


gemma_peft_config =
peft.IA3Config(
    task_type=TaskType.SEQ_CLS,
    peft_type = "IA3",
    target_modules=['v_proj','up_proj','gate_proj','down_proj'],)

gemma_peft_model = get_peft_model(model, gemma_peft_config)
gemma_peft_model.print_trainable_parameters()


# In[ ]:


pos_weights= torch.tensor([2., 3., 2., 2., 2., 3., 2., 3., 2., 4., 4.])


# In[ ]:


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()  # Ensure labels are float for BCE loss
        outputs = model(**inputs)
        logits = outputs.get("logits")

        device = next(model.parameters()).device

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# In[ ]:


trainer = CustomTrainer(
    model=gemma_peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,

)


# In[ ]:


wandb.login()
get_ipython().run_line_magic('env', 'WANDB_PROJECT = emotions_kaggle_S2024')


# In[ ]:


trainer.train()  # start training


# In[ ]:


eval_results = trainer.evaluate(emotion_data['valid'])


# In[ ]:


eval_results


# In[ ]:


## fine tuning fte-large using q-lora


# In[ ]:


checkpoint = 'BAAI/bge-large-zh-v1.5'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, token = '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')


# In[ ]:


def tokenize_fn(batch):
    return tokenizer(text = batch["text"], truncation=True)


# In[ ]:


tokenized_dataset= emotion_data.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(
    ['text']
)
# tokenized_dataset.set_format(type='torch')


# In[ ]:


accuracy_metric = evaluate.load('accuracy', 'multilabel')
f1 = evaluate.load('f1','multilabel')

def compute_metrics(eval_pred):
    pred, labels = eval_pred
    # print(logits.shape)
    preds = (pred > 0).astype(int)
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1_micro = f1.compute(predictions=preds, references=labels, average='micro')
    f1_macro = f1.compute(predictions=preds, references=labels, average='macro')
    return {'f1_micro':f1_micro['f1'],
            'f1_macro':f1_macro['f1'],
            'accuracy':accuracy['accuracy'],
            }


# In[ ]:


# Define the directory where model checkpoints will be saved
run_name = "google/gemma"
base_folder = Path(basepath)
model_folder = base_folder / "models"/run_name
# Create the directory if it doesn't exist
model_folder.mkdir(exist_ok=True, parents=True)

# Configure training parameters
training_args = TrainingArguments(
    # Training-specific configurations
    num_train_epochs=10,  # Total number of training epochs
    # Number of samples per training batch for each device
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    # gradient_accumulation_steps=8,

    weight_decay=0.1,  # Apply L2 regularization to prevent overfitting
    learning_rate=1e-4,  # Step size for the optimizer during training
    lr_scheduler_type='linear',
    warmup_steps=0,  # Number of warmup steps for the learning rate scheduler
    optim='adamw_torch',  # Optimizer,
    max_grad_norm = 1.0,

    # Checkpoint saving and model evaluation settings
    output_dir=str(model_folder),  # Directory to save model checkpoints
    evaluation_strategy='steps',  # Evaluate model at specified step intervals
    eval_steps=100,  # Perform evaluation every 10 training steps
    save_strategy="steps",  # Save model checkpoint at specified step intervals
    save_steps=100,  # Save a model checkpoint every 10 training steps
    load_best_model_at_end=True,  # Reload the best model at the end of training
    save_total_limit=2,  # Retain only the best and the most recent model checkpoints
    # Use 'accuracy' as the metric to determine the best model
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,  # A model is 'better' if its accuracy is higher


    # Experiment logging configurations (commented out in this example)
    logging_strategy='steps',
    logging_steps=100,
    report_to='wandb',  # Log metrics and results to Weights & Biases platform
    run_name=run_name,  # Experiment name for Weights & Biases

    # fp16=False,
    # bf16=False,
    # tf32= False
)


# In[ ]:


bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
  llm_int8_skip_modules = ['classifier'],
  bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=11,
                                                           problem_type="multi_label_classification",
                                                           token = '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',
                                                           quantization_config = bnb_config  )
model = prepare_model_for_kbit_training(model)
config = AutoConfig.from_pretrained(checkpoint)
model.config = config


# In[ ]:


model


# In[ ]:





# In[ ]:


model.config.pad_token_id = tokenizer.pad_token_id
model.config


# In[ ]:


from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[ ]:


import re
model_modules = str(model.modules)
pattern = r'\((\w+)\): Linear'
linear_layer_names = re.findall(pattern, model_modules)

names = []
for name in linear_layer_names:
    names.append(name)
target_modules = list(set(names))
target_modules


# In[ ]:


gemma_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=128,
    lora_alpha=256,
    lora_dropout=0.01,
    bias="lora_only",
    modules_to_save = ['dense', 'classifier'],
    target_modules = ['value', 'query', 'key'])

gemma_peft_model = get_peft_model(model, gemma_peft_config)
gemma_peft_model.print_trainable_parameters()


# In[ ]:


pos_weights= torch.tensor([2., 3., 2., 2., 2., 3., 2., 3., 2., 4., 4.])


# In[ ]:


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()  # Ensure labels are float for BCE loss
        outputs = model(**inputs)
        logits = outputs.get("logits")

        device = next(model.parameters()).device

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# In[ ]:


trainer = CustomTrainer(
    model=gemma_peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,

)


# In[ ]:


wandb.login()
get_ipython().run_line_magic('env', 'WANDB_PROJECT = emotions_kaggle_S2024')


# In[ ]:


trainer.train()  # start training


# In[ ]:


eval_results = trainer.evaluate(emotion_data['valid'])


# In[ ]:


eval_results

