# Teaching SmolLM to do Grammatical Error Correction

The goal of this project is to train a SmolLM-135M model to perform grammatical error correction (GEC) using the Grammarly CoEdIT dataset. This dataset, derived from the CoEdIT project, provides a rich collection of text editing instructions and examples. The task involves several key steps that mimic conventional alignment processes.

## Project Overview

This project involves the following major steps:

1. **Supervised Fine-Tuning (SFT) on Training Data**
    - Fine-tune the [SmolLM-135M model](https://huggingface.co/HuggingFaceTB/SmolLM-135M) using the CoEdIT dataset, which includes input sentences with grammatical errors and their corrected versions.
    - Use the training GEC portion of the CoEdIT dataset to teach the model how to correct grammatical errors effectively.
    - Calculate the BLEU score on the validation set to evaluate the model's performance in generating grammatically correct sentences. Ensure that this evaluation process is reusable for later comparisons.
    - Search for an optimal set of hyperparameters, such as the learning rate.

2. **Create a Preference Optimization Dataset**
    - Generate Output Variants: For each input sentence in the training set, use the fine-tuned model to generate two different output variants. Different decoding strategies will be used, such as varying the temperature or beam size, to produce diverse outputs.
    - Preference Annotation: Measure the edit distance between each generated predicted variant and ground truth correction. The variant with the lower edit distance will be labelled as "chosen" and the one with the higher edit distance as "rejected."

## Setup the Environment

### Install Dependencies

```sh
pip install datasets
pip install trl
pip install fast_edit_distance
pip install evaluate
```

### Download the GEC Data

```python
from datasets import load_dataset

# Download the GEC data
full_train_ds = load_dataset("grammarly/coedit", split="train")
full_test_ds = load_dataset("grammarly/coedit", split="validation")
```

### Filter the Dataset

```python
# Filter examples, keeping only GEC task
def filter_dataset(input_dataset):
    # Filter the dataset for GEC values only
    filtered_dataset = input_dataset.filter(lambda example: example['task'] == 'gec')
    return filtered_dataset

full_train_ds = filter_dataset(full_train_ds)
full_test_ds = filter_dataset(full_test_ds)
```

## Fine-Tuning the Model

### Load the Model and Tokenizer

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "HuggingFaceTB/SmolLM-135M"

# Load the model and the tokenizer from huggingface
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Add a padding token to the tokenizer
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
```

### Train the Model

```python
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

def train_model_wit_sft(train_dataset, eval_dataset, load=True, model_path=None, model=None):
    if load and model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    else:
        sft_config = SFTConfig(
            output_dir="./results",
            learning_rate=1e-2,
            num_train_epochs=3,
            lr_scheduler_type="cosine",
            report_to="none",
            max_seq_length=512,
            packing=False,
            eval_strategy="steps",
        )

        instruction_template = "### Instruction:"
        response_template = " ### Response:"

        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template,
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
        )

        trainer.train()
        if model_path is not None:
            trainer.save_model(model_path)

    return model

model = train_model_wit_sft(
    train_dataset=full_train_ds,
    eval_dataset=full_test_ds,
    load=True,
    model_path=model_path,
    model=model
)
```

## Evaluate the Model

### Evaluate Model Performance

```python
import evaluate
from tqdm import tqdm

def evaluate_model(model, tokenizer, eval_dataset):
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=25,
        shuffle=False,
        collate_fn=batch_collate_eval
    )

    preds = []
    targets = []

    batch_bar = tqdm(total=len(eval_dataloader), dynamic_ncols=True, leave=False, position=0, desc='Evaluation')

    for i, batch in enumerate(eval_dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        tgt = batch['tgt']

        outputs = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id
        )

        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend([process_output(text.strip()) for text in output_texts])
        targets.extend(tgt)

        batch_bar.set_postfix(iteration="{}".format(i))
        batch_bar.update()

    batch_bar.close()

    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=preds, references=targets)

    return results["bleu"]

# Evaluate model, use the function given above
evaluate_model(model, tokenizer, full_test_ds)
```

## Generate Preference Optimization Dataset

### Generate Output Variants

```python
from fast_edit_distance import edit_distance
import pandas as pd

def generate_ds_variants(model, tokenizer, ds, batch_size):
    set_seed(42)  # Set a seed for repeatability

    generated_dataset = {
        "chosen_variants": [],
        "rejected_variants": [],
    }

    eval_dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=batch_collate_eval
    )

    batch_bar = tqdm(total=len(eval_dataloader), dynamic_ncols=True, leave=False, position=0, desc='Dataset Generation')

    for i, batch in enumerate(eval_dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ground_truth_batch = batch['tgt']

        with torch.no_grad():
            var1_outputs = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=8,
                early_stopping=True
            )
            var2_outputs = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.9,
                top_k=40,
                top_p=0.9
            )

        var1_texts = tokenizer.batch_decode(var1_outputs, skip_special_tokens=True)
        var2_texts = tokenizer.batch_decode(var2_outputs, skip_special_tokens=True)

        for var1, var2, ground_truth in zip(var1_texts, var2_texts, ground_truth_batch):
            var1_edit_distance = edit_distance(var1, ground_truth)
            var2_edit_distance = edit_distance(var2, ground_truth)

            if var1_edit_distance < var2_edit_distance:
                generated_dataset["chosen_variants"].append(var1)
                generated_dataset["rejected_variants"].append(var2)
            else:
                generated_dataset["chosen_variants"].append(var2)
                generated_dataset["rejected_variants"].append(var1)

        batch_bar.update()

    batch_bar.close()

    return generated_dataset

generated_dataset = generate_ds_variants(
  model=model,
  tokenizer=tokenizer,
  ds=full_train_ds,
  batch_size=25
)

# Create a dataframe from the dataset
generated_dataset_df = pd.DataFrame(generated_dataset)
# Save the dataset
generated_dataset_df.to_csv("/kaggle/working/generated_DPO_dataset.csv", index=False)
```

## Inference

### Run Inference on a Single Example

```python
# Example of how to run inference on a single example
text = "Fix grammatically: I likes turtles"

def format_text(text: str) -> str:
    return text

def make_inference(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return output_text

print(make_inference(model, tokenizer, text))
```

Expected output: `I like turtles.`

## Conclusion

This project demonstrates how to fine-tune a language model for grammatical error correction using the Grammarly CoEdIT dataset. The model is evaluated using the BLEU score, and a preference optimization dataset is generated to further improve the model's performance.