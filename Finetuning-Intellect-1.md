
# Beta/Unstable Tutorial: Fine-Tune ChatCat on TPU V4-8

> **Note:** This script has been confirmed to run, but it is not yet optimized. Feel free to update it or help us improve it! (its all because i do not know Pytorch at all)

In this tutorial, you'll learn how to set up a TPU V4-8, install the necessary dependencies, and run a fine-tuning script for ChatCat using PyTorch, Torch-XLA, and Hugging Face Transformers. The model and dataset used here are for demonstration purposes, and you can update them as needed.

---

## 1. Create Your TPU V4-8

Use the following `gcloud` command to create a TPU V4-8 instance:

```bash
gcloud alpha compute tpus queued-resources create TPUtemp \
  --node-id TPUtemp \
  --zone us-central2-b \
  --accelerator-type v4-8 \
  --runtime-version tpu-ubuntu2204-base
```

> **Tip:** Before proceeding, check the TPU's status. If its status is not **active**, wait until it becomes active.

---

## 2. SSH Into Your TPU

Once the TPU status is active, SSH into the TPU:

```bash
gcloud compute tpus tpu-vm ssh TPUtemp --zone us-central2-b
```

---

## 3. Install Required Packages

### a. Install PyTorch and Torch-XLA

Inside the TPU SSH session, install PyTorch, Torch-XLA (with TPU support), and torchvision by running:

```bash
pip install torch torch_xla[tpu] torchvision \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### b. Set Environment Variable

Set the following environment variable so that PJRT recognizes the TPU:

```bash
export PJRT_DEVICE=TPU
```

### c. Install Additional Dependencies

Install the remaining Python packages:

```bash
pip install datasets transformers accelerate
```

---

## 4. Create the Fine-Tuning Script

Create a new Python file named `finetunecat.py`:

```bash
nano finetunecat.py
```

Paste the following code into the file:

```python
# Make sure to install the required packages before running this script:
# !pip install transformers datasets torch torch_xla accelerate -qU

import re
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# Configuration
MODEL_NAME = "PrimeIntellect/INTELLECT-1"
DATASET_NAME = "TESTtm7873/ChatCat"  # Update this if necessary

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
assistant_header = "<|im_start|>assistant\n"
assistant_header_tokens = tokenizer.encode(assistant_header, add_special_tokens=False)

def preprocess_function(examples):
    """
    Preprocess dataset examples for training.
    This function expects the dataset to have 'input' and 'output' columns.
    It attempts to extract an emotion from the input text (in the format `<emotion:...>`)
    and then creates a conversation structure using a chat template.
    """
    messages_list = []
    for input_text, assistant_text in zip(examples["input"], examples["output"]):
        # Extract emotion using regex, if present.
        emotion_match = re.search(r"<emotion:(.*?)>", input_text)
        if emotion_match:
            emotion = emotion_match.group(1).strip()
            # Remove the emotion tag from the input text.
            input_text_clean = re.sub(r"<emotion:.*?>", "", input_text).strip()
        else:
            emotion = "neutral"
            input_text_clean = input_text

        # Construct the prompt with the extracted emotion.
        prompt = f"Respond with a {emotion} emotion: {input_text_clean}"
        
        # Build the conversation structure.
        messages_list.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text}
        ])
    
    # Tokenize the conversations using the tokenizer's chat template function.
    tokenized_conversations = tokenizer.apply_chat_template(
        messages_list,
        padding=True,           # Enable padding for uniform tensor shapes.
        add_generation_prompt=False,
        truncation=True,
        return_tensors="pt"
    )
    
    # Build labels for causal LM by masking out non-assistant tokens.
    labels_list = []
    for input_ids in tokenized_conversations:
        input_ids = input_ids.tolist()
        start_idx = None
        # Locate the assistant header in the tokenized sequence.
        for i in range(len(input_ids) - len(assistant_header_tokens) + 1):
            if input_ids[i:i+len(assistant_header_tokens)] == assistant_header_tokens:
                start_idx = i + len(assistant_header_tokens)
                break
        
        if start_idx is None:
            labels = [-100] * len(input_ids)
        else:
            labels = [-100] * start_idx + input_ids[start_idx:]
        
        labels_list.append(labels)
    
    return {
        "input_ids": tokenized_conversations,
        "labels": labels_list
    }

def train_fn(process_index):
    """TPU training function.
    
    The `process_index` argument is passed automatically by xmp.spawn.
    """
    # Load the model and move it to the appropriate TPU device.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    device = xm.xla_device()
    model.to(device)
    
    # Load and preprocess the dataset.
    dataset = load_dataset(DATASET_NAME)
    # Optionally, inspect dataset columns:
    # print("Dataset columns:", dataset["train"].column_names)
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Data collator for causal language modeling.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # TPU-optimized training arguments.
    training_args = TrainingArguments(
        output_dir="./chatcat-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        logging_steps=100,
        save_steps=500,
        report_to="none",
        tpu_num_cores=8,
        optim="adafactor",
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine"
    )
    
    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Begin training.
    trainer.train()
    
    # Save the final model.
    trainer.save_model("./final-model")
    xm.master_print("Training complete! Model saved to ./final-model")

if __name__ == "__main__":
    # xmp.spawn automatically passes the process index to train_fn.
    xmp.spawn(train_fn, args=())
```

> **Tip:** Make sure to save the file after pasting (press `Ctrl+O` then `Ctrl+X` if using nano).

---

## 5. Run the Fine-Tuning Script

Start the training process by running:

```bash
python finetunecat.py
```

Wait for the training to complete. Once finished, the model will be saved to the `./final-model` directory.

---

Happy coding and happy training on TPU!  
*Written by TESTTM, Polished by O3-MINI-HIGH*
