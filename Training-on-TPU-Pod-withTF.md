
# TPU Tutorial: Dataset Downloading & Training on V4-32

This tutorial will guide you through setting up a TPU (v4-32) on Google Cloud, downloading a dataset into a Cloud Storage bucket, testing your TPU configuration, and finally running a training script. We will cover the following steps:

1. Creating a TPU
2. Creating a Google Cloud Storage bucket
3. Running a dataset download script on the TPU
4. Testing your TPU with a simple script
5. Running a training script using data from your bucket

---

## 1. Create a TPU

First, create a TPU using the following command:

```bash
gcloud alpha compute tpus queued-resources create TPU \
  --node-id TPU \
  --zone us-central2-b \
  --accelerator-type v4-32 \
  --runtime-version tpu-vm-tf-2.16.1-pod-pjrt
```

Wait for the TPU to be created. You can check its status with:

```bash
gcloud alpha compute tpus queued-resources describe TPU --zone us-central2-b
```

---

## 2. Create a Google Cloud Storage Bucket

Next, create a bucket that will store your data:

```bash
gcloud storage buckets create gs://my-bucket-fineweb --location=us-central2
```

---

## 3. Downloading the Dataset

### a. SSH into the TPU

After your TPU and bucket are set up, SSH into the TPU:

```bash
gcloud compute tpus tpu-vm ssh TPU --zone us-central2-b
```

### b. Create the Download Script

Create a new file called `downloadgcs.py`:

```bash
nano downloadgcs.py
```

Then paste the following Python script into the file:

```python
import os
import concurrent.futures
import tensorflow as tf
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import warnings
import threading
import time
import sys

# ======================
# Global Configuration
# ======================
BLOCK_SIZE = 1024
SHARD_SIZE = 1_000_000
TOKENIZER_MODEL = "gpt2"
OUTPUT_DIR = "gs://my-bucket-fineweb/fineweb_tfrecords_tokenized"

# ======================
# Global State
# ======================
tokenizer = None
PAD_TOKEN_ID = None
EOS_TOKEN_ID = None
total_chunks = 0
total_samples = 0
total_shards = 1
total_chunks_lock = threading.Lock()
total_samples_lock = threading.Lock()
total_shards_lock = threading.Lock()
start_time = time.time()
stop_progress = False

# ANSI Color Codes
COLORS = {
    'header': '\033[95m',
    'okblue': '\033[94m',
    'okgreen': '\033[92m',
    'warning': '\033[93m',
    'fail': '\033[91m',
    'endc': '\033[0m',
    'bold': '\033[1m',
    'underline': '\033[4m'
}

def progress_printer():
    """Dynamic progress display with proper cursor control"""
    last_chunks = 0
    last_samples = 0
    emoji_states = ['🚀', '🌠', '✨', '🌟', '💫', '🔥']
    moon_phases = ['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘']
    
    # Reserve space for 2-line display
    print("\n\n", end="")
    
    while not stop_progress:
        time.sleep(0.2)
        # Get metrics
        with total_chunks_lock:
            current_chunks = total_chunks
        with total_samples_lock:
            current_samples = total_samples
        with total_shards_lock:
            current_shards = total_shards
        
        # Calculate metrics
        elapsed = time.time() - start_time
        chunks_per_sec = (current_chunks - last_chunks) / 0.2
        samples_per_sec = (current_samples - last_samples) / 0.2
        shard_progress = (current_chunks % SHARD_SIZE) / SHARD_SIZE
        last_chunks = current_chunks
        last_samples = current_samples
        
        # Dynamic elements
        phase = moon_phases[int(elapsed * 2) % len(moon_phases)]
        emoji = emoji_states[min(int(chunks_per_sec / 500), len(emoji_states)-1)]
        speed_bar = '█' * min(int(chunks_per_sec / 100), 20) + '░' * (20 - min(int(chunks_per_sec / 100), 20))
        
        # Format progress text
        progress_text = (
            f"{COLORS['bold']}{phase}{COLORS['endc']} "
            f"{COLORS['warning']}SHARD:{COLORS['endc']} {COLORS['okgreen']}{current_shards} ({shard_progress:.0%}){COLORS['endc']} "
            f"{COLORS['okblue']}▸{COLORS['endc']} "
            f"{COLORS['warning']}SAMPLES:{COLORS['endc']} {COLORS['okgreen']}{current_samples:,}{COLORS['endc']} "
            f"{COLORS['okblue']}▸{COLORS['endc']} "
            f"{COLORS['warning']}CHUNKS:{COLORS['endc']} {COLORS['okgreen']}{current_chunks:,}{COLORS['endc']}\n"
            f"{COLORS['okblue']}┊{speed_bar}┊{COLORS['endc']} "
            f"{emoji} {COLORS['warning']}{chunks_per_sec:,.0f}/s{COLORS['endc']} "
            f"{COLORS['okblue']}⏱️ {elapsed:.1f}s{COLORS['endc']}"
        )
        
        # Update display
        sys.stdout.write(f"\033[2F\033[K{progress_text}")
        sys.stdout.flush()
    
    # Final cleanup
    sys.stdout.write("\033[2K\033[1B\033[2K")
    sys.stdout.flush()

def tokenize_and_chunk(text: str, sample_idx: int):
    """Convert text to tokenized chunks with padding"""
    global tokenizer, PAD_TOKEN_ID, EOS_TOKEN_ID, BLOCK_SIZE
    try:
        encoding = tokenizer(text, add_special_tokens=False, truncation=False)
        token_ids = encoding["input_ids"]
        chunks = []

        for i in range(0, len(token_ids), BLOCK_SIZE):
            chunk = token_ids[i:i + BLOCK_SIZE]
            if len(chunk) < BLOCK_SIZE:
                chunk += [PAD_TOKEN_ID] * (BLOCK_SIZE - len(chunk))
            labels = chunk[1:] + [EOS_TOKEN_ID]
            chunks.append((chunk, labels))

        return chunks
    except Exception as e:
        print(f"\n{COLORS['fail']}❌ Sample #{sample_idx}: {e}{COLORS['endc']}")
        return []

def serialize_example(input_ids, labels):
    """Create TFRecord Example"""
    return tf.train.Example(features=tf.train.Features(feature={
        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
        "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
    })).SerializeToString()

def main():
    global tokenizer, PAD_TOKEN_ID, EOS_TOKEN_ID, total_chunks, total_samples, total_shards, stop_progress

    print(f"{COLORS['bold']}=== FineWeb TFRecords Conversion ==={COLORS['endc']}")
    
    # Detect and configure CPU cores
    detected_cpus = os.cpu_count()
    display_cpus = detected_cpus or 4  # For display purposes
    cpu_count = detected_cpus or 1     # For thread calculation
    
    # Calculate optimal thread count (use half of available cores)
    NUM_THREADS = max(cpu_count // 2, 1)
    
    print(f"{COLORS['okblue']}TensorFlow: {tf.__version__} | CPU Cores: {display_cpus} | Using threads: {NUM_THREADS}{COLORS['endc']}")

    # Initialize output directory
    try:
        tf.io.gfile.makedirs(OUTPUT_DIR)
        print(f"\n{COLORS['okgreen']}✅ Output directory:{COLORS['endc']} {OUTPUT_DIR}")
    except Exception as e:
        print(f"\n{COLORS['fail']}❌ Storage error: {e}{COLORS['endc']}")
        raise

    # Load dataset
    try:
        dataset = load_dataset("HuggingFaceFW/fineweb", "sample-100BT", split="train", streaming=True)
        print(f"\n{COLORS['okgreen']}✅ Dataset stream initialized{COLORS['endc']}")
    except Exception as e:
        print(f"\n{COLORS['fail']}❌ Dataset error: {e}{COLORS['endc']}")
        raise

    # Initialize tokenizer
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_MODEL)
            tokenizer.model_max_length = int(1e12)  # Disable length warnings
        tokenizer.pad_token = tokenizer.eos_token
        EOS_TOKEN_ID = tokenizer.eos_token_id
        PAD_TOKEN_ID = tokenizer.pad_token_id
        print(f"\n{COLORS['okgreen']}✅ Tokenizer ready{COLORS['endc']} (vocab: {tokenizer.vocab_size})")
    except Exception as e:
        print(f"\n{COLORS['fail']}❌ Tokenizer error: {e}{COLORS['endc']}")
        raise

    # Processing setup
    print(f"\n{COLORS['bold']}🚀 Starting processing pipeline{COLORS['endc']}")
    print(f"{COLORS['okblue']}• Thread workers: {NUM_THREADS}")
    print(f"• Shard size: {SHARD_SIZE:,} chunks")
    print(f"• Block size: {BLOCK_SIZE} tokens{COLORS['endc']}")

    shard_id = 0
    writer = tf.io.TFRecordWriter(f"{OUTPUT_DIR}/fineweb_{shard_id:05d}.tfrecord")
    print(f"\n{COLORS['okgreen']}📦 Initial shard created:{COLORS['endc']} fineweb_{shard_id:05d}.tfrecord")

    # Start progress thread
    progress_thread = threading.Thread(target=progress_printer, daemon=True)
    progress_thread.start()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = []
            
            for sample_idx, ex in enumerate(dataset):
                text = ex.get("text", "")
                if not text: continue

                # Submit processing task
                future = executor.submit(tokenize_and_chunk, text, sample_idx)
                futures.append(future)
                with total_samples_lock:
                    total_samples += 1

                # Process completed tasks while controlling memory
                while len(futures) >= NUM_THREADS * 2:
                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        for input_ids, labels in future.result():
                            writer.write(serialize_example(input_ids, labels))
                            with total_chunks_lock:
                                total_chunks += 1
                                if total_chunks % SHARD_SIZE == 0:
                                    writer.close()
                                    shard_id += 1
                                    with total_shards_lock:
                                        total_shards += 1
                                    writer = tf.io.TFRecordWriter(
                                        f"{OUTPUT_DIR}/fineweb_{shard_id:05d}.tfrecord"
                                    )
                        futures.remove(future)

            # Process remaining tasks
            for future in concurrent.futures.as_completed(futures):
                for input_ids, labels in future.result():
                    writer.write(serialize_example(input_ids, labels))
                    with total_chunks_lock:
                        total_chunks += 1
                        if total_chunks % SHARD_SIZE == 0:
                            writer.close()
                            shard_id += 1
                            with total_shards_lock:
                                total_shards += 1
                            writer = tf.io.TFRecordWriter(
                                f"{OUTPUT_DIR}/fineweb_{shard_id:05d}.tfrecord"
                            )

    except KeyboardInterrupt:
        print(f"\n{COLORS['warning']}⚠️  Processing interrupted!{COLORS['endc']}")
    except Exception as e:
        print(f"\n{COLORS['fail']}❌ Critical error: {e}{COLORS['endc']}")
        raise
    finally:
        stop_progress = True
        progress_thread.join()
        writer.close()
        print(f"\n{COLORS['bold']}📊 Final Report{COLORS['endc']}")
        print(f"{COLORS['okblue']}• Samples processed: {COLORS['okgreen']}{total_samples:,}{COLORS['endc']}")
        print(f"{COLORS['okblue']}• Chunks written: {COLORS['okgreen']}{total_chunks:,}{COLORS['endc']}")
        print(f"{COLORS['okblue']}• TFRecords created: {COLORS['okgreen']}{total_shards}{COLORS['endc']}")
        print(f"{COLORS['okblue']}• Output location: {COLORS['okgreen']}{OUTPUT_DIR}{COLORS['endc']}")
        print(f"\n{COLORS['bold']}=== Processing complete ==={COLORS['endc']}")

if __name__ == "__main__":
    main()
```

> **Note:** Save the file by pressing `Ctrl + O` and exit nano with `Ctrl + X`.

### c. Run the Script in a Tmux Session

To ensure your download continues even if you disconnect from SSH, start a new tmux session:

```bash
tmux new -s downloading
python downloadgcs.py
```

You will see a dynamic progress bar as the script runs. If you need to detach and later reattach to check progress, use:

```bash
tmux attach -t downloading
```

---

## 4. Testing Your TPU

Before proceeding with training, set the following environment variables:

```bash
export TPU_NAME=TPU
export TPU_LOAD_LIBRARY=0
```

Create a test script to verify your TPU configuration:

```bash
nano testtpu.py
```

Paste the following code:

```python
import tensorflow as tf
print("Tensorflow version " + tf.__version__)

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
print('Running on TPU ', cluster_resolver.cluster_spec().as_dict()['worker'])

tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

@tf.function
def add_fn(x, y):
    return x + y

x = tf.constant(1.)
y = tf.constant(1.)
z = strategy.run(add_fn, args=(x, y))
print(z)
```

Save and exit nano (`Ctrl + O`, then `Ctrl + X`), then run the test script:

```bash
python testtpu.py
```

You should see output similar to:

```
Tensorflow version 2.16.1
Running on TPU  {'worker': ['...']}
PerReplica:{
  0: tf.Tensor(2.0, shape=(), dtype=float32),
  1: tf.Tensor(2.0, shape=(), dtype=float32),
  ...
}
```

---

## 5. Running the Training Script

Now that your dataset is downloaded and the TPU is tested, create a training script:

```bash
nano train.py
```

Paste the following code into `train.py`:

```python
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

##############################################################################
# 1. Basic Config & TFRecord Pattern from Your Script (Training Only)
##############################################################################
BLOCK_SIZE = 1024  
global_batch_size = 16  # Simple batch size for demonstration

# Training TFRecord pattern from your bucket
train_pattern = "gs://fineweb-training/fineweb_tfrecords_tokenized/fineweb_000*.tfrecord"

# Define the feature description (from your script)
feature_description = {
    "input_ids": tf.io.FixedLenFeature([BLOCK_SIZE], tf.int64),
    "labels": tf.io.FixedLenFeature([BLOCK_SIZE], tf.int64),
}

def parse_example(example_proto):
    # Parse the input tf.Example proto using the feature description.
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    input_ids = tf.cast(parsed["input_ids"], tf.int32)
    labels = tf.cast(parsed["labels"], tf.int32)
    return input_ids, labels

##############################################################################
# 2. Create the Training Dataset from TFRecord Files
##############################################################################
train_files = tf.data.Dataset.list_files(train_pattern, shuffle=True)
# Optionally, take a subset of shards for demonstration purposes.
train_files = train_files.take(1)

train_ds = train_files.interleave(
    lambda fp: tf.data.TFRecordDataset(fp, num_parallel_reads=tf.data.AUTOTUNE),
    cycle_length=1,  # Process one file at a time for simplicity
    block_length=16,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False
)
train_ds = train_ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(global_batch_size, drop_remainder=True)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

##############################################################################
# 3. Build a Simple Model
##############################################################################
# In this minimal example, we build a simple language-model–style network:
# - An Embedding layer
# - A GlobalAveragePooling1D layer to collapse the sequence dimension
# - A Dense softmax output layer that predicts token probabilities
vocab_size = 50257   # GPT-2 vocab size (as in your tokenizer script)
embedding_dim = 128  # Small embedding dimension for demonstration

inputs = keras.Input(shape=(BLOCK_SIZE,), dtype=tf.int32, name="input_ids")
# Note: We do not use mask_zero since token 0 may be valid.
x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
x = layers.GlobalAveragePooling1D()(x)
# Output shape: (batch_size, vocab_size) — one prediction per sequence.
outputs = layers.Dense(vocab_size, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

##############################################################################
# 4. Train the Model (Without Validation)
##############################################################################
# This example trains the model for one epoch. Adjust epochs and steps per epoch as needed.
model.fit(train_ds, epochs=1)
```

Save and exit nano (`Ctrl + O`, then `Ctrl + X`), and run the training script:

```bash
python train.py
```

> **Note:** The training code is untested at this stage—it should work, but further refinements and output examples may be added later.

---

## Final Notes

- **Cost Considerations:** Keep in mind that Cloud Storage buckets and TPU usage incur costs on your Google Cloud account.
- **TMUX Usage:** Running your long processes in tmux sessions ensures that you don’t lose progress when disconnecting from SSH.

---

Writen by TESTTM, Polished by O3-MINI-HIGH
