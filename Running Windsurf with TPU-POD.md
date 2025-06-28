# Connecting Windsurf to Your Google Cloud TPU Pod
*The Definitive Guide*

> **Goal:** Walk through a complete, battle-tested workflow for connecting the Windsurf IDE to a Google Cloud **TPU VM**, from SSH configuration to automated training runs.

---

## Table of Contents
1. [The Core Problem: Making SSH Aware of Your TPU](#1-the-core-problem-making-ssh-aware-of-your-tpu)
2. [Connecting from Windsurf](#2-connecting-from-windsurf)
3. [Creating a Bulletproof Workflow with Scripts](#3-creating-a-bulletproof-workflow-with-scripts)
4. [Teaching the AI: The Windsurf Rule File](#4-teaching-the-ai-the-windsurf-rule-file)
5. [Conclusion](#5-conclusion)

---

## 1. The Core Problem: Making SSH Aware of Your TPU

Before Windsurf can connect, your local SSH client needs three essential pieces of information about your TPU VM:

1. **IP address**
2. **Username**
3. **Private key** for authentication

Google's `gcloud` CLI can populate your `~/.ssh/config`, but the required command is in the **alpha** track.

### 1.1. Install the Alpha Component

```bash
gcloud components install alpha
```

⚠️ **Important:** Open a **new** terminal after the installation completes.

### 1.2. Generate the SSH Configuration

```bash
gcloud alpha compute tpus tpu-vm config-ssh \
  --project <YOUR_PROJECT_ID> \
  --zone <YOUR_TPU_ZONE>
```

> 💡 **Helper Note:** Replace `<YOUR_PROJECT_ID>` (e.g., `my-tpu-project-12345`) and `<YOUR_TPU_ZONE>` (e.g., `us-central2-b`).

This command appends a configuration block like this to your `~/.ssh/config`:

```sshconfig
# --- Added automatically by gcloud ---
Host tpu-Name
    HostName 34.109.xxx.xxx   # external IP of the TPU VM
    User pc                   # TPU username
    IdentityFile ~/.ssh/google_compute_engine
    IdentitiesOnly yes
    ServerAliveInterval 30
```

### 1.3. No External IP? Use IAP Tunneling

1. **Generate the ProxyCommand** (dry-run):

   ```bash
   gcloud compute tpus tpu-vm ssh TPUJ \
     --zone us-central2-b \
     --tunnel-through-iap \
     --dry-run
   ```

2. **Copy** the long `ProxyCommand …` line into the correct `Host` block of `~/.ssh/config`.

3. **Permissions:** Your account must have the `iap.tunnelResourceAccessor` IAM role.

### 1.4. Manual Configuration (If Automatic Methods Fail)

If the automatic methods don't work, try configuring manually:

1. Navigate to **Google Cloud Console** → **Compute Engine** → **TPUs**
2. Copy the **External IP** address
3. Note the SSH username (usually shown as `user@hostname` in SSH examples)
4. Locate your identity file:
   - **Windows:** `C:\Users\[USERNAME]\.ssh\google_compute_engine`
   - **Mac/Linux:** `~/.ssh/google_compute_engine`
   - Enable viewing hidden folders if necessary
5. Add the configuration manually to your SSH config file

---

## 2. Connecting from Windsurf

### Step-by-Step Connection Process

1. Press **`Ctrl + Shift + P`** → **Remote-SSH: Connect to Host…**
2. Select **"Add New SSH Host"**
3. Paste your SSH configuration:

   ```sshconfig
   Host tpu-NAME
       HostName 34.109.xxx.xxx   # external IP of the TPU VM
       User pc                   # TPU username
       IdentityFile ~/.ssh/google_compute_engine
       IdentitiesOnly yes
       ServerAliveInterval 30
   ```

4. Choose the host you just created (e.g., `tpu-NAME`)

> ⚠️ **Crucial:** Windsurf ships with its *own* SSH extension. **Do not** install Microsoft's "Remote - SSH" extension as it conflicts with Windsurf's implementation.

---

## 3. Creating a Bulletproof Workflow with Scripts

Create a `scripts/` directory in your project and add the following helper scripts.

> 🚀 **Quick Setup:** After saving the scripts, make them executable:
>
> ```bash
> chmod +x scripts/tpuj_*.sh
> ```

### 3.1. `scripts/tpuj_sync.sh`
*Synchronize code to all TPU workers*

```bash
#!/usr/bin/env bash
# Copy updated train.py to every TPU worker
set -euo pipefail

# --- CONFIGURATION ---
TPU_NAME="TPUJ"
ZONE="us-central2-b"
SCRIPT_TO_SYNC="train.py"
# ---------------------

gcloud compute tpus tpu-vm scp "./$SCRIPT_TO_SYNC" "$TPU_NAME:" \
  --worker=all --zone="$ZONE"
```

### 3.2. `scripts/tpuj_launch.sh`
*Launch training on the entire pod*

```bash
#!/usr/bin/env bash
# Launch train.py on the entire pod
set -euo pipefail

# --- CONFIGURATION ---
TPU_NAME="TPUJ"
ZONE="us-central2-b"
SCRIPT_TO_RUN="train.py"
# ---------------------

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --worker=all \
  --command="python3 $SCRIPT_TO_RUN"
```

### 3.3. `scripts/tpuj_restart.sh`
*Restart stuck processes*

```bash
#!/usr/bin/env bash
# Restart stuck Python processes on every worker
set -euo pipefail

# --- CONFIGURATION ---
TPU_NAME="TPUJ"
ZONE="us-central2-b"
# ---------------------

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --worker=all \
  --command="sudo pkill -u \$(whoami) python3; sudo pkill -u \$(whoami) python"
```

### 3.4. `scripts/tpuj_test.sh`
*Quick validation run*

```bash
#!/usr/bin/env bash
# 200-step sanity check
set -euo pipefail

# --- CONFIGURATION ---
TPU_NAME="TPUJ"
ZONE="us-central2-b"
SCRIPT_TO_RUN="train.py"
# ---------------------

# 1. Sync latest code
gcloud compute tpus tpu-vm scp "./$SCRIPT_TO_RUN" "$TPU_NAME:" \
  --worker=all --zone="$ZONE"

# 2. Run the training script (steps capped inside train.py)
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --worker=all \
  --command="python3 $SCRIPT_TO_RUN"
```

---

## 4. Teaching the AI: The Windsurf Rule File

Save this configuration as `.windsurf/rules/tpuj-training-protocol.md` to guide Windsurf's AI behavior:
or add it as a Workspace rule in GUI, which will look like this:

![Windsurf Rules Configuration](./images/Zrzut%20ekranu%202025-06-28%20144501.png)

```markdown
# TPUJ Training Protocol (v3)

**Audience:** Cascade (Windsurf's AI agent)  
**Scope:** All interactions with the *TPUJ* Cloud-TPU pod

---

## 0. Golden Rule
> **Never run raw Python directly on the pod.**  
> **Always** use approved helper scripts or sync Python files first with `scripts/tpuj_sync.sh`.

---

## 1. Helper Scripts Reference

| Script | Purpose | Auto-executable |
|--------|---------|----------------|
| `scripts/tpuj_sync.sh` | Copy updated Python files to all workers | ✅ |
| `scripts/tpuj_test.sh` | Run 200-step validation | ✅ |
| `scripts/tpuj_launch.sh` | Start full production training | ⚠️ Human approval required |
| `scripts/tpuj_restart.sh` | Kill stuck Python processes | ✅ |

---

## 2. Standard Workflows

| Scenario | Required Actions | Command Sequence |
|----------|------------------|------------------|
| **Code Changes** | Sync after any `.py` file edit | `scripts/tpuj_sync.sh` |
| **Quick Test** | Debug/experiment runs (≤200 steps) | `scripts/tpuj_test.sh` |
| **Production Run** | Full training with human approval | `tpuj_sync.sh` → `tpuj_launch.sh` |
| **Recovery** | Fix "TPU hosts could not initialize" | `scripts/tpuj_restart.sh` |

---

## 3. Running New Code Snippets

1. **Save** snippet in `tpu_snippets/` (e.g., `tpu_snippets/exp_2025-01-15.py`)
2. **Version control** with `git add` (optional)
3. **Sync** to workers: `scripts/tpuj_sync.sh`
4. **Execute** via test harness: `scripts/tpuj_test.sh`

> 🚫 **Forbidden:** Interactive Python execution or here-doc patterns

---

## 4. Safety Guidelines

- ✅ Use helper scripts exclusively
- ✅ Save code to files before execution
- ✅ Maintain ≤200 step limit for tests
- ✅ Re-sync when in doubt (idempotent operation)
- ❌ No raw `gcloud` commands
- ❌ No inline Python execution
- ❌ No production runs without approval

---

## 5. Example Interactions

**User Request:** "Try this new attention module"  
**AI Response:** Create `tpu_snippets/attention_exp.py` → Sync → Test → Stream logs

**User Request:** "Run full training"  
**AI Response:** Request production approval → Sync → Launch → Monitor

**Error:** "TPU hosts initialization failed"  
**AI Response:** Execute restart → Re-sync → Retry original action
```

---

## 5. Conclusion

With proper SSH configuration, automated helper scripts, and comprehensive AI guidelines, you now have a professional-grade TPU development workflow that:

- ✅ Minimizes manual errors
- ✅ Maximizes reproducibility  
- ✅ Ensures safe operations
- ✅ Streamlines the development cycle

**Happy coding and successful training! 🚀**

The final workflow will look like this:

![Final Workflow Result](./images/Zrzut%20ekranu%202025-06-28%20145136.png)
