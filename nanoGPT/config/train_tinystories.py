# Output
out_dir= "/kaggle/working/upload_results/training-results"
dataset = "baseline"

# Evaluation
# Final:
eval_interval = 500
eval_iters = 200

# Test:
# eval_interval = 1000
# eval_iters = 50


# Logging
log_interval = 10


# Model architecture
batch_size = 16
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384

# Optimizer
learning_rate = 1e-3
weight_decay = 1e-1


# Learning rate decay
# Final baseline:
max_iters = 5000
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4
decay_lr = True

# Test:
# max_iters = 20
# warmup_iters = 100
# lr_decay_iters = 5000
# min_lr = 1e-4
# decay_lr = True

# System
device = "cuda"
dtype = "float16"
compile = True