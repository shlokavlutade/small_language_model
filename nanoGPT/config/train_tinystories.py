# Output
experiment = "baseline" 
out_dir = f"/content/drive/MyDrive/thesis_data/results/{experiment}"

# Evaluation
# Final baseline:
eval_interval = 500
eval_iters = 200

# Current test:
# eval_interval = 1000
# eval_iters = 50


# Logging
log_interval = 10


# Dataset
dataset = f"tinystories_{experiment}"


# Model architecture
batch_size = 16
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384

dropout = 0.0
bias = False


# Optimizer
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0


# Learning rate decay

# Final baseline:
max_iters = 5000
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4
decay_lr = True

# Current test run:
# max_iters = 20
# warmup_iters = 100
# lr_decay_iters = 5000
# min_lr = 1e-4
# decay_lr = True

# System
device = "cuda"
dtype = "float16"
compile = True