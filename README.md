# ProperModelVault

## Description
ProperModelVault is a secure, file-system based registry for machine learning models. It focuses on data integrity and safe storage using atomic writes and SHA-256 checksums. By default, it uses `safetensors` to avoid the security risks of pickle while ensuring fast loading speeds (the original ModelVault used pickle!)

## Features
- **Atomic Saves**: Prevents data corruption by writing to temp files first.
- **Integrity Checks**: Automatically verifies SHA-256 hashes on every load.
- **Safe Serialization**: Uses `safetensors` by default (no code execution risks).
- **Version Control**: Automatic versioning with metadata tracking.
- **Pluggable Backend**: Easy to swap serializers via Protocol interface.

## Why not pickle?
Pickle is unsafe because it can execute arbitrary code during loading. ModelVault defaults to `safetensors` to keep your environment secure and your loads fast.

## Requirements
```bash
pip install torch safetensors
```

## Quick Start
```python
import torch.nn as nn
from modelvault import ModelVault

# define your model architecture
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

# initialize vault
vault = ModelVault("./models")

# save a model instance
model = MyNet()
version = vault.save("classifier", model, metadata={"accuracy": 0.92, "epoch": 10})
print(f"Saved classifier as version {version}")

# load the latest version
empty_model = MyNet()
loaded_model = vault.load_latest("classifier", model=empty_model)
```
