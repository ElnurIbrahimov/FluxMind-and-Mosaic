# FluxMind Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model Components](#model-components)
3. [DSL System](#dsl-system)
4. [Training](#training)
5. [Inference](#inference)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)

---

## Architecture Overview

FluxMind uses a meta-learning architecture designed for rapid pattern acquisition from minimal examples.

```
┌─────────────────────────────────────────────────────────────────┐
│                        FluxMind v0.83                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Support Set (32 examples)                                       │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                      │
│  │ Before  │ +  │   Op    │ +  │  After  │                      │
│  │ (4×4)   │    │  (emb)  │    │  (4×4)  │                      │
│  └────┬────┘    └────┬────┘    └────┬────┘                      │
│       │              │              │                            │
│       └──────────────┴──────────────┘                            │
│                      │                                           │
│              ┌───────▼───────┐                                   │
│              │Support Encoder│                                   │
│              │   (MLP + Att) │                                   │
│              └───────┬───────┘                                   │
│                      │                                           │
│              Support Encoding                                    │
│                      │                                           │
├──────────────────────┼──────────────────────────────────────────┤
│                      │                                           │
│  Query               │         Reasoning Circuits                │
│  ┌─────────┐    ┌────┴────┐   ┌─────────────────────────┐       │
│  │ Before  │    │   Op    │   │ Comparison │ Permutation │       │
│  │ (4×4)   │    │  (emb)  │   │  Circuit   │   Circuit   │       │
│  └────┬────┘    └────┬────┘   └─────┬──────┴──────┬──────┘       │
│       │              │              │             │               │
│       └──────────────┘              │   Parameter │               │
│              │                      │   Circuit   │               │
│      ┌───────▼───────┐              └──────┬──────┘               │
│      │ Cross-Attention│◄──── Support Encoding                    │
│      │  (Query → Sup) │              │                            │
│      └───────┬───────┘              │                            │
│              │                       │                            │
│              └───────────────────────┘                            │
│                          │                                        │
│                  ┌───────▼───────┐                               │
│                  │   Predictor   │                               │
│                  │     (MLP)     │                               │
│                  └───────┬───────┘                               │
│                          │                                        │
│                  ┌───────▼───────┐                               │
│                  │  Bit Logits   │                               │
│                  │    (4×4)      │                               │
│                  └───────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Bit-Level Representation**: States are encoded as 4×4 binary arrays
2. **Per-Operation Attention**: Model attends only to examples matching query operation
3. **Reasoning Circuits**: Specialized modules for different pattern types
4. **Meta-Learning**: Learns to learn from examples, not specific patterns

---

## Model Components

### BitFluxMind Core

The main model class implementing the meta-learning architecture.

```python
class FluxMindV083(nn.Module):
    def __init__(self, config: FluxMindV083Config):
        # Bit embedding: 2 -> bit_embed_dim
        self.bit_embed = nn.Embedding(2, config.bit_embed_dim)

        # State encoders
        self.before_encoder = nn.Sequential(...)
        self.after_encoder = nn.Sequential(...)

        # Transition network: encodes (before, after) pairs
        self.transition_net = nn.Sequential(...)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(...)

        # Reasoning circuits
        self.comparison_circuit = ComparisonCircuit(...)
        self.permutation_circuit = PermutationCircuit(...)
        self.parameter_circuit = ParameterInferenceCircuit(...)

        # Output predictor
        self.predictor = nn.Sequential(...)
```

### Reasoning Circuits

#### ComparisonCircuit
Handles conditional logic by computing pairwise dimension comparisons.

```python
class ComparisonCircuit(nn.Module):
    """
    Computes features for all 6 pairs of dimensions:
    (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

    Useful for min/max operations and comparisons.
    """
```

#### PermutationCircuit
Tracks dimension swaps using self-attention over support examples.

```python
class PermutationCircuit(nn.Module):
    """
    Identifies permutation patterns in support set.
    Uses attention to find consistent swap relationships.
    """
```

#### ParameterInferenceCircuit
Infers hidden parameters (e.g., modulus values) from examples.

```python
class ParameterInferenceCircuit(nn.Module):
    """
    Predicts discrete parameter values from support context.
    Outputs soft embedding of inferred parameter.
    """
```

### Circuit Gating

A learned gate determines which circuits to use for each query:

```python
self.circuit_gate = nn.Sequential(
    nn.Linear(example_embed_dim, 64),
    nn.LayerNorm(64),
    nn.GELU(),
    nn.Linear(64, 3),  # 3 circuits
    nn.Sigmoid()
)
```

---

## DSL System

### State Representation

States are 4-dimensional vectors with values in [1, 15]:

```python
state = [3, 7, 12, 1]  # Example state
```

### Bit Encoding

States are converted to 4×4 binary arrays:

```python
def state_to_bits(state):
    """Convert state to binary representation."""
    bits = np.zeros((4, 4), dtype=np.float32)
    for dim in range(4):
        for bit in range(4):
            bits[dim, bit] = ((state[dim] - 1) >> bit) & 1
    return bits
```

### DSL Families

| Family | Description | Example |
|--------|-------------|---------|
| Arithmetic | Add/subtract constants | `x[0] += 3` |
| Bitwise | XOR, AND, OR with masks | `x[1] ^= 7` |
| Comparison | Min/max between dimensions | `x[2] = min(x[2], x[3])` |
| Modular | Modular arithmetic | `x[0] = (x[0] + 5) % 11` |
| Shift | Bit shift operations | `x[3] <<= 2` |
| Swap | Exchange dimensions | `swap(x[0], x[1])` |
| Mixed | Combinations of above | Various |

### DSL Generator

```python
class DSLGenerator:
    def generate_train_test_dsls(self, n_train_per_family=12, n_test_per_family=3):
        """
        Generate diverse DSLs for training and testing.

        Returns:
            train_dsls: List of 84 training DSLs (12 per family)
            test_dsls: List of 21 test DSLs (3 per family)
        """
```

---

## Training

### Configuration

```python
@dataclass
class FluxMindV083Config:
    state_dim: int = 4           # Dimensions in state
    bits_per_value: int = 4      # Bits per dimension
    num_operations: int = 8      # Operations per DSL
    bit_embed_dim: int = 48      # Bit embedding dimension
    bit_state_embed_dim: int = 192
    bit_example_embed_dim: int = 288
    bit_context_dim: int = 288
    bit_hidden_dim: int = 576
    bit_num_heads: int = 8
    dropout: float = 0.1
    comparison_dim: int = 64
    permutation_dim: int = 64
    parameter_dim: int = 32
```

### Training Loop

```python
# Each epoch: 100 batches
for batch in range(100):
    # 1. Sample random DSL
    dsl = random.choice(train_dsls)

    # 2. Generate examples
    examples = generate_examples(dsl, n=64)
    support = examples[:32]
    query = examples[32]

    # 3. Forward pass
    support_enc = model.encode_support(support_before, support_ops, support_after)
    bit_logits, confidence = model(query_bits, query_op, support_enc, support_ops)

    # 4. Compute loss
    loss = F.binary_cross_entropy_with_logits(bit_logits, target_bits)

    # 5. Backprop
    loss.backward()
    optimizer.step()
```

### Training Command

```bash
python src/train_v083.py \
    --epochs 5000 \
    --lr 1e-4 \
    --device cuda \
    --eval_interval 500 \
    --n_train_per_family 12 \
    --n_test_per_family 3
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 1e-4 | With cosine annealing |
| Weight Decay | 0.01 | AdamW regularization |
| Batch Size | 32 | Queries per batch |
| Support Size | 32 | Examples per query |
| Gradient Clip | 0.5 | Stability |
| Epochs | 5000 | Full training |

---

## Inference

### Basic Usage

```python
import torch
from train_v083 import FluxMindV083, FluxMindV083Config, state_to_bits

# Load model
config = FluxMindV083Config()
model = FluxMindV083(config)
ckpt = torch.load('checkpoints/fluxmind_v083_best.pt')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Prepare support set (32 examples of the new DSL)
support_before = torch.tensor([state_to_bits(ex[0]) for ex in support_examples])
support_ops = torch.tensor([ex[1] for ex in support_examples])
support_after = torch.tensor([state_to_bits(ex[2]) for ex in support_examples])

# Encode support set
support_enc = model.encode_support(
    support_before.unsqueeze(0),
    support_ops.unsqueeze(0),
    support_after.unsqueeze(0)
)

# Query
query_bits = torch.tensor(state_to_bits([3, 7, 12, 1])).unsqueeze(0)
query_op = torch.tensor([2])  # Operation index

# Predict
with torch.no_grad():
    bit_logits, confidence = model(query_bits, query_op, support_enc, support_ops.unsqueeze(0))
    pred_bits = (torch.sigmoid(bit_logits) > 0.5).float()
    pred_state = bits_to_state(pred_bits[0].numpy())

print(f"Predicted state: {pred_state}")
print(f"Confidence: {torch.sigmoid(confidence).item():.2%}")
```

### Batch Inference

```python
# Multiple queries with same support set
queries = [
    (state_to_bits([1, 2, 3, 4]), 0),
    (state_to_bits([5, 6, 7, 8]), 1),
    (state_to_bits([9, 10, 11, 12]), 2),
]

query_bits = torch.stack([torch.tensor(q[0]) for q in queries])
query_ops = torch.tensor([q[1] for q in queries])

# Expand support for batch
batch_size = len(queries)
support_enc_batch = support_enc.expand(batch_size, -1, -1)
support_ops_batch = support_ops.unsqueeze(0).expand(batch_size, -1)

with torch.no_grad():
    bit_logits, confidences = model(query_bits, query_ops, support_enc_batch, support_ops_batch)
```

---

## Configuration

### Model Size Variants

| Variant | Parameters | VRAM | Use Case |
|---------|------------|------|----------|
| Small | ~450K | 2GB | Edge devices |
| Base | ~900K | 4GB | Standard |
| Large | ~2M | 8GB | Maximum accuracy |

### Adjusting Configuration

```python
# Smaller model for edge deployment
config = FluxMindV083Config(
    bit_embed_dim=32,
    bit_state_embed_dim=128,
    bit_example_embed_dim=192,
    bit_context_dim=192,
    bit_hidden_dim=384,
    bit_num_heads=4
)

# Larger model for maximum accuracy
config = FluxMindV083Config(
    bit_embed_dim=64,
    bit_state_embed_dim=256,
    bit_example_embed_dim=384,
    bit_context_dim=384,
    bit_hidden_dim=768,
    bit_num_heads=12
)
```

---

## API Reference

### FluxMindV083

```python
class FluxMindV083(nn.Module):
    """
    Main FluxMind model for meta-learning over DSLs.

    Args:
        config: FluxMindV083Config instance

    Methods:
        encode_support(bits_before, ops, bits_after) -> Tensor
            Encode support set examples into context vectors.

        forward(query_bits, query_op, support_context, support_ops) -> Tuple[Tensor, Tensor]
            Predict output bits and confidence for query.
    """
```

### DSLGenerator

```python
class DSLGenerator:
    """
    Generate diverse DSLs for training and evaluation.

    Args:
        seed: Random seed for reproducibility

    Methods:
        generate_train_test_dsls(n_train_per_family, n_test_per_family)
            Generate train/test split of DSLs.
    """
```

### Utility Functions

```python
def state_to_bits(state: List[int]) -> np.ndarray:
    """Convert 4-dim state to 4x4 bit array."""

def bits_to_state(bits: np.ndarray) -> List[int]:
    """Convert 4x4 bit array back to state."""

def generate_examples(dsl, n: int, rng) -> List[Tuple]:
    """Generate n random examples from a DSL."""
```

---

## Performance Optimization

### GPU Optimization

```python
# Use mixed precision
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    bit_logits, confidence = model(query_bits, query_op, support_enc, support_ops)
    loss = F.binary_cross_entropy_with_logits(bit_logits, target_bits)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Inference Optimization

```python
# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Use inference mode
with torch.inference_mode():
    bit_logits, confidence = model(query_bits, query_op, support_enc, support_ops)
```

---

*For questions or issues, please open a GitHub issue.*
