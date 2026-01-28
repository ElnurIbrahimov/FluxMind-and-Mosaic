# FluxMind

**A Meta-Learning Architecture for Rapid Pattern Acquisition**

FluxMind is a lightweight meta-learning system that can learn custom reasoning rules from minimal examples (10-30 demonstrations) and apply them with sub-millisecond inference times.

> *"Upload 10 examples -> Custom reasoning engine in seconds"*

## Key Results

| Metric | Value |
|--------|-------|
| Test Accuracy (Novel DSLs) | **94.0%** |
| Model Parameters | 2.07M |
| Inference Time | <1ms |
| Random Baseline | 6.25% |

FluxMind demonstrates **true meta-learning**: the ability to generalize to completely novel Domain-Specific Languages (DSLs) never seen during training.

## What Makes FluxMind Different

Unlike traditional models that memorize patterns, FluxMind learns *how to learn from examples*:

- **Meta-Learning**: Given 32 examples of a new rule system, FluxMind correctly predicts outputs 94% of the time
- **Edge-Deployable**: ~8MB model size, real-time inference on consumer hardware
- **No Fine-Tuning Required**: Adapts to new DSLs at inference time without retraining

## Architecture

FluxMind uses a specialized attention-based architecture with:

- **Support Set Encoding**: Learns from (before, operation, after) triplets
- **Cross-Attention**: Query attends to relevant support examples
- **Reasoning Circuits**: Specialized modules for comparison, permutation, and parameter inference
- **Bit-Level Representation**: 4x4 binary encoding captures structural patterns

## Performance by DSL Family

| Family | Accuracy |
|--------|----------|
| Swap | 97.8% |
| Comparison | 89.7% |
| Arithmetic | 89.3% |
| Shift | 86.5% |
| Mixed | 77.0% |
| Modular | 67.0% |
| Bitwise | 65.0% |

## Quick Start

### Training

```bash
python src/train_v083.py --epochs 5000 --lr 1e-4 --device cuda
```

### Using a Trained Model

```python
import torch
from src.train_v083 import FluxMindV083, FluxMindV083Config

# Load model
config = FluxMindV083Config()
model = FluxMindV083(config)
checkpoint = torch.load('checkpoints/fluxmind_v083_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference: provide support examples and query
# See docs/TECHNICAL.md for detailed usage
```

## Project Structure

```
FluxMind/
├── README.md              # This file
├── LICENSE                # MIT License
├── CHANGELOG.md           # Version history
├── .gitignore
├── docs/
│   ├── PAPER.md           # FluxMind research paper
│   ├── TECHNICAL.md       # Technical documentation
│   └── MOSAIC_PAPER.md    # MOSAIC cognitive architecture paper
├── src/
│   └── train_v083.py      # Training script with model definition
└── checkpoints/           # Model checkpoints (.pt files)
```

## Relationship to MOSAIC

FluxMind is the empirical validation component of **MOSAIC** (Modular Open-ended Self-Advancing Intelligence Construct), a cognitive architecture for continuously learning AI systems. FluxMind demonstrates that the core meta-learning principles underlying MOSAIC are sound.

See [docs/MOSAIC_PAPER.md](docs/MOSAIC_PAPER.md) for the full MOSAIC architecture proposal.

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | GTX 1060 (6GB) | RTX 3060+ |
| VRAM | 4 GB | 8 GB |
| RAM | 8 GB | 16 GB |
| Training Time (5000 epochs) | ~4 hours | ~2.5 hours |

## Citation

If you use FluxMind in your research, please cite:

```bibtex
@software{fluxmind2026,
  author = {Elnur Ibrahimov},
  title = {FluxMind: A Meta-Learning Architecture for Rapid Pattern Acquisition},
  year = {2026},
  url = {https://github.com/elnur-ibrahimov/fluxmind}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Elnur Ibrahimov** - Independent Researcher, Azerbaijan

> **Status:** Active research. FluxMind validated.
> Seeking collaborators with GPU access.
> Contact: elnuribrahimov83@gmail.com

---

*FluxMind - Teaching machines how to learn*
