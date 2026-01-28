# Changelog

All notable changes to FluxMind are documented in this file.

## [0.83] - January 2026

### Overview
**Reasoning Circuits Edition** - Added specialized circuits for different reasoning patterns.

### Changes
- **ComparisonCircuit**: Pairwise dimension comparisons for conditional logic
- **PermutationCircuit**: Tracks dimension swaps via attention
- **ParameterInferenceCircuit**: Infers hidden parameters (modulus, constants)
- **Circuit Gating**: Learns when to use each circuit
- **Fixed NaN Issues**: Proper initialization and LayerNorm throughout
- **Balanced Evaluation**: Equal representation of all DSL families in test set

### Results
| Metric | v0.82 | v0.83 |
|--------|-------|-------|
| Overall Test Accuracy | 67.7% | 80%+ target |
| Swap Family | 63.0% | 97.8% |
| Comparison Family | 44.7% | 89.7% |

---

## [0.82] - January 2026

### Overview
**Scaled Model** - 2M parameter version with improved coverage.

### Changes
- Scaled model to 2,066,673 parameters
- 100 unique DSLs (80 train / 20 test)
- 5000 epoch training on RTX 4060
- 255 minutes training time

### Results
| Metric | Value |
|--------|-------|
| Best Test Accuracy | 94.0% |
| Final Test Accuracy | 86.2% |
| Train Accuracy | 90.6% |

---

## [0.81] - January 27, 2026

### Overview
**TRUE META-LEARNING ACHIEVED** - First version to demonstrate generalization to completely novel DSLs.

### Key Achievement
The model learned **HOW to learn from examples**, not just specific DSL patterns. Test accuracy matches train accuracy, proving true generalization.

### Changes
- Massive-scale training: 80 train / 20 test DSLs
- BitFluxMind architecture (877K params)
- Per-operation attention mechanism
- Bit-level state representation (4x4 binary)

### Results
| Metric | Value | vs Random |
|--------|-------|-----------|
| Test Accuracy (Novel DSLs) | 78.6% | 12.6x |
| Best Single DSL | 99% (Swap_77) | - |
| Model Size | ~3.5 MB | - |
| Inference Time | <1ms | - |

### Per-Family Results (Novel DSLs)
| Family | Accuracy |
|--------|----------|
| Swap | 97.4% |
| Arithmetic | 81.0% |
| Shift | 75.2% |
| Mixed | 71.5% |
| Bitwise | 70.0% |
| Comparison | 68.3% |
| Modular | 61.5% |

---

## [0.80] - January 2026

### Overview
**Hybrid Routing** - Combined multiple approaches.

### Changes
- HybridFluxMind architecture
- Attempted continual learning approaches (EWC, PackNet, NeuroDream)
- These approaches failed to improve meta-learning

### Results
- 94.4% on known DSLs
- Failed to generalize to novel DSLs (17-18% accuracy)

### Lessons Learned
- Regularization tricks don't solve meta-learning
- Need diversity of tasks, not clever constraints

---

## [0.77.1] - January 2026

### Overview
**Critical Fixes** for DSL encoder being ignored.

### Changes
- **FiLM Conditioning**: DSL embedding modulates hidden layers
- **Op Permutation**: Randomize operation indices per episode
- **Contrastive Loss**: InfoNCE pushes different DSL embeddings apart
- **Multi-Sample Training**: Sample K=3 DSL embeddings, average loss

### Model Specs
| Spec | v0.77.0 | v0.77.1 |
|------|---------|---------|
| DSL Embedding Dim | 32 | 64 |
| Architecture | Concat | FiLM |
| Parameters | 656K | ~750K |

---

## [0.77.0] - January 2026

### Overview
First meta-learning version. Introduced DSL induction from examples.

### Added
- `MetaFluxMindCore`: Main model with meta-learning
- `DSLEncoder`: Encodes support set into DSL embedding
- Probabilistic embedding: (mu, log_var) output
- Uncertainty-aware inference

### Known Issues (Fixed in v0.77.1)
- DSL encoder output ignored by core network
- Confidence collapsed to constant ~0.36
- Few-shot curve flat (more examples don't help)

---

## [0.76.4] - January 2026

### Overview
Final stable release before meta-learning extension.

### Highlights
- 99.0% 5-DSL compositional accuracy
- 99.5% length generalization (4->12 steps)
- Properly calibrated confidence

---

## [0.75] - January 2026

### Overview
Basic multi-DSL support with compositional reasoning.

### Results
- 99.6% compositional OOD accuracy
- Strong length generalization

---

## Version Numbering

```
0.83.x  - Reasoning Circuits
0.82.x  - Scaled 2M Model
0.81.x  - True Meta-Learning
0.80.x  - Hybrid Routing
0.77.x  - MetaFluxMind (meta-learning)
0.76.x  - FluxMind (calibrated)
0.75.x  - FluxMind (compositional)
```

---

## Roadmap

### v0.85 (Future)
- Confidence calibration for novel DSLs
- Variable support set sizes (10-100 examples)
- Multi-step reasoning chains

### v0.90 (Future)
- Production API
- Edge deployment optimization
- Custom DSL interface

### v1.0 (Future)
- Public SDK release
- Web playground
- Pre-trained model zoo

---

*Maintained by Elnur Ibrahimov*
