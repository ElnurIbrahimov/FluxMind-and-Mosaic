"""
FluxMind v0.83 Training - FIXED VERSION
=======================================
Fixed NaN issues with proper initialization and normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
import time
import random
import argparse
from collections import defaultdict

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class FluxMindV083Config:
    state_dim: int = 4
    bits_per_value: int = 4
    num_operations: int = 8
    bit_embed_dim: int = 48
    bit_state_embed_dim: int = 192
    bit_example_embed_dim: int = 288
    bit_context_dim: int = 288
    bit_hidden_dim: int = 576
    bit_num_heads: int = 8
    dropout: float = 0.1
    comparison_dim: int = 64
    permutation_dim: int = 64
    parameter_dim: int = 32


# =============================================================================
# REASONING CIRCUITS (FIXED)
# =============================================================================

class ComparisonCircuit(nn.Module):
    def __init__(self, bit_embed_dim, bits_per_value, comparison_dim, output_dim):
        super().__init__()
        self.n_pairs = 6
        dim_input = bits_per_value * bit_embed_dim
        self.dim_encoder = nn.Sequential(
            nn.Linear(dim_input, comparison_dim), 
            nn.LayerNorm(comparison_dim),  # Added
            nn.GELU()
        )
        self.pair_compare = nn.Sequential(
            nn.Linear(comparison_dim * 2, comparison_dim), 
            nn.LayerNorm(comparison_dim),  # Added
            nn.GELU(),
            nn.Linear(comparison_dim, comparison_dim)
        )
        self.comparison_out = nn.Sequential(
            nn.Linear(self.n_pairs * comparison_dim, output_dim),
            nn.LayerNorm(output_dim)  # Added
        )
        
    def forward(self, state_bits_embedded):
        batch = state_bits_embedded.shape[0]
        dim_flat = state_bits_embedded.reshape(batch, 4, -1)
        dim_enc = self.dim_encoder(dim_flat)
        pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        pair_features = [self.pair_compare(torch.cat([dim_enc[:,i], dim_enc[:,j]], dim=-1)) for i,j in pairs]
        return self.comparison_out(torch.cat(pair_features, dim=-1))


class PermutationCircuit(nn.Module):
    def __init__(self, example_embed_dim, permutation_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(example_embed_dim, permutation_dim),
            nn.LayerNorm(permutation_dim)  # Added
        )
        self.perm_attention = nn.MultiheadAttention(permutation_dim, 4, batch_first=True, dropout=0.1)
        self.perm_out = nn.Sequential(
            nn.Linear(permutation_dim, permutation_dim), 
            nn.LayerNorm(permutation_dim),  # Added
            nn.GELU(),
            nn.Linear(permutation_dim, output_dim),
            nn.LayerNorm(output_dim)  # Added
        )
        
    def forward(self, support_context):
        proj = self.proj(support_context)
        attended, _ = self.perm_attention(proj, proj, proj)
        return self.perm_out(attended.mean(dim=1))


class ParameterInferenceCircuit(nn.Module):
    def __init__(self, context_dim, parameter_dim, output_dim, n_params=8):
        super().__init__()
        self.n_params = n_params
        self.support_aggregator = nn.Sequential(
            nn.Linear(context_dim, parameter_dim * 2), 
            nn.LayerNorm(parameter_dim * 2),  # Added
            nn.GELU(),
            nn.Linear(parameter_dim * 2, parameter_dim),
            nn.LayerNorm(parameter_dim)  # Added
        )
        self.param_attention = nn.MultiheadAttention(parameter_dim, 4, batch_first=True, dropout=0.1)
        self.param_predictor = nn.Linear(parameter_dim, n_params)
        self.param_embed = nn.Embedding(n_params, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)  # Added
        
    def forward(self, support_context):
        support_proj = self.support_aggregator(support_context)
        attended, _ = self.param_attention(support_proj, support_proj, support_proj)
        pooled = attended.mean(dim=1)
        param_logits = self.param_predictor(pooled)
        param_probs = F.softmax(param_logits / 0.5, dim=-1)  # Temperature scaling
        param_features = torch.matmul(param_probs, self.param_embed.weight)
        return self.out_norm(param_features), param_logits


# =============================================================================
# MAIN MODEL
# =============================================================================

class FluxMindV083(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = config
        
        # Base architecture
        self.bit_embed = nn.Embedding(2, cfg.bit_embed_dim)
        state_input = cfg.state_dim * cfg.bits_per_value * cfg.bit_embed_dim
        
        self.before_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(state_input, cfg.bit_state_embed_dim),
            nn.LayerNorm(cfg.bit_state_embed_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bit_state_embed_dim, cfg.bit_state_embed_dim), nn.GELU()
        )
        self.after_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(state_input, cfg.bit_state_embed_dim),
            nn.LayerNorm(cfg.bit_state_embed_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bit_state_embed_dim, cfg.bit_state_embed_dim), nn.GELU()
        )
        self.transition_net = nn.Sequential(
            nn.Linear(cfg.bit_state_embed_dim * 2, cfg.bit_example_embed_dim),
            nn.LayerNorm(cfg.bit_example_embed_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bit_example_embed_dim, cfg.bit_example_embed_dim)
        )
        self.op_embed = nn.Embedding(cfg.num_operations, cfg.bit_context_dim // 2)
        self.query_proj = nn.Linear(cfg.bit_state_embed_dim, cfg.bit_context_dim)
        self.key_proj = nn.Linear(cfg.bit_example_embed_dim, cfg.bit_context_dim)
        self.value_proj = nn.Linear(cfg.bit_example_embed_dim, cfg.bit_context_dim)
        self.attention = nn.MultiheadAttention(cfg.bit_context_dim, cfg.bit_num_heads, cfg.dropout, batch_first=True)
        
        # Reasoning circuits
        circuit_out = cfg.bit_state_embed_dim // 2
        self.comparison_circuit = ComparisonCircuit(cfg.bit_embed_dim, cfg.bits_per_value, cfg.comparison_dim, circuit_out)
        self.permutation_circuit = PermutationCircuit(cfg.bit_example_embed_dim, cfg.permutation_dim, circuit_out)
        self.parameter_circuit = ParameterInferenceCircuit(cfg.bit_example_embed_dim, cfg.parameter_dim, circuit_out)
        
        # Gate with proper initialization
        self.circuit_gate = nn.Sequential(
            nn.Linear(cfg.bit_example_embed_dim, 64), 
            nn.LayerNorm(64),  # Added
            nn.GELU(), 
            nn.Linear(64, 3), 
            nn.Sigmoid()
        )
        
        # Predictor
        pred_input = cfg.bit_state_embed_dim + cfg.bit_context_dim // 2 + cfg.bit_context_dim + circuit_out * 3
        self.predictor = nn.Sequential(
            nn.Linear(pred_input, cfg.bit_hidden_dim), nn.LayerNorm(cfg.bit_hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bit_hidden_dim, cfg.bit_hidden_dim), nn.LayerNorm(cfg.bit_hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bit_hidden_dim, cfg.bit_hidden_dim // 2), nn.GELU(),
            nn.Linear(cfg.bit_hidden_dim // 2, cfg.state_dim * cfg.bits_per_value)
        )
        self.confidence_head = nn.Sequential(nn.Linear(pred_input, 128), nn.GELU(), nn.Linear(128, 1))
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with smaller values for stability."""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def _embed_bits(self, bits):
        return self.bit_embed(bits.long())
    
    def encode_support(self, bits_before, ops, bits_after):
        batch_size, n_examples = bits_before.shape[:2]
        bb_emb = self._embed_bits(bits_before)
        ba_emb = self._embed_bits(bits_after)
        bb_flat = bb_emb.reshape(batch_size * n_examples, *bb_emb.shape[2:])
        ba_flat = ba_emb.reshape(batch_size * n_examples, *ba_emb.shape[2:])
        before_enc = self.before_encoder(bb_flat)
        after_enc = self.after_encoder(ba_flat)
        trans_emb = self.transition_net(torch.cat([before_enc, after_enc], dim=-1))
        return trans_emb.reshape(batch_size, n_examples, -1)
    
    def forward(self, query_bits, query_op, support_context, support_ops):
        cfg = self.config
        query_emb = self._embed_bits(query_bits)
        state_emb = self.before_encoder(query_emb)
        op_emb = self.op_embed(query_op)
        op_mask = (support_ops != query_op.unsqueeze(1))
        q = self.query_proj(state_emb).unsqueeze(1)
        k, v = self.key_proj(support_context), self.value_proj(support_context)
        context, _ = self.attention(q, k, v, key_padding_mask=op_mask)
        context = context.squeeze(1)
        
        # Reasoning circuits with safety checks
        gates = self.circuit_gate(support_context.mean(dim=1))
        
        comparison_feat = self.comparison_circuit(query_emb)
        perm_feat = self.permutation_circuit(support_context)
        param_feat, _ = self.parameter_circuit(support_context)
        
        # Apply gates
        comparison_feat = comparison_feat * gates[:, 0:1]
        perm_feat = perm_feat * gates[:, 1:2]
        param_feat = param_feat * gates[:, 2:3]
        
        combined = torch.cat([state_emb, op_emb, context, comparison_feat, perm_feat, param_feat], dim=-1)
        bit_logits = self.predictor(combined).view(-1, cfg.state_dim, cfg.bits_per_value)
        confidence = self.confidence_head(combined)
        return bit_logits, confidence


# =============================================================================
# DSL GENERATOR
# =============================================================================

@dataclass
class GeneratedDSL:
    name: str
    family: str
    ops: Dict[int, Callable]
    def execute(self, state, op):
        return self.ops[op](state.copy())

class DSLGenerator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
    
    def _clamp(self, x):
        return max(1, min(15, x))
    
    def _make_arithmetic_dsl(self, dsl_id):
        ops = {}
        for op in range(8):
            dim, const = op % 4, self.rng.randint(-7, 7) or 1
            def make_op(d=dim, c=const):
                def fn(s): s[d] = self._clamp(s[d] + c); return s
                return fn
            ops[op] = make_op()
        return GeneratedDSL(f"Arith_{dsl_id}", "arithmetic", ops)
    
    def _make_bitwise_dsl(self, dsl_id):
        ops = {}
        for op in range(8):
            dim, bit_op, mask = op % 4, self.rng.choice(['xor','and','or']), self.rng.randint(1,15)
            def make_op(d=dim, bo=bit_op, m=mask):
                def fn(s):
                    v = s[d] - 1
                    v = (v ^ m) if bo == 'xor' else ((v & m) if bo == 'and' else (v | m))
                    s[d] = self._clamp(v + 1); return s
                return fn
            ops[op] = make_op()
        return GeneratedDSL(f"Bitwise_{dsl_id}", "bitwise", ops)
    
    def _make_comparison_dsl(self, dsl_id):
        ops = {}
        for op in range(8):
            src, dst, use_max = op % 4, (op + 1) % 4, op >= 4
            def make_op(s=src, d=dst, mx=use_max):
                def fn(st): st[d] = max(st[s], st[d]) if mx else min(st[s], st[d]); return st
                return fn
            ops[op] = make_op()
        return GeneratedDSL(f"Compare_{dsl_id}", "comparison", ops)
    
    def _make_modular_dsl(self, dsl_id):
        ops, mod = {}, self.rng.choice([3,5,7,11,13])
        for op in range(8):
            dim, av = op % 4, self.rng.randint(1, mod-1)
            def make_op(d=dim, a=av, m=mod):
                def fn(s): s[d] = (s[d]-1+a) % m + 1; return s
                return fn
            ops[op] = make_op()
        return GeneratedDSL(f"Modular_{dsl_id}", "modular", ops)
    
    def _make_shift_dsl(self, dsl_id):
        ops = {}
        for op in range(8):
            dim, sl, sa = op % 4, op >= 4, self.rng.randint(1,3)
            def make_op(d=dim, left=sl, amt=sa):
                def fn(s):
                    v = s[d] - 1
                    v = ((v << amt) & 0xF) if left else (v >> amt)
                    s[d] = self._clamp(v + 1); return s
                return fn
            ops[op] = make_op()
        return GeneratedDSL(f"Shift_{dsl_id}", "shift", ops)
    
    def _make_swap_dsl(self, dsl_id):
        ops, pairs = {}, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        self.rng.shuffle(pairs)
        for op in range(8):
            a, b, do_swap = *pairs[op % len(pairs)], op < 4
            def make_op(i=a, j=b, sw=do_swap):
                def fn(s):
                    if sw: s[i], s[j] = s[j], s[i]
                    return s
                return fn
            ops[op] = make_op()
        return GeneratedDSL(f"Swap_{dsl_id}", "swap", ops)
    
    def _make_mixed_dsl(self, dsl_id):
        ops = {}
        for op in range(8):
            dim, op_type = op % 4, op // 4
            if op_type == 0:
                const = self.rng.randint(-3, 3) or 1
                def make_op(d=dim, c=const):
                    def fn(s): s[d] = self._clamp(s[d] + c); return s
                    return fn
            else:
                mask = self.rng.randint(1, 15)
                def make_op(d=dim, m=mask):
                    def fn(s): s[d] = self._clamp((s[d]-1) ^ m + 1); return s
                    return fn
            ops[op] = make_op()
        return GeneratedDSL(f"Mixed_{dsl_id}", "mixed", ops)
    
    def generate_train_test_dsls(self, n_train_per_family=12, n_test_per_family=3):
        families = [self._make_arithmetic_dsl, self._make_bitwise_dsl, self._make_comparison_dsl,
                    self._make_modular_dsl, self._make_shift_dsl, self._make_swap_dsl, self._make_mixed_dsl]
        train_dsls, test_dsls = [], []
        dsl_id = 0
        for fn in families:
            for i in range(n_train_per_family):
                train_dsls.append(fn(dsl_id))
                dsl_id += 1
            for i in range(n_test_per_family):
                test_dsls.append(fn(dsl_id + 1000))
                dsl_id += 1
        self.rng.shuffle(train_dsls)
        self.rng.shuffle(test_dsls)
        return train_dsls, test_dsls


# =============================================================================
# UTILITIES
# =============================================================================

def state_to_bits(state):
    bits = np.zeros((4, 4), dtype=np.float32)
    for di in range(4):
        for bi in range(4):
            bits[di, bi] = ((state[di] - 1) >> bi) & 1
    return bits

def bits_to_state(bits):
    return [sum(int(bits[di, bi]) << bi for bi in range(4)) + 1 for di in range(4)]

def random_state(rng):
    return [int(rng.randint(1, 16)) for _ in range(4)]

def generate_examples(dsl, n, rng):
    examples = []
    for op in range(8):
        for _ in range(n // 8 + 1):
            state = random_state(rng)
            try:
                ns = dsl.execute(state, op)
                if all(1 <= v <= 15 for v in ns):
                    examples.append((state, op, ns))
            except: pass
    rng.shuffle(examples)
    return examples[:n]


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, train_dsls, optimizer, device, rng, batch_size=32):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for _ in range(100):
        dsl = rng.choice(train_dsls)
        examples = generate_examples(dsl, 64, rng)
        if len(examples) < 33: continue
        
        support, query_ex = examples[:32], examples[32]
        s_before = np.stack([state_to_bits(ex[0]) for ex in support])
        s_after = np.stack([state_to_bits(ex[2]) for ex in support])
        
        query_bits = torch.tensor(state_to_bits(query_ex[0]), dtype=torch.float32, device=device).unsqueeze(0)
        query_op = torch.tensor([query_ex[1]], dtype=torch.long, device=device)
        target_bits = torch.tensor(state_to_bits(query_ex[2]), dtype=torch.float32, device=device).unsqueeze(0)
        support_before = torch.tensor(s_before, dtype=torch.float32, device=device).unsqueeze(0)
        support_ops = torch.tensor([[ex[1] for ex in support]], dtype=torch.long, device=device)
        support_after = torch.tensor(s_after, dtype=torch.float32, device=device).unsqueeze(0)
        
        optimizer.zero_grad()
        support_enc = model.encode_support(support_before, support_ops, support_after)
        bit_logits, _ = model(query_bits, query_op, support_enc, support_ops)
        
        loss = F.binary_cross_entropy_with_logits(bit_logits, target_bits)
        
        # Check for NaN
        if torch.isnan(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Stricter clipping
        optimizer.step()
        
        pred_bits = (torch.sigmoid(bit_logits) > 0.5).float()
        correct = (pred_bits == target_bits).all().item()
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += 1
    
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def evaluate(model, test_dsls, device, n_episodes=100):
    model.eval()
    rng = np.random.RandomState(999)
    family_results = defaultdict(list)
    
    for dsl in test_dsls:
        correct, total = 0, 0
        for _ in range(n_episodes):
            examples = generate_examples(dsl, 64, rng)
            if len(examples) < 33: continue
            support, query_ex = examples[:32], examples[32]
            s_before = np.stack([state_to_bits(ex[0]) for ex in support])
            s_after = np.stack([state_to_bits(ex[2]) for ex in support])
            
            query_bits = torch.tensor(state_to_bits(query_ex[0]), dtype=torch.float32, device=device).unsqueeze(0)
            query_op = torch.tensor([query_ex[1]], dtype=torch.long, device=device)
            support_before = torch.tensor(s_before, dtype=torch.float32, device=device).unsqueeze(0)
            support_ops = torch.tensor([[ex[1] for ex in support]], dtype=torch.long, device=device)
            support_after = torch.tensor(s_after, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                support_enc = model.encode_support(support_before, support_ops, support_after)
                bit_logits, _ = model(query_bits, query_op, support_enc, support_ops)
                pred_bits = (torch.sigmoid(bit_logits) > 0.5).float()
            
            pred_state = bits_to_state(pred_bits[0].cpu().numpy())
            if pred_state == query_ex[2]: correct += 1
            total += 1
        
        acc = correct / max(total, 1)
        family_results[dsl.family].append(acc)
    
    return family_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-4)  # Lower LR
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--n_train_per_family', type=int, default=12)
    parser.add_argument('--n_test_per_family', type=int, default=3)
    args = parser.parse_args()
    
    print("=" * 70)
    print("FluxMind v0.83 Training - Reasoning Circuits (FIXED)")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    
    # Generate DSLs
    gen = DSLGenerator(seed=42)
    train_dsls, test_dsls = gen.generate_train_test_dsls(args.n_train_per_family, args.n_test_per_family)
    print(f"\nTraining DSLs: {len(train_dsls)} ({args.n_train_per_family} per family)")
    print(f"Test DSLs: {len(test_dsls)} ({args.n_test_per_family} per family, BALANCED)")
    
    # Create model
    config = FluxMindV083Config()
    model = FluxMindV083(config).to(args.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {params:,}")
    
    # Check for NaN in initial params
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            print(f"WARNING: NaN in {name}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
    
    rng = np.random.RandomState(42)
    best_acc, best_epoch = 0, 0
    
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_epoch(model, train_dsls, optimizer, args.device, rng)
        scheduler.step()
        
        if epoch % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Train Acc: {train_acc:.1%} | LR: {lr:.2e}")
        
        if epoch % args.eval_interval == 0:
            family_results = evaluate(model, test_dsls, args.device, n_episodes=50)
            
            print("\n" + "-" * 50)
            print(f"EVALUATION @ Epoch {epoch}")
            print("-" * 50)
            
            overall_accs = []
            for fam in sorted(family_results.keys()):
                accs = family_results[fam]
                mean_acc = np.mean(accs) * 100
                overall_accs.extend(accs)
                print(f"  {fam:12s}: {mean_acc:5.1f}%")
            
            overall = np.mean(overall_accs) * 100
            print(f"  {'OVERALL':12s}: {overall:5.1f}%")
            
            if overall > best_acc:
                best_acc = overall
                best_epoch = epoch
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'mean_test_acc': overall / 100,
                    'family_results': {f: np.mean(a) for f, a in family_results.items()}
                }, 'fluxmind_v083_best.pt')
                print(f"  ✓ New best! Saved checkpoint.")
            
            print("-" * 50 + "\n")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Best accuracy: {best_acc:.1f}% @ epoch {best_epoch}")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (100 episodes)")
    print("=" * 70)
    
    ckpt = torch.load('fluxmind_v083_best.pt', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    family_results = evaluate(model, test_dsls, args.device, n_episodes=100)
    
    print("\nPer-family results:")
    for fam in sorted(family_results.keys()):
        accs = family_results[fam]
        print(f"  {fam:12s}: {np.mean(accs)*100:5.1f}% ± {np.std(accs)*100:4.1f}%")
    
    overall = np.mean([a for accs in family_results.values() for a in accs]) * 100
    print(f"\n  OVERALL: {overall:.1f}%")
    
    # Comparison with v0.82 baseline
    v082_baseline = {'arithmetic': 87.7, 'shift': 84.3, 'bitwise': 67.3, 'mixed': 67.7,
                     'swap': 63.0, 'modular': 59.3, 'comparison': 44.7}
    
    print("\n" + "=" * 70)
    print("COMPARISON vs v0.82 BASELINE (67.7%)")
    print("=" * 70)
    for fam in sorted(family_results.keys()):
        v082 = v082_baseline.get(fam, 0)
        v083 = np.mean(family_results[fam]) * 100
        delta = v083 - v082
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {fam:12s}: {v082:5.1f}% → {v083:5.1f}% ({delta:+5.1f}% {arrow})")
    
    print(f"\n  OVERALL: 67.7% → {overall:.1f}%")


if __name__ == '__main__':
    main()
