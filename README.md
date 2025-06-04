# MOTOP: Making of the Optimus Prime 🤖

## GPT-2 Implementation Following Andrej Karpathy's Approach

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure) 
- [Implementation Details](#implementation-details)
- [Getting Started](#getting-started)
- [Training Process](#training-process)
- [Model Architecture](#model-architecture)
- [File Descriptions](#file-descriptions)
- [Performance Optimizations](#performance-optimizations)
- [Usage Examples](#usage-examples)
- [Technical Deep Dive](#technical-deep-dive)
- [Contributing](#contributing)
- [References](#references)

---

## 🎯 Overview

**MOTOP** is a comprehensive, educational implementation of GPT-2 (Generative Pre-trained Transformer 2) built from scratch in PyTorch, following the methodological approach pioneered by **Andrej Karpathy** in his renowned "build-nanogpt" tutorial series. This repository serves as both a learning resource and a fully functional GPT-2 implementation that closely mirrors the original OpenAI paper specifications.

### Key Highlights

- ✅ **Complete GPT-2 Architecture**: Full implementation of the transformer decoder architecture
- ✅ **Multiple Implementation Stages**: Progressive complexity from basic to fully optimized versions
- ✅ **Educational Focus**: Extensively commented code with clear explanations
- ✅ **Optimization Techniques**: Gradient accumulation, mixed precision, torch.compile integration
- ✅ **Training Infrastructure**: Comprehensive training loops with validation and checkpointing
- ✅ **Text Generation**: Advanced sampling techniques with temperature control
- ✅ **Data Processing**: Efficient tokenization and data loading utilities
- ✅ **Pre-trained Integration**: Compatibility with HuggingFace pre-trained models

---

**Total Codebase**: 3,087 lines across 10 Python files

---

## 🛠 Implementation Details

### GPT-2 Model Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model Size** | 124M parameters | GPT-2 small configuration |
| **Layers** | 12 | Transformer decoder blocks |
| **Attention Heads** | 12 | Multi-head attention |
| **Embedding Dimension** | 768 | Hidden state size |
| **Context Length** | 1024 | Maximum sequence length |
| **Vocabulary Size** | 50,257 | BPE tokens + special tokens |

### Technical Implementation

- **Architecture**: Decoder-only transformer with causal self-attention
- **Tokenization**: TikToken BPE (Byte Pair Encoding) tokenizer
- **Optimization**: AdamW with cosine learning rate scheduling
- **Precision**: Mixed precision training (bfloat16)
- **Activation**: GELU (Gaussian Error Linear Unit)
- **Normalization**: Layer normalization with pre-normalization pattern

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python packages
pip install torch>=2.0.0
pip install tiktoken
pip install transformers
pip install numpy
pip install tqdm
```

### Quick Start

1. **Clone the Repository**
```bash
git clone https://github.com/Ujjwal238/MOTOP.git
cd MOTOP
```

2. **Basic Model Testing**
```bash
python model_no_optimization.py
```

3. **Full Training Run**
```bash
python completed_model.py
```

4. **Text Generation**
```bash
python text_generation.py
```

---

## 🏋️ Training Process

### Training Configuration

The implementation uses gradient accumulation to simulate large batch sizes on limited GPU memory:

```python
# Training hyperparameters (from GPT-2 paper)
total_batch_size = 524288    # ~0.5M tokens
micro_batch_size = 2         # Fits in 6GB GPU
sequence_length = 1024       # Context window
grad_accum_steps = 256       # Simulates large batch
max_lr = 4e-4               # Peak learning rate
weight_decay = 0.1          # Regularization
```

### Learning Rate Schedule

- **Warmup**: Linear increase for 10 steps
- **Cosine Decay**: Smooth decrease following cosine curve
- **Min LR**: 10% of maximum learning rate

### Memory Optimizations

1. **Gradient Accumulation**: Simulate large batches with limited memory
2. **Mixed Precision**: bfloat16 for 2x speedup with minimal accuracy loss
3. **torch.compile**: JIT compilation for kernel fusion and optimization
4. **Flash Attention**: Memory-efficient attention computation

---

## 🧠 Model Architecture

The GPT-2 implementation follows the transformer decoder architecture with several key components:

### Core Components

1. **Token & Position Embeddings**
   - Learnable token embeddings (50,257 vocab)
   - Learnable positional embeddings (1024 positions)
   - Embedding sum for input representation

2. **Transformer Blocks** (12 layers)
   - Pre-normalization with LayerNorm
   - Causal self-attention with masking
   - Feed-forward network (4x expansion)
   - Residual connections

3. **Output Head**
   - Final layer normalization
   - Linear projection to vocabulary
   - Weight sharing with input embeddings

### Attention Mechanism

```python
# Causal self-attention implementation
def forward(self, x):
    B, T, C = x.size()
    
    # Compute Q, K, V
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    
    # Reshape for multi-head attention
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    
    # Flash attention with causal masking
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # Reassemble and project
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.c_proj(y)
```

---

## 📋 File Descriptions

### Core Implementation Files

#### `completed_model.py` (414 lines)
**Most Advanced Implementation**
- Complete training loop with all optimizations
- Gradient accumulation for memory efficiency
- Cosine learning rate scheduling with warmup
- Mixed precision training (bfloat16)
- AdamW optimizer with weight decay
- Gradient clipping for stability

#### `model.py` (464 lines)  
**Production-Ready Implementation**
- HuggingFace integration for pre-trained weights
- Configurable optimizer with parameter grouping
- Weight initialization following OpenAI specifications
- Support for loading GPT-2 checkpoints
- Comprehensive model configuration

#### `train_draft_1.py` (555 lines)
**Training with Validation**
- Validation loss tracking
- Model checkpointing system
- Periodic text sampling during training
- Logging infrastructure
- Early stopping capabilities

#### `text_generation.py` (586 lines)
**Text Generation Engine**
- Temperature-controlled sampling
- Top-k and top-p sampling methods
- Batch text generation
- Model evaluation metrics
- Interactive generation interface

### Data Processing Files

#### `get_data.py` (86 lines)
**Data Preprocessing Pipeline**
- Multi-processing tokenization
- Data sharding for large datasets
- Memory-efficient token processing
- Progress tracking with tqdm
- Numpy array optimization

#### `input_embeddings.py` (117 lines)
**Embedding Utilities**
- Custom PyTorch Dataset implementation
- Sliding window data chunking
- Token and position embedding examples
- DataLoader configuration
- Batch processing utilities

#### `target.py` (102 lines)
**Tokenization Examples**
- BPE tokenization demonstrations
- Context window examples
- Token-to-text conversion utilities
- Educational tokenization walkthroughs

### Development Versions

#### `model_no_optimization.py` (224 lines)
**Basic Implementation**
- Minimal GPT-2 architecture
- Simple forward pass
- Basic training loop
- No optimizations (educational baseline)

#### `non_algo_optimized_model.py` (303 lines)  
**Intermediate Implementation**
- Weight initialization improvements
- Basic gradient accumulation
- Simple optimizer configuration
- Performance timing utilities

#### `loaded_parameter_model.py` (236 lines)
**Parameter Loading Utilities**
- Pre-trained model loading
- Parameter compatibility checks
- Weight transfer utilities
- Model state management

---

## ⚡ Performance Optimizations

### Speed Optimizations

1. **torch.compile** (8x speedup)
   ```python
   model = torch.compile(model)  # JIT compilation
   ```

2. **Mixed Precision** (2x speedup)
   ```python
   with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
       logits, loss = model(x, y)
   ```

3. **Flash Attention** (Memory efficient)
   ```python
   y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
   ```

4. **Optimized Vocabulary Size**
   ```python
   # Rounded to nearest nice number for better GPU utilization
   vocab_size = 50304  # Instead of 50257
   ```

### Memory Optimizations

- **Gradient Accumulation**: Handle large effective batch sizes
- **Parameter Grouping**: Separate weight decay for different parameter types
- **Efficient Data Loading**: Minimize memory footprint during training

---

## 💡 Usage Examples

### Training from Scratch

```python
# Load model and data
model = GPT(GPTConfig(vocab_size=50304))
train_loader = DataLoaderLite(B=2, T=1024)

# Configure optimizer
optimizer = model.configure_optimizers(
    weight_decay=0.1, 
    learning_rate=4e-4, 
    device='cuda'
)

# Training loop
for step in range(max_steps):
    # Gradient accumulation loop
    for micro_step in range(grad_accum_steps):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
```

### Text Generation

```python
# Initialize model for generation
model = GPT(GPTConfig())
model.eval()

# Generate text
prompt = "The future of AI is"
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

# Sampling loop
with torch.no_grad():
    for _ in range(50):  # Generate 50 tokens
        logits, _ = model(tokens)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

generated_text = enc.decode(tokens[0].tolist())
```

### Loading Pre-trained Weights

```python
# Load from HuggingFace
model = GPT.from_pretrained('gpt2')

# Or load custom checkpoint
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model'])
```

---

## 🔬 Technical Deep Dive

### Key Implementation Decisions

1. **Pre-normalization Pattern**
   - LayerNorm applied before attention and MLP (not after)
   - Improves training stability and convergence

2. **Weight Sharing**
   - Input embedding weights shared with output layer
   - Reduces parameters and improves performance

3. **GELU Activation**
   - Smooth, probabilistic activation function
   - Better gradient flow compared to ReLU

4. **Causal Attention Masking**
   - Prevents attention to future tokens
   - Essential for autoregressive generation

### Training Stability Features

- **Gradient Clipping**: Prevents exploding gradients
- **Weight Initialization**: Proper parameter initialization
- **Learning Rate Warmup**: Gradual learning rate increase
- **Residual Scaling**: Scaled initialization for deeper networks

---

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/Ujjwal238/MOTOP.git
cd MOTOP

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

---

## 📚 References

### Core Papers

1. **[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** - Radford et al., 2019 (GPT-2 Paper)

2. **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** - Vaswani et al., 2017 (Transformer Architecture)

3. **[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)** - Radford et al., 2018 (GPT-1)

### Educational Resources

4. **[Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)** - Comprehensive deep learning course

5. **[Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)** - Karpathy's GPT-2 implementation video

6. **[The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)** - Visual explanation by Jay Alammar

### Technical References

7. **[PyTorch Documentation](https://pytorch.org/docs/)** - Framework documentation
8. **[HuggingFace Transformers](https://huggingface.co/docs/transformers/)** - Model hub and utilities
9. **[TikToken Documentation](https://github.com/openai/tiktoken)** - OpenAI's tokenizer

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Andrej Karpathy** for the exceptional educational content and methodological approach
- **OpenAI** for the original GPT-2 research and implementation
- **PyTorch Team** for the excellent deep learning framework
- **HuggingFace** for transformer utilities and pre-trained models

---

## 📞 Contact

For questions, suggestions, or discussions about this implementation:

- **Repository**: [MOTOP GitHub](https://github.com/Ujjwal238/MOTOP)
- **Issues**: [GitHub Issues](https://github.com/Ujjwal238/MOTOP/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ujjwal238/MOTOP/discussions)

---

*"Making language models accessible through education and open implementation."*
