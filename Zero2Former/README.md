# Zero2Former: Building Transformers from Scratch 🚀

Welcome to Zero2Former, a hands-on educational project designed to understand transformer networks by implementing them from scratch! This repository contains step-by-step implementations and detailed explanations of transformer architecture components.

## 🎯 Project Goals

- Provide clear, well-documented implementations of transformer components
- Break down complex concepts into digestible pieces
- Offer interactive Colab notebooks for hands-on learning
- Help developers understand the inner workings of transformer models

## 📚 Learning Path

1. **Fundamentals**
   - Attention mechanism basics
   - Self-attention implementation
   - Multi-head attention

2. **Core Components**
   - Positional encodings
   - Feed-forward networks
   - Layer normalization

3. **Full Architecture**
   - Encoder stack
   - Decoder stack
   - Complete transformer model

## 📂 Repository Structure

```
Zero2Former/
├── .github/          # GitHub Actions workflows
├── notebooks/        # Interactive Colab notebooks
├── examples/         # Usage examples
├── tests/            # Unit tests
├── zero2former/      # Main package source
├── pyproject.toml    # Project configuration
└── .pre-commit-config.yaml  # Code quality tools config
```

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Zero2Former.git
cd Zero2Former
```

2. Install dependencies using Poetry:
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

### Development

1. Run tests:
```bash
pytest
```

2. Check code quality:
```bash
ruff check .
mypy zero2former tests
```

3. Auto-format code:
```bash
ruff format .
```

1. Clone the repository
2. Install dependencies (requirements will be provided)
3. Follow the notebooks in order
4. Experiment with the implementations

## 📖 Resources

- Implementation follows the original "Attention Is All You Need" paper
- Additional references and explanations will be provided in each notebook

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add explanations
- Share your implementations

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

⭐ If you find this project helpful, please consider giving it a star!
