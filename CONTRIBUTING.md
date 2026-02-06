# Contributing to YOLOv8 CSV-to-YOLO Pipeline

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

---

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive experience for everyone.

---

## How to Contribute

### Types of Contributions

We welcome the following types of contributions:

- **Bug fixes** — Fix issues in the conversion pipeline, training scripts, or utilities
- **New features** — Add support for new CSV formats, export targets, or visualization tools
- **Documentation** — Improve README, guides, docstrings, or add tutorials
- **Examples** — Add new example scripts or sample datasets
- **Tests** — Add or improve unit tests and integration tests
- **Performance** — Optimize conversion speed, memory usage, or training efficiency

---

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/yolov8-csv-to-yolo.git
cd yolov8-csv-to-yolo
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Coding Standards

### Style Guide

- Follow **PEP 8** for Python code
- Use **Black** for code formatting (line length: 100)
- Use **Ruff** for linting
- Write **docstrings** for all public functions and classes (Google style)

### Formatting Commands

```bash
# Format code
make format
# or
black src/ scripts/ --line-length 100

# Lint code
make lint
# or
ruff check src/ scripts/
```

### File Organization

| Directory | Purpose |
|-----------|---------|
| `src/yolov8_pipeline/` | Core library code (parsing, conversion, visualization) |
| `scripts/` | Standalone CLI scripts |
| `notebooks/` | Jupyter/Colab notebooks |
| `configs/` | YAML configuration files |
| `docs/` | Documentation files |
| `examples/` | Example scripts and sample data |

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

---

## Submitting Changes

### Pull Request Process

1. **Ensure your code passes** all linting and formatting checks
2. **Update documentation** if your changes affect the public API or usage
3. **Add examples** if you introduce new functionality
4. **Write a clear PR description** explaining what changed and why

### PR Title Format

Use a descriptive title with a prefix:

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation changes
- `refactor:` — Code refactoring
- `test:` — Adding or updating tests
- `chore:` — Maintenance tasks

**Examples:**

```
feat: add support for COCO JSON annotation format
fix: handle images with zero-area bounding boxes
docs: add troubleshooting section for GPU memory errors
```

### Commit Messages

Write clear, concise commit messages:

```
feat: add Format D support for COCO JSON annotations

- Parse COCO JSON structure with categories and annotations
- Map COCO category IDs to contiguous class indices
- Add unit tests for COCO format detection
```

---

## Reporting Issues

When reporting a bug, please include:

1. **Description** — Clear description of the issue
2. **Steps to reproduce** — Minimal steps to trigger the bug
3. **Expected behavior** — What you expected to happen
4. **Actual behavior** — What actually happened
5. **Environment** — Python version, OS, GPU, package versions
6. **Error logs** — Full traceback or error messages
7. **Sample data** — Minimal CSV/image sample that reproduces the issue (if applicable)

### Issue Template

```markdown
**Description:**
Brief description of the issue.

**Steps to reproduce:**
1. Step one
2. Step two
3. ...

**Expected behavior:**
What should happen.

**Actual behavior:**
What happens instead.

**Environment:**
- OS: Ubuntu 22.04
- Python: 3.10
- Ultralytics: 8.2.103
- GPU: NVIDIA T4

**Error log:**
```
Paste error traceback here
```
```

---

## Feature Requests

We welcome feature requests! Please open an issue with:

1. **Use case** — Describe the problem you are trying to solve
2. **Proposed solution** — Your idea for how to solve it
3. **Alternatives** — Any alternative approaches you considered

---

## Questions?

If you have questions about contributing, feel free to open a discussion or issue. We are happy to help!
