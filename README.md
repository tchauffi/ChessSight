# ChessSight

ChessSight is a computer vision project designed to identify and classify chess pieces on a chessboard using image processing techniques. This repository contains the code and resources needed to detect chess pieces in real-time, providing a foundation for applications in automated chess analysis and gameplay.

## Table of Contents
- [ChessSight](#chesssight)
  - [Development roadmap](#development-roadmap)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [For Users](#for-users)
    - [For Developers](#for-developers)
  - [Development Commands](#development-commands)
  - [Code Quality](#code-quality)

## Development roadmap

The goal of the projet is to create a chessboard detection and piece classification system that can be used in various applications, such as:
- **Chess training**: Helping players improve their skills by analyzing their games.
- **Chess analysis**: Providing insights into game strategies and tactics.
- **Chess streaming**: Enhancing the viewing experience for online chess games by overlaying real-time analysis and commentary.

Current roadmap is :

* [ ] Chessboard detection: Implementing a robust algorithm to detect chessboards in images and videos.
* [ ] Piece classification: Developing a machine learning model to classify chess pieces based on their appearance.
* [ ] Real-time processing: Optimizing the system for real-time performance, allowing for live analysis of chess games.

Nice to have features include:
* [ ] 3D chessboard detection: Enhancing the chessboard detection algorithm to work with 3D chessboards, allowing for more accurate piece classification.
* [ ] Multi-camera support: Enabling the system to work with multiple cameras, providing a more comprehensive view of the chessboard.
* [ ] Rest API: Creating a REST API using FastAPI to allow for easy integration with other applications and services.




## Prerequisites

Before installing, make sure you have:
- Python 3.12 or higher
- Poetry (Python package manager)
- Make

On macOS, you can install these using Homebrew:
```bash
brew install python@3.12 poetry make
```

## Installation

### For Users

To install the package for use:

```bash
pip install .
```

### For Developers

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ChessSight.git
cd ChessSight
```

2. Install dependencies and set up pre-commit hooks:
```bash
make install
```



## Development Commands

The project uses a Makefile to simplify common development tasks:

- `make help` - Show available commands
- `make install` - Install dependencies and pre-commit hooks
- `make format` - Format code using black and ruff
- `make lint` - Run all linters (ruff, mypy)
- `make test` - Run tests
- `make pre-commit` - Run pre-commit hooks manually
- `make clean` - Remove cache and build files


## Code Quality

This project uses several tools to maintain code quality:
- Black for code formatting
- Ruff for linting
- MyPy for type checking
- Pre-commit hooks for automated checks

These are automatically run on commit, but you can run them manually:
```bash
make pre-commit
```
