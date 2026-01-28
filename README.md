# AutoUQ

## Installation

### Prereqiuisites

- [uv](https://github.com/astral-sh/uv): running scripts; managing virtual environments

### Development
For development, install with [`uv`](https://github.com/astral-sh/uv):
```bash
uv sync --extra dev
```

If contributing to the codebase, you can run 
```bash 
 pre-commit install 
 ```
This will setup the pre-commit checks so any pushed commits will pass the CI. 