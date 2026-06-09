# Architecture (Dynamic PyTorch Model Builder)
## Overview

This module provides a simple, flexible neural network framework built on top of torch.nn.Module. It supports:

* Dynamic model construction
* Sequential layer stacking
* Residual connections via custom blocks
* Easy experimentation with CNN architectures

It consists of two core components:

* ResidualBlock
* Architecture

## ResidualBlock
### Purpose

Implements a standard residual learning block where the input is added to the transformed output:
``` math
y=H(x)+x
```
This helps improve gradient flow and stabilizes training in deeper networks.

### Implementation Idea
Internally uses `nn.Sequential` to define the transformation function `H(x)`
Adds skip connection from input to output

<b>Behavior</b>

Given input x:
``` text
x → layers (H) → output
output + x → final output
```
Example
``` python
block = ResidualBlock(
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1)
)
```

This computes:

```
output = conv2(relu(conv1(x))) + x
```

## Architecture
### Purpose

A flexible container for building neural networks dynamically. It allows stacking:

* Standard layers (Conv2d, ReLU, Linear, etc.)
* Residual blocks
* Custom modules

All components are executed sequentially.

### Core Design
* Inherits from torch.nn.Module
* Stores all components in nn.ModuleList
* Executes forward pass sequentially

### Methods
1.  ```python
    add(module)
    ```
    Adds a single layer or block to the model.
    

2.  ``` python
    model.add(nn.Conv2d(3, 64, 3, padding=1))
    ```
    Returns:
    - self (enables chaining)
    - residual(*layers)


3. Creates and adds a ResidualBlock.

    ``` python
    model.residual(
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1)
    )

    Equivalent to:
    ```
    x → H(x) + x
    ```

### Forward Pass

Each component is applied sequentially:
``` python
def forward(self, x):
    for block in self.blocks:
        x = block(x)
    return x
```
Example Usage
``` python
model = (
    Architecture()
    .add(nn.Conv2d(3, 64, 3, padding=1))
    .add(nn.ReLU())

    .residual(
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1)
    )

    .add(nn.MaxPool2d(2))
)
```

### Key Characteristics
* Fully dynamic architecture building
* Supports arbitrary stacking of layers and residual blocks
* Compatible with PyTorch optimizers and autograd
* Suitable for research and experimentation

### Notes
* Residual connections always operate on the input of the block, not the full network input.
* All layers must be valid torch.nn.Module objects.
* Channel/shape compatibility must be ensured manually when using residual blocks.