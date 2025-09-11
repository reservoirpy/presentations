# ICANN 2025 - Tutorial on ReservoirPy

This is a [Tutorial](https://sites.google.com/view/reservoircomputing2025/tutorial) presented at the [3rd Reservoir Computing workshop](https://sites.google.com/view/reservoircomputing2025/workshop-program) held at [ICANN 2025](https://e-nns.org/icann2025/rc/).

The aim is to:
- make participant familiar with the key concepts of Reservoir Computing;
- make participant familiar with [ReservoirPy library](https://github.com/reservoirpy/reservoirpy) the most popular reservoir library in Python;
- give you quick insights on ReservoirPy key features;
- show you how to optimize dynamically your hyperparameter search
- propose a quick hackathon and see which participant manages to reach the best performances while testing other reservoir nodes:
  - [ES2N node](https://github.com/reservoirpy/reservoirpy/tree/master/examples/Edge%20of%20Stability%20Echo%20State%20Network)
  - [Intrinsic Plasticity](https://github.com/reservoirpy/reservoirpy/tree/master/examples/Improving%20reservoirs%20using%20Intrinsic%20Plasticity)
  - [Next Generation Reservoir Computing](https://github.com/reservoirpy/reservoirpy/tree/master/examples/Next%20Generation%20Reservoir%20Computing)

## Follow these steps

1. Install ReservoirPy on your machine and within terminal: 
```bash
pip install reservoirpy[hyper]
``` 

2. Start the [Jupyter notebook here](https://github.com/reservoirpy/presentations/blob/main/ICANN-2025/01_Introduction.ipynb).

3. Play with [Jupyter notebook here](https://github.com/reservoirpy/presentations/blob/main/ICANN-2025/02_Hackathon.ipynb). 

Note that if you already installed ReservoirPy more than one month ago, you would need to upgrade it to the new main version, because this tutorial won't work with previous main 0.3 version. Indeed, we now move to version 0.4 with several main new features (see the last [main release of ReservoirPy0.4](https://github.com/reservoirpy/reservoirpy/releases/tag/v0.4.0)).