# Gold Price Prediction

This package implements a simple machine learning pipeline in Julia to predict gold prices using financial indicators (SPX, USO, SLV, GLD).  
It includes functions for dataset handling, train-test splitting, model training, evaluation metrics, and visualization.

---

## 🚀 Getting Started

### 1. Activate and instantiate the environment
From the project root folder (where `Project.toml` lives), open Julia and run:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Pluto
Pluto.run()
```
Then Pluto environment will open and select gold_price_pred.jl
