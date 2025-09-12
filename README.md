# Gold Price Prediction

This package implements a simple machine learning pipeline in Julia to predict gold prices using financial indicators (SPX, USO, SLV, GLD).  
It includes functions for dataset handling, data cleaning and preprocessing, train-test splitting, model training, evaluation metrics, and visualization.

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

#  Utils: 
## r2_score(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real}) -> Float64

Compute the R² score (coefficient of determination) between the true values (`y_true`)
and predicted values (`y_pred`).  
The R² score measures how well the predictions approximate the actual data, with `1.0`
being a perfect fit.

Arguments
- `y_true` : Vector of true target values.
- `y_pred` : Vector of predicted values.

Returns
- `Float64` : The R² score.

Examples
```julia
julia> y_true = [3.0, -0.5, 2.0, 7.0];
julia> y_pred = [2.5, 0.0, 2.0, 8.0];
julia> r2_score(y_true, y_pred)
0.9486081370449679

julia> r2_score([1,2,3,4], [1.1,1.9,3.2,3.8])
0.9899999999999999
```
## train_test_split(X::DataFrame, y::AbstractVector{<:Real}; frac::Float64=0.8, rng=Random.default_rng())

Split a dataset into training and testing sets.

By default, `frac=0.8` keeps 80% of the data for training and 20% for testing.
The split is randomized but can be made reproducible by passing a random number generator via `rng`.

Arguments
- `X` : DataFrame of features.
- `y` : Vector of target values.
- `frac` (keyword) : Fraction of the data to include in the training set (default = 0.8).
- `rng` (keyword) : Random number generator (default = `Random.default_rng()`).

Returns
A tuple `(X_train, X_test, y_train, y_test)`.

Examples
```julia
julia> using DataFrames, Random

julia> X = DataFrame(a=1:10, b=11:20);
julia> y = collect(21:30);

julia> X_train, X_test, y_train, y_test = train_test_split(X, y; frac=0.7, rng=MersenneTwister(42));

julia> size(X_train, 1), size(X_test, 1)
(7, 3)

julia> length(y_train), length(y_test)
(7, 3)
```

## train_models(X_train::DataFrame, y_train::Vector{<:Real}, X_test::DataFrame) -> Tuple{Vector{Float64}, Vector{Float64}}

Train two regression models (Linear Regression and Random Forest) using MLJ, 
fit them on the training set, and return predictions on the test set.

Arguments
- `X_train` : DataFrame of training features.
- `y_train` : Vector of training target values.
- `X_test`  : DataFrame of testing features.

Returns
- `(y_pred_lin, y_pred_rf)` : A tuple of predicted values from:
  1. Linear Regression (`y_pred_lin`)
  2. Random Forest Regression (`y_pred_rf`)

Notes
- This function automatically handles conversion from `DataFrame` to `Matrix` 
  and back to ensure compatibility with MLJ.  
- It raises an error if missing values are present.

Examples
```julia
julia> using DataFrames, MLJ, MLJLinearModels, DecisionTree

julia> X_train = DataFrame(x1 = rand(100), x2 = rand(100));
julia> y_train = 3 .* X_train.x1 .+ 2 .* X_train.x2 .+ rand(100);
julia> X_test = DataFrame(x1 = rand(10), x2 = rand(10));

julia> y_pred_lin, y_pred_rf = train_models(X_train, y_train, X_test);

julia> length(y_pred_lin), length(y_pred_rf)
(10, 10)

julia> typeof(y_pred_lin), typeof(y_pred_rf)
(Vector{Float64}, Vector{Float64})
```
