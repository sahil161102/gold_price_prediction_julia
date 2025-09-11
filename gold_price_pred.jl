### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ a60f1ce0-892c-11f0-2805-ed8eb3b20852
begin
    import Pkg
    Pkg.activate(".")
    Pkg.instantiate()
end

# ╔═╡ a7d4bbc4-09b1-432f-957a-f0dff3e56661
begin
	include("src/utils.jl")
    using .GoldPriceUtils
    using CSV, DataFrames, Dates, StatsBase, Plots
end

# ╔═╡ b09606bf-4fc8-47ca-bf13-7b89ca0e1391
md"""
# Gold Price Prediction with Julia 
"""

# ╔═╡ b617fd78-bd35-47c1-b0ae-c4f88e21d524
begin
    # 1) Load data
    data_path = "gold_price_data.csv"
    if !isfile(data_path)
        @warn "File $(data_path) not found. Please put your dataset in the working directory."
    end
    df = CSV.read(data_path, DataFrame)
    first(df, 5)
end

# ╔═╡ 08ec388f-ceae-4269-a55b-21b58b1c7e7c
begin
    using Statistics

    # Function to normalize/cap outliers
    function outlier_removal(col::AbstractVector)
        lower_limit = quantile(skipmissing(col), 0.05)
        upper_limit = quantile(skipmissing(col), 0.95)
        return [x < lower_limit ? lower_limit : x > upper_limit ? upper_limit : x for x in col]
    end

    # Columns to normalize (excluding Date)
    cols_to_normalize = ["SPX", "GLD", "USO", "EUR/USD"]

    # Apply outlier removal to each column in place
    for col in cols_to_normalize
        df[!, col] = outlier_removal(df[!, col])
    end

    # first(df, 5)  # preview
end

# ╔═╡ 97467d9f-3822-4fd3-9f36-a631b0adf8bb
begin
	using Random
# --- Prepare data ---
    X_ = select(df, Not(["EUR/USD", "Date"]))        # features
	y_ = df[!, "EUR/USD"]                            # target
	
	# Drop rows with missing values in features or target
	complete_rows = completecases(hcat(X_, y_))
	X = X_[complete_rows, :]
	y = y_[complete_rows]
	X = coalesce.(X, 0.0)
    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, frac=0.8, rng=Random.default_rng())
end

# ╔═╡ 1e0103c3-900c-4888-b9e8-3b1dc2aa5741
md"""
### 📊 Dataset Variables

- **Date** → Trading date of the record.  
- **SPX** → S&P 500 Index (U.S. stock market performance).  
- **GLD** → SPDR Gold Shares ETF (proxy for gold price, our target).  
- **USO** → United States Oil Fund (tracks crude oil prices).  
- **SLV** → iShares Silver Trust (tracks silver prices).  
- **EUR/USD** → Euro to U.S. Dollar exchange rate.  

These features are used to study and predict the movement of **gold prices**.
"""

# ╔═╡ 2247b467-bcc6-4eab-8ab8-50504ad46650
begin
    # 2) Inspect data
    describe(df, :nmissing), names(df), eltype.(eachcol(df))
end

# ╔═╡ 0c1a1335-d810-431c-87ae-c94f6c8075d0
begin
    # Dataset information
    println("Dataset info:")
    println(size(df))         # (rows, columns)
    # println("Column names and types:")
    # println(schema(df))       # shows column names and types
    
    println("\nSummary statistics:")
    describe(df)
end

# ╔═╡ a454b789-7e88-4e34-aaf8-d6c15127fa3e


# ╔═╡ 48df40b0-30c7-4e29-a0b6-27782ee3c60a
begin
    # using DataFrames, Statistics, StatsPlots

    # Select numeric columns
    numeric_cols = names(df, Number)

    # Compute correlation matrix
    corr_matrix = cor(Matrix(df[:, numeric_cols]))

    # Create annotation labels
    # Build flat vector of annotations
	annots = [(i, j, string(round(corr_matrix[j, i], digits=2)))
	          for i in 1:length(numeric_cols), j in 1:length(numeric_cols)]
	annots = vec(annots)  # flatten into 1D vector
	
	heatmap(
	    corr_matrix;
	    xticks=(1:length(numeric_cols), string.(numeric_cols)),
	    yticks=(1:length(numeric_cols), string.(numeric_cols)),
	    color=:coolwarm,
	    aspect_ratio=:equal,
	    clims=(-1, 1),
	    title="Correlation Matrix Heatmap",
	    xlabel="Features",
	    ylabel="Features",
	    annotate=annots,
	    annotationfontsize=7
)
end


# ╔═╡ 65854fcf-fdb7-4f39-a68b-88c12ac9631e
begin
    using StatsPlots  # extends Plots with histogram + density combos


    # Layout: 2 rows, 3 columns (adjust if more cols)
    layout = @layout [a b c; d e f]

    plt_list = [
        plot(
            histogram(df[!, col], normalize=true, alpha=0.4, legend=false),
            density(df[!, col], lw=2, color=:red, legend=false),
            title=col,
            xlabel=col,
            ylabel="Density"
        )
        for col in numeric_cols
    ]

    plot(plt_list..., layout=layout, size=(1600, 1600))
         # title="Distribution of data across columns")
end

# ╔═╡ 1e41ece5-6f56-43c4-b8c0-7874ee647fdd
begin
    using RollingFunctions

    #  Rolling mean with window=20, padded to match df length
	trend = [fill(missing, 19); rollmean(df."EUR/USD", 20)]
	
	df.price_trend = trend
	
	plot(
	    df.Date, df.price_trend;
	    title="Trend in price of gold through date",
	    xlabel="Date",
	    ylabel="Price (EUR/USD)",
	    lw=2,
	    color=:red,
	    legend=false
	)
end

# ╔═╡ ed90a6a4-61b9-47a8-b1d8-e057106f1247
begin
    # Drop SLV column since it's highly correlated with GLD
    if "SLV" in names(df)
        select!(df, Not("SLV"))   # modifies df in place, drops SLV column
    else
        @warn "Column 'SLV' not found in DataFrame."
    end

    first(df, 5)   # show first 5 rows after dropping
end

# ╔═╡ bfa3edc0-8fee-459d-bf34-d905c5f1353b
begin
	# Parse Date column to Date type
    df.Date = Date.(df.Date, dateformat"m/d/y")

    # Plot EUR/USD against Date
    plot(
        df.Date, df."EUR/USD";
        title="Change in price of gold through date",
        xlabel="Date",
        ylabel="EUR/USD",
        legend=false,
        lw=2,
        color=:blue
    )
end

# ╔═╡ 313556d4-417c-4638-993e-0d9d0255d21d
begin
    # using StatsBase

 	# Re-define numeric columns excluding Date
    num_cols_skew = setdiff(names(df, Number), ["Date"])

    # Compute skewness
    skew_vals = Dict(
        col => skewness(collect(skipmissing(df[!, col])))
        for col in num_cols_skew
    )

    println("Skewness values:")
    for (col, val) in skew_vals
        println("$(col): $(round(val, digits=2))")
    end
end

# ╔═╡ 95153f37-38ca-4576-9cf3-14355dbeee33
begin
    if "USO" in names(df)
        df.USO = sqrt.(df.USO)   # elementwise sqrt
    else
        @warn "Column 'USO' not found in DataFrame."
    end

    first(df, 5)  # show preview
end

# ╔═╡ 7722c1d3-ee31-4a07-be5b-68b7aa49ba75
md"""
### Remove Outlier
"""

# ╔═╡ 05490d22-7d62-42b8-94e0-fc9db459547b
begin
    # using StatsPlots, DataFrames

    # Only keep numeric columns (skip Date or any non-numeric)
    # Select numeric columns only
    numeric_cols_box = [col for col in names(df) if eltype(df[!, col]) <: Real]

    # Layout: 2 rows, dynamic number of columns
    n = length(numeric_cols_box)
    nrows = 2
    ncols = ceil(Int, n / nrows)
    layout_box = @layout grid(nrows, ncols)

    # Generate boxplots for each numeric column
    boxplot_list = [
        boxplot(df[!, col], color=:violet, legend=false, ylabel=col)
        for col in numeric_cols_box
    ]

    # Combine all boxplots in a single figure
    plot(boxplot_list..., layout=layout_box, size=(800, 800))

end

# ╔═╡ 209ed814-6aeb-4f73-8bba-04f21d04ccee
md"""
### Data Preprocessing
"""

# ╔═╡ 6113ba82-5f5e-4ebc-a120-aabd9f98b1e2


# ╔═╡ 853d7e98-efbd-4ad6-8cdf-0ab7f9f51d55
md"""
### Model Training

If some bug arise here, try reruning code cell 2 to reload the utils. Or you may ignore this and check the final output
"""

# ╔═╡ bd57314a-fc4e-43fe-8702-c66eb17a627e
begin
	
    # --- Train models ---
    y_pred_lin, y_pred_rf = train_models(X_train, y_train, X_test)

    # --- Evaluate performance ---
    println("Linear Regression:")
    println("RMSE: ", round(rmse(y_test, y_pred_lin), digits=3))
    println("R²:   ", round(r2_score(y_test, y_pred_lin), digits=3))

    println("\nRandom Forest Regression:")
    println("RMSE: ", round(rmse(y_test, y_pred_rf), digits=3))
    println("R²:   ", round(r2_score(y_test, y_pred_rf), digits=3))
end


# ╔═╡ 5bf814d4-98b0-4a80-b02b-36b0e8828207
"""
If some bug arise here, try reruning code cell 2 to reload the utils.
"""


# ╔═╡ 4d1f15b9-2098-454d-bbfe-a9ad8978172c
begin
    # Predictions on training data
    y_pred_train_lin, y_pred_train_rf = train_models(X_train, y_train, X_train)

    # Create scatter plot with proper labels
    scatter(y_train, y_pred_train_lin,
        xlabel="Actual Price of Gold in EUR/USD",
        ylabel="Predicted Gold Price",
        title="Predicted vs Actual ",
        label="Linear Regression",    # label for legend
        color=:blue,
        markersize=4
    )

    # Add Random Forest predictions
    scatter!(y_train, y_pred_train_rf,
        label="Random Forest",
        color=:red,
        markersize=4
    )

    # Add 45° line for reference
    plot!(x -> x, minimum(y_train):maximum(y_train),
        linestyle=:dash,
        color=:black,
        label="Ideal Fit"
    )

    # Show the legend
    plot!(legend=:topleft)
end

# ╔═╡ Cell order:
# ╠═a60f1ce0-892c-11f0-2805-ed8eb3b20852
# ╠═a7d4bbc4-09b1-432f-957a-f0dff3e56661
# ╠═b09606bf-4fc8-47ca-bf13-7b89ca0e1391
# ╠═b617fd78-bd35-47c1-b0ae-c4f88e21d524
# ╠═1e0103c3-900c-4888-b9e8-3b1dc2aa5741
# ╟─2247b467-bcc6-4eab-8ab8-50504ad46650
# ╠═0c1a1335-d810-431c-87ae-c94f6c8075d0
# ╠═a454b789-7e88-4e34-aaf8-d6c15127fa3e
# ╠═48df40b0-30c7-4e29-a0b6-27782ee3c60a
# ╠═ed90a6a4-61b9-47a8-b1d8-e057106f1247
# ╠═bfa3edc0-8fee-459d-bf34-d905c5f1353b
# ╠═1e41ece5-6f56-43c4-b8c0-7874ee647fdd
# ╠═65854fcf-fdb7-4f39-a68b-88c12ac9631e
# ╠═313556d4-417c-4638-993e-0d9d0255d21d
# ╠═95153f37-38ca-4576-9cf3-14355dbeee33
# ╟─7722c1d3-ee31-4a07-be5b-68b7aa49ba75
# ╠═05490d22-7d62-42b8-94e0-fc9db459547b
# ╟─209ed814-6aeb-4f73-8bba-04f21d04ccee
# ╠═6113ba82-5f5e-4ebc-a120-aabd9f98b1e2
# ╠═08ec388f-ceae-4269-a55b-21b58b1c7e7c
# ╠═97467d9f-3822-4fd3-9f36-a631b0adf8bb
# ╟─853d7e98-efbd-4ad6-8cdf-0ab7f9f51d55
# ╠═bd57314a-fc4e-43fe-8702-c66eb17a627e
# ╟─5bf814d4-98b0-4a80-b02b-36b0e8828207
# ╠═4d1f15b9-2098-454d-bbfe-a9ad8978172c
