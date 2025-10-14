import polars as pl
import yfinance as yf
import datetime as dt
import kagglehub

print("--- Starting Data Preparation ---")
print("Downloading and loading raw tweet data from KaggleHub...")

try:
    path = kagglehub.dataset_download("codebreaker619/donald-trump-tweets-dataset")
    # Load and parse the date column immediately, making it timezone-aware
    df = pl.read_csv(path + "/tweets.csv").with_columns(
        pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC")
    )
    print(f"Loaded {len(df)} raw tweets.")
except Exception as e:
    print(f"Error downloading or reading Kaggle dataset: {e}")
    exit()


trump_start = dt.datetime(2017, 1, 17, tzinfo=dt.timezone.utc)
trump_end = dt.datetime(2021, 1, 9, tzinfo=dt.timezone.utc)

print(f"Filtering tweets from {trump_start.date()} to {trump_end.date()}...")
filtered_df = df.filter(
    pl.col("date").is_between(trump_start, trump_end)
)
print(f"Found {len(filtered_df)} tweets in the specified date range.")

print(f"Fetching VOO data...")
voo_data = yf.download("VOO", start=trump_start.date(), end=trump_end.date())
voo_data.index = voo_data.index.tz_localize('UTC')
print("VOO data fetched successfully.")

print("Calculating forward returns and assigning labels...")
lookahead_period = dt.timedelta(hours=24)

def get_future_price(current_time, price_series, time_delta):
    future_time = current_time + time_delta
    future_index = price_series.index.searchsorted(future_time)
    if future_index < len(price_series):
        return price_series.iloc[future_index]
    return None

forward_returns = []

for id, text, isRetweet, isDeleted, device, favorites, retweets, date, isFlagged in df.iter_rows():
    try:
        start_price_idx = voo_data.index.searchsorted(date)
        if start_price_idx < len(voo_data.index):
            start_price = voo_data['Close'].iloc[start_price_idx]
        else:
            start_price = None
        future_price = get_future_price(date, voo_data['Close'], lookahead_period)
        if start_price is not None and future_price is not None:
            returns = ((future_price / start_price) - 1) * 100
            forward_returns.append((int(id), returns))
        else:
            forward_returns.append((int(id), None))
    except Exception as e:
        print(f"Error processing tweet {id}: {e}")
        forward_returns.append((int(id), None))

returns_df = pl.DataFrame(forward_returns, schema=["id", "forward_return_%"])
joined_df = filtered_df.join(returns_df, on="id", how="left")

labeled_df = joined_df.with_columns(
    pl.when(pl.col("forward_return_%") > 0.5).then(pl.lit(2))
    .when(pl.col("forward_return_%") < -0.5).then(pl.lit(0))
    .otherwise(pl.lit(1))
    .alias("label")
).filter(
    (pl.col("text").is_not_null()) &
    (pl.col("text") != "") &
    (~pl.col("text").str.contains(r"^https?://\S+$"))
)

labeled_df.write_ipc("labeled_dataset.arrow")
# labeled_df.write_csv("labeled_dataset.csv")
print("✅ Data preparation complete.")
