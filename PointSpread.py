import os
import pymysql
from sshtunnel import SSHTunnelForwarder
import pandas as pd
import time
import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

import io
import base64

def get_quote_data(date, symbol):
    """
    Fetch quote data for a specific date and symbol from Alp_Quotes.
    Checks if data already exists in a file; skips query if it does.
    """
    # -------------------------------
    # 0. Ensure the Data directory exists
    # -------------------------------
    data_dir = 'Data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    # -------------------------------
    # 1. Transform symbol for filename
    # -------------------------------
    symbol_transformed = symbol.replace('/', '')

    # -------------------------------
    # 2. Build the partition name
    # -------------------------------
    date_ts = pd.Timestamp(date)
    month_map = {
        1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
        7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"
    }
    partition_name = f"p_{month_map[date_ts.month]}_{date_ts.year}"

    # -------------------------------
    # 3. Build time filter boundaries
    # -------------------------------
    start_str = date_ts.strftime("%Y-%m-%d 00:00:00")
    end_str = (date_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")

    # -------------------------------
    # 4. Clean up symbol for file naming
    # -------------------------------
    date_str = date_ts.strftime("%Y-%m-%d")
    file_name = f"{symbol_transformed}_{date_str}.pkl"
    file_path = os.path.join(data_dir, file_name)

    # -------------------------------
    # 5. Check if file exists locally
    # -------------------------------
    if os.path.exists(file_path):
        try:
            print(f"Loading data from local file: {file_path}")
            df = pd.read_pickle(file_path)
            return df
        except Exception as e:
            print(f"ERROR loading local file {file_path}: {str(e)}")
            # Proceed to fetch from server if loading fails

    # -------------------------------
    # 6. Build the SQL query
    # -------------------------------
    query = f"""
        SELECT 
            MakerId, 
            CoreSymbol, 
            TimeRecorded, 
            TimeSent, 
            TimeReceived, 
            Depth, 
            Side, 
            Price, 
            Size, 
            Provider, 
            IndicativeFlags, 
            QuoteFlags, 
            DisabledFlags, 
            ForwardPriceDelta, 
            id
        FROM Alp_Quotes PARTITION ({partition_name})
        FORCE INDEX (idx_time_recorded)
        WHERE 
            CoreSymbol = '{symbol}'
            AND TimeRecorded >= '{start_str}'
            AND TimeRecorded < '{end_str}';
    """

    ssh_host = '18.133.184.11'
    ssh_user = 'ubuntu'
    ssh_key_file = '/Users/jackhan/Desktop/Alpfin/OneZero_Data.pem'
    db_host = '127.0.0.1'
    db_port = 3306
    db_user = 'Ruize'
    db_password = 'Ma5hedPotato567='
    db_name = 'Alp_CPT_Data'

    columns = [
        "MakerId",
        "CoreSymbol",
        "TimeRecorded",
        "TimeSent",
        "TimeReceived",
        "Depth",
        "Side",
        "Price",
        "Size",
        "Provider",
        "IndicativeFlags",
        "QuoteFlags",
        "DisabledFlags",
        "ForwardPriceDelta",
        "id"
    ]

    try:
        with SSHTunnelForwarder(
            (ssh_host, 22),
            ssh_username=ssh_user,
            ssh_pkey=ssh_key_file,
            remote_bind_address=(db_host, db_port),
            allow_agent=False,
            host_pkey_directories=[]
        ) as tunnel:
            
            connection = pymysql.connect(
                host='127.0.0.1',
                port=tunnel.local_bind_port,
                user=db_user,
                password=db_password,
                database=db_name,
                connect_timeout=10
            )
            
            try:
                cursor = connection.cursor()
                
                # Start Query Timer
                query_start_time = time.time()
                print("Start Query: ", query)
                cursor.execute(query)
                query_duration = time.time() - query_start_time
                print(f"Query Execution Time: {query_duration:.2f} seconds.")
                
                # Data Transfer Timer
                transfer_start_time = time.time()
                rows = cursor.fetchall()
                transfer_duration = time.time() - transfer_start_time
                print(f"Data Transfer Time: {transfer_duration:.2f} seconds.")
                
                total_duration = query_duration + transfer_duration
                print(f"[{symbol} | {date_str}] Total Time: {total_duration:.2f} seconds.")
                
                print("Fetch Success")
                df = pd.DataFrame(rows, columns=columns)
                
                # Save the dataframe locally for future use
                try:
                    df.to_pickle(file_path)
                    print(f"Data saved locally to {file_path}")
                except Exception as e:
                    print(f"ERROR saving data to {file_path}: {str(e)}")
                
                return df
            
            finally:
                cursor.close()
                connection.close()
                
    except Exception as e:
        print(f"ERROR for {symbol} on {date_str}: {str(e)}")
        return None

def PointSpreadDisplay(df_input, trade_vol, date, maker_id="Britannia", top_of_book=False, symbol=""):
    """
    Calculates point difference across the quote data, returning
    multiple plots (as base64) and a statistics dictionary.
    """

    # Filter by MakerId
    df_loaded = df_input[df_input['MakerId'] == maker_id].copy()

    # Convert TimeRecorded to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_loaded['TimeRecorded']):
        df_loaded['TimeRecorded'] = pd.to_datetime(df_loaded['TimeRecorded'])

    # Split by depth & side
    depth_dfs = {}
    for depth in range(7):  # Depth ranges from 0 to 6
        # Sell side
        sell_df = df_loaded[(df_loaded["Side"] == 0) & (df_loaded["Depth"] == depth)].copy()
        sell_df = sell_df.rename(
            columns={
                "Price": f"Sell_Price_Depth{depth}", 
                "Size": f"Sell_Size_Depth{depth}"
            }
        )
        sell_df = sell_df[["CoreSymbol", "TimeRecorded", f"Sell_Price_Depth{depth}", f"Sell_Size_Depth{depth}"]]
        depth_dfs[f'sell_df_depth{depth}'] = sell_df

        # Buy side
        buy_df = df_loaded[(df_loaded["Side"] == 1) & (df_loaded["Depth"] == depth)].copy()
        buy_df = buy_df.rename(
            columns={
                "Price": f"Depth{depth}_Buy_Price", 
                "Size": f"Depth{depth}_Buy_Size"
            }
        )
        buy_df = buy_df[["CoreSymbol", "TimeRecorded", f"Depth{depth}_Buy_Price", f"Depth{depth}_Buy_Size"]]
        depth_dfs[f'buy_df_depth{depth}'] = buy_df

    # Merge them all
    merged_df = depth_dfs['sell_df_depth0']
    for depth in range(1, 7):
        merged_df = merged_df.merge(depth_dfs[f'sell_df_depth{depth}'], on=["CoreSymbol", "TimeRecorded"], how="outer")
    for depth in range(7):
        merged_df = merged_df.merge(depth_dfs[f'buy_df_depth{depth}'], on=["CoreSymbol", "TimeRecorded"], how="outer")

    # Sort by TimeRecorded so it makes sense in time-series
    merged_df = merged_df.sort_values(by="TimeRecorded").reset_index(drop=True)

    # Calculate Point_Diff
    if top_of_book:
        # Just Depth0_Buy_Price - Sell_Price_Depth0
        merged_df["Point_Diff"] = (
            merged_df["Depth0_Buy_Price"] - merged_df["Sell_Price_Depth0"]
        )
    else:
        # Full fill simulation based on trade_vol
        def calculate_price_difference_per_row(row, vol):
            # Initialize variables
            total_buy_vol = 0.0
            total_sell_vol = 0.0
            weighted_buy_price = 0.0
            weighted_sell_price = 0.0

            # Max available volumes
            max_buy_vol = np.nansum([row.get(f"Depth{d}_Buy_Size", 0) for d in range(7)])
            max_sell_vol = np.nansum([row.get(f"Sell_Size_Depth{d}", 0) for d in range(7)])

            # Actual volume for the trade is limited by what's available
            capped_vol = min(vol, max_buy_vol, max_sell_vol)

            # Simulate buy
            remaining_vol = capped_vol
            for d in range(7):
                buy_price = row.get(f"Depth{d}_Buy_Price", np.nan)
                buy_size = row.get(f"Depth{d}_Buy_Size", 0.0)
                if pd.isna(buy_price):
                    continue
                if remaining_vol <= 0:
                    break

                buyable_vol = min(buy_size, remaining_vol)
                weighted_buy_price += buyable_vol * buy_price
                total_buy_vol += buyable_vol
                remaining_vol -= buyable_vol

            # Simulate sell
            remaining_vol = capped_vol
            for d in range(7):
                sell_price = row.get(f"Sell_Price_Depth{d}", np.nan)
                sell_size = row.get(f"Sell_Size_Depth{d}", 0.0)
                if pd.isna(sell_price):
                    continue
                if remaining_vol <= 0:
                    break

                sellable_vol = min(sell_size, remaining_vol)
                weighted_sell_price += sellable_vol * sell_price
                total_sell_vol += sellable_vol
                remaining_vol -= sellable_vol

            # Averages
            avg_buy = weighted_buy_price / total_buy_vol if total_buy_vol > 0 else np.nan
            avg_sell = weighted_sell_price / total_sell_vol if total_sell_vol > 0 else np.nan

            # difference
            return avg_buy - avg_sell if (not np.isnan(avg_buy) and not np.isnan(avg_sell)) else np.nan

        merged_df["Point_Diff"] = merged_df.apply(
            lambda row: calculate_price_difference_per_row(row, trade_vol), axis=1
        )

    # ---------- PLOTS ----------
    # Common title substring
    vol_label = "Top-of-Book" if top_of_book else f"Volume {trade_vol}"
    common_title = f"{symbol} on {date} ({vol_label})"

    # SECTION 1: Point difference over time (all data)
    plt.figure(figsize=(12, 6))
    plt.plot(
        merged_df["TimeRecorded"], 
        merged_df["Point_Diff"], 
        marker="o",
        linestyle="none",
        markersize=2,
        alpha=0.5
    )
    plt.title(f"Point Difference Over Time - {common_title}", fontsize=16)
    plt.xlabel("TimeRecorded", fontsize=14)
    plt.ylabel("Point Difference", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot1_url = "data:image/png;base64," + base64.b64encode(img1.getvalue()).decode()
    plt.close()

    # SECTION 2: 24 subplots, one per hour
    # Ensure we have an Hour column
    merged_df["Hour"] = merged_df["TimeRecorded"].dt.hour
    # Create figure with 24 subplots
    fig, axes = plt.subplots(nrows=24, ncols=1, figsize=(12, 60), sharex=False, sharey=False)
    fig.suptitle(f"Point Difference Over Time by Hour - {common_title}", fontsize=16)

    y_min = merged_df["Point_Diff"].min()
    y_max = merged_df["Point_Diff"].max()

    for hour in range(24):
        ax = axes[hour]
        hour_df = merged_df[merged_df["Hour"] == hour]
        ax.plot(
            hour_df["TimeRecorded"],
            hour_df["Point_Diff"],
            marker="o",
            linestyle="none",
            markersize=2,
            alpha=0.5
        )
        # You could set same y-limits across all if desired:
        # ax.set_ylim([y_min, y_max])
        ax.set_title(f"Hour {hour:02d}:00", fontsize=12)
        ax.set_xlabel("TimeRecorded", fontsize=10)
        ax.set_ylabel("Point Diff", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    plot2_url = "data:image/png;base64," + base64.b64encode(img2.getvalue()).decode()
    plt.close()

    # SECTION 3a: Distribution of Point_Diff (normal scale)
    plt.figure(figsize=(10, 6))
    plt.hist(merged_df["Point_Diff"].dropna(), bins=50, edgecolor="k", alpha=0.7)
    plt.title(f"Distribution of Point_Diff - {common_title}", fontsize=16)
    plt.xlabel("Point_Diff", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    img3 = io.BytesIO()
    plt.savefig(img3, format='png')
    img3.seek(0)
    plot3_url = "data:image/png;base64," + base64.b64encode(img3.getvalue()).decode()
    plt.close()

    # SECTION 3b: Distribution of Point_Diff (log scale)
    plt.figure(figsize=(10, 6))
    plt.hist(merged_df["Point_Diff"].dropna(), bins=50, edgecolor="k", alpha=0.7, log=True)
    plt.title(f"Distribution of Point_Diff (Log Scale) - {common_title}", fontsize=16)
    plt.xlabel("Point_Diff", fontsize=14)
    plt.ylabel("Frequency (log scale)", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    img4 = io.BytesIO()
    plt.savefig(img4, format='png')
    img4.seek(0)
    plot4_url = "data:image/png;base64," + base64.b64encode(img4.getvalue()).decode()
    plt.close()

    # SECTION 4: Outlier Plot
    Q1 = merged_df["Point_Diff"].quantile(0.25)
    Q3 = merged_df["Point_Diff"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = merged_df[(merged_df["Point_Diff"] < lower_bound) | (merged_df["Point_Diff"] > upper_bound)]
    num_outliers = len(outliers)

    plt.figure(figsize=(12, 7))
    # All data
    sns.histplot(
        merged_df["Point_Diff"], 
        bins=50, 
        color='skyblue', 
        edgecolor='black', 
        label='Data', 
        alpha=0.7
    )
    # Outliers
    if not outliers.empty:
        sns.histplot(
            outliers["Point_Diff"], 
            bins=50, 
            color='red', 
            edgecolor='black', 
            label='Outliers', 
            alpha=0.7
        )
    # Bounds
    plt.axvline(lower_bound, color='green', linestyle='--', linewidth=2, label=f'Lower Bound ({lower_bound:.2f})')
    plt.axvline(upper_bound, color='purple', linestyle='--', linewidth=2, label=f'Upper Bound ({upper_bound:.2f})')
    plt.title(f"Histogram of Point_Diff (Outliers Highlighted) - {common_title}", fontsize=18)
    plt.xlabel("Point_Diff", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    img5 = io.BytesIO()
    plt.savefig(img5, format='png')
    img5.seek(0)
    plot5_url = "data:image/png;base64," + base64.b64encode(img5.getvalue()).decode()
    plt.close()

    # ---------- STATISTICS ----------
    describe_default = merged_df['Point_Diff'].describe().to_dict()
    describe_custom = merged_df['Point_Diff'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()

    mean_val = merged_df['Point_Diff'].mean()
    median_val = merged_df['Point_Diff'].median()
    std_val = merged_df['Point_Diff'].std()
    var_val = merged_df['Point_Diff'].var()
    min_val = merged_df['Point_Diff'].min()
    max_val = merged_df['Point_Diff'].max()
    skew_val = merged_df['Point_Diff'].skew()
    kurt_val = merged_df['Point_Diff'].kurt()

    specific_stats = {
        "Mean": mean_val,
        "Median": median_val,
        "Standard Deviation": std_val,
        "Variance": var_val,
        "Min": min_val,
        "Max": max_val,
        "Skewness": skew_val,
        "Kurtosis": kurt_val
    }

    statistics = {
        "describe_default": describe_default,
        "describe_custom": describe_custom,
        "specific_stats": specific_stats,
        "outlier_info": {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Number of Outliers": num_outliers
        }
    }

    return {
        "plot1_url": plot1_url,  # Section 1
        "plot2_url": plot2_url,  # Section 2
        "plot3_url": plot3_url,  # Section 3 (normal scale)
        "plot4_url": plot4_url,  # Section 3 (log scale)
        "plot5_url": plot5_url,  # Section 4 (outlier)
        "statistics": statistics
    }