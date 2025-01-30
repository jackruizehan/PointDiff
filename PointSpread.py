import os
import pymysql
import pandas as pd
import time
import numpy as np
import seaborn as sns
from sshtunnel import SSHTunnelForwarder

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

import io
import base64

def get_quote_data(date_from, date_to, symbol, use_ssh=False):
    """
    Fetch quote data for a specific date range (YYYY-MM-DD) and symbol from Alp_Quotes or local storage.
    
    - If all required local `.pkl` files exist for the date range and symbol, load data from local files.
    - Otherwise, fetch data from the database, handling multiple partitions if necessary.
    - Allows connecting directly to the DB or via SSH based on the use_ssh flag.
    
    Parameters:
        date_from (str): Start date in 'YYYY-MM-DD' format.
        date_to (str): End date in 'YYYY-MM-DD' format.
        symbol (str): The symbol to fetch data for.
        use_ssh (bool): Whether to connect via SSH. Default is False.
    
    Returns:
        pd.DataFrame: DataFrame containing the fetched quote data, or None if an error occurs.
    """
    try:
        # -------------------------------
        # 1. Transform symbol for SQL query and local file
        # -------------------------------
        symbol_transformed = symbol.replace('/', '')
        print(f"Transformed symbol: {symbol_transformed}")

        # -------------------------------
        # 2. Generate list of all dates in the range
        # -------------------------------
        from_date_ts = pd.Timestamp(date_from)
        to_date_ts = pd.Timestamp(date_to)
        all_dates = pd.date_range(start=from_date_ts, end=to_date_ts, freq='D')
        date_strings = all_dates.strftime('%Y-%m-%d').tolist()
        print(f"Date range: {date_strings}")

        # -------------------------------
        # 3. Check for local data availability
        # -------------------------------
        local_data_available = True
        local_dataframes = []
        for date_str in date_strings:
            file_path = os.path.join('Data', f"{symbol_transformed}_{date_str}.pkl")
            if not os.path.exists(file_path):
                print(f"Local file missing: {file_path}")
                local_data_available = False
                break
            else:
                print(f"Loading local file: {file_path}")
                try:
                    df_local = pd.read_pickle(file_path)
                    # Select relevant columns
                    df_local = df_local[[
                        "MakerId",
                        "CoreSymbol",
                        "TimeRecorded",
                        "Depth",
                        "Side",
                        "Price",
                        "Size",
                    ]]
                    local_dataframes.append(df_local)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
                    local_data_available = False
                    break

        if local_data_available:
            print("All local data files found. Loading data from local storage.")
            if local_dataframes:
                result_df = pd.concat(local_dataframes, ignore_index=True)
                print(f"Loaded {len(result_df)} rows from local files.")
                return result_df
            else:
                print("No data found in local files for the specified range.")
                return pd.DataFrame(columns=[
                    "MakerId",
                    "CoreSymbol",
                    "TimeRecorded",
                    "Depth",
                    "Side",
                    "Price",
                    "Size",
                ])
        else:
            print("Local data not fully available. Proceeding to fetch data from the database.")

        # -------------------------------
        # 4. Determine all relevant partitions
        # -------------------------------
        month_map = {
            1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
            7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"
        }

        # Generate list of months between from_date and to_date inclusive
        all_partitions = []
        current = from_date_ts.replace(day=1)
        end = to_date_ts.replace(day=1)
        while current <= end:
            partition_name = f"p_{month_map[current.month]}_{current.year}"
            all_partitions.append(partition_name)
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        print(f"Identified partitions: {all_partitions}")

        # -------------------------------
        # 5. Build time filter boundaries
        # -------------------------------
        start_str = from_date_ts.strftime("%Y-%m-%d 00:00:00")
        end_str = to_date_ts.strftime("%Y-%m-%d 00:00:00")
        print(f"Time filter: {start_str} to {end_str}")

        # -------------------------------
        # 6. Database Connection Parameters
        # -------------------------------
        ssh_host = '18.133.184.11'
        ssh_user = 'ubuntu'
        ssh_key_file = '/Users/jackhan/Desktop/Alpfin/OneZero_Data.pem'

        db_host_direct = '127.0.0.1'  # Replace with actual direct DB host
        db_host_via_ssh = '127.0.0.1'  # DB host when connecting via SSH
        db_port = 3306
        db_user = 'Ruize'
        db_password = 'Ma5hedPotato567='
        db_name = 'Alp_CPT_Data'

        columns = [
            "MakerId",
            "CoreSymbol",
            "TimeRecorded",
            "Depth",
            "Side",
            "Price",
            "Size",
        ]

        # -------------------------------
        # 7. Build the SQL query for each partition
        # -------------------------------
        queries = []
        for partition in all_partitions:
            query = f"""
                SELECT 
                    MakerId, 
                    CoreSymbol, 
                    TimeRecorded, 
                    Depth, 
                    Side, 
                    Price, 
                    Size
                FROM Alp_Quotes PARTITION ({partition})
                FORCE INDEX (idx_time_recorded)
                WHERE 
                    CoreSymbol = '{symbol}'
                    AND TimeRecorded >= '{start_str}'
                    AND TimeRecorded < '{end_str}';
            """
            queries.append(query)
        
        # -------------------------------
        # 8. Connect to the database and fetch data
        # -------------------------------
        if use_ssh:
            # SSH connection parameters
            with SSHTunnelForwarder(
                (ssh_host, 22),
                ssh_username=ssh_user,
                ssh_pkey=ssh_key_file,
                remote_bind_address=(db_host_via_ssh, db_port),
                allow_agent=False,
                host_pkey_directories=[],  # Disable loading keys from ~/.ssh
            ) as tunnel:
                local_port = tunnel.local_bind_port
                print(f"SSH Tunnel established on local port {local_port}")
                connection = pymysql.connect(
                    host='127.0.0.1',
                    port=local_port,
                    user=db_user,
                    password=db_password,
                    database=db_name,
                    connect_timeout=10

                )

                try:
                    print("Establishing DB Connection via SSH")
                    cursor = connection.cursor()
                    print(f"Executing queries for {symbol} from {date_from} to {date_to}...")
                    
                    # Execute each query and collect results
                    dataframes = []
                    for q in queries:
                        print(f"Executing query:\n{q}")
                        cursor.execute(q)
                        rows = cursor.fetchall()
                        print(f"Fetched {len(rows)} rows from partition.")
                        df = pd.DataFrame(rows, columns=columns)
                        dataframes.append(df)
                    
                    # Concatenate all DataFrames
                    if dataframes:
                        result_df = pd.concat(dataframes, ignore_index=True)
                        print(f"All data fetched and concatenated. Total rows: {len(result_df)}")
                    else:
                        result_df = pd.DataFrame(columns=columns)
                        print("No data fetched from database.")
                    
                    return result_df

                finally:
                    cursor.close()
                    connection.close()
                    print("Database connection via SSH closed.")
        else:
            # Direct DB connection parameters
            connection = pymysql.connect(
                host=db_host_direct,
                port=db_port,
                user=db_user,
                password=db_password,
                database=db_name,
                connect_timeout=10
            )
            try:
                print("Establishing DB Connection directly")
                cursor = connection.cursor()
                print(f"Executing queries for {symbol} from {date_from} to {date_to}...")
                
                # Execute each query and collect results
                dataframes = []
                for q in queries:
                    print(f"Executing query:\n{q}")
                    cursor.execute(q)
                    rows = cursor.fetchall()
                    print(f"Fetched {len(rows)} rows from partition.")
                    df = pd.DataFrame(rows, columns=columns)
                    dataframes.append(df)
                
                # Concatenate all DataFrames
                if dataframes:
                    result_df = pd.concat(dataframes, ignore_index=True)
                    print(f"All data fetched and concatenated. Total rows: {len(result_df)}")
                else:
                    result_df = pd.DataFrame(columns=columns)
                    print("No data fetched from database.")
                
                return result_df

            finally:
                cursor.close()
                connection.close()
                print("Direct database connection closed.")

    except Exception as e:
        print(f"ERROR fetching data for {symbol} from {date_from} to {date_to}: {str(e)}")
        return None
    


    
def PointSpreadDisplay(df_input, trade_vol, date_range, maker_id="Britannia", top_of_book=True, symbol="XAU/USD"):
    """
    Create:
      1) Main Point Diff Over Time plot (entire range).
      2) Distribution plots (normal histogram, log histogram, outliers).
      3) A dictionary of point-diff-over-time *by date & hour* 
         so the front-end can display whichever (date, hour) combination is selected.

    Return:
      {
        "main_time_plot": <base64>,
        "distribution_plots": {
            "hist_normal": <base64>,
            "hist_log": <base64>,
            "outlier": <base64>,
        },
        "hourly_plots": {
            "YYYY-MM-DD": {
                "0": <base64 plot>,
                "1": <base64 plot>,
                ...
            },
            "YYYY-MM-DD+1": {...},
            ...
        },
        "statistics": {...}
      }
    """
    df_loaded = df_input[df_input['MakerId'] == maker_id]

    # Convert TimeRecorded to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_loaded['TimeRecorded']):
        df_loaded['TimeRecorded'] = pd.to_datetime(df_loaded['TimeRecorded'])

    # Sort by TimeRecorded
    df_loaded.sort_values(by="TimeRecorded", inplace=True)

    # 1. Merge Sell/Buy by Depth (like before)
    depth_dfs = {}
    for depth in range(7):
        # Sell side: Side=0
        sell_df = df_loaded[(df_loaded["Side"] == 0) & (df_loaded["Depth"] == depth)]
        sell_df = sell_df.rename(
            columns={
                "Price": f"Sell_Price_Depth{depth}", 
                "Size": f"Sell_Size_Depth{depth}"
            }
        )
        sell_df = sell_df[["CoreSymbol", "TimeRecorded", f"Sell_Price_Depth{depth}", f"Sell_Size_Depth{depth}"]]
        depth_dfs[f'sell_df_depth{depth}'] = sell_df

        # Buy side: Side=1
        buy_df = df_loaded[(df_loaded["Side"] == 1) & (df_loaded["Depth"] == depth)]
        buy_df = buy_df.rename(
            columns={
                "Price": f"Buy_Price_Depth{depth}", 
                "Size": f"Buy_Size_Depth{depth}"
            }
        )
        buy_df = buy_df[["CoreSymbol", "TimeRecorded", f"Buy_Price_Depth{depth}", f"Buy_Size_Depth{depth}"]]
        depth_dfs[f'buy_df_depth{depth}'] = buy_df

    merged_df = None
    # Start with Depth0 sell
    merged_df = depth_dfs['sell_df_depth0']

    # Merge sells depth1..6
    for depth in range(1, 7):
        merged_df = merged_df.merge(depth_dfs[f'sell_df_depth{depth}'], on=["CoreSymbol", "TimeRecorded"], how="outer")

    # Merge buys depth0..6
    for depth in range(7):
        merged_df = merged_df.merge(depth_dfs[f'buy_df_depth{depth}'], on=["CoreSymbol", "TimeRecorded"], how="outer")

    merged_df.sort_values(by="TimeRecorded", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # 2. Calculate "Point_Diff" depending on top_of_book or volume fill
    if top_of_book:
        # Depth0 only: Buy_Price_Depth0 - Sell_Price_Depth0
        merged_df["Point_Diff"] = (
            merged_df["Buy_Price_Depth0"] - merged_df["Sell_Price_Depth0"]
        )
    else:
        # Full fill simulation
        def calculate_price_difference_per_row(row, vol):
            total_buy_vol = 0.0
            total_sell_vol = 0.0
            weighted_buy_price = 0.0
            weighted_sell_price = 0.0

            max_buy_vol = np.nansum([row.get(f"Buy_Size_Depth{d}", 0) for d in range(7)])
            max_sell_vol = np.nansum([row.get(f"Sell_Size_Depth{d}", 0) for d in range(7)])
            capped_vol = min(vol, max_buy_vol, max_sell_vol)
            
            # buy side
            remaining = capped_vol
            for d in range(7):
                buy_p = row.get(f"Buy_Price_Depth{d}", np.nan)
                buy_s = row.get(f"Buy_Size_Depth{d}", 0.0)
                if pd.isna(buy_p):
                    continue
                if remaining <= 0:
                    break
                used_vol = min(buy_s, remaining)
                weighted_buy_price += used_vol * buy_p
                total_buy_vol += used_vol
                remaining -= used_vol
            
            # sell side
            remaining = capped_vol
            for d in range(7):
                sell_p = row.get(f"Sell_Price_Depth{d}", np.nan)
                sell_s = row.get(f"Sell_Size_Depth{d}", 0.0)
                if pd.isna(sell_p):
                    continue
                if remaining <= 0:
                    break
                used_vol = min(sell_s, remaining)
                weighted_sell_price += used_vol * sell_p
                total_sell_vol += used_vol
                remaining -= used_vol
            
            if total_buy_vol > 0 and total_sell_vol > 0:
                avg_buy = weighted_buy_price / total_buy_vol
                avg_sell = weighted_sell_price / total_sell_vol
                return avg_buy - avg_sell
            else:
                return np.nan

        merged_df["Point_Diff"] = merged_df.apply(lambda r: calculate_price_difference_per_row(r, trade_vol), axis=1)

    # 3. Build Plots
    #    We'll label them with the entire date range
    from_date_str, to_date_str = date_range
    vol_label = "Top-of-Book" if top_of_book else f"Volume {trade_vol}"
    common_title = f"{symbol} [{from_date_str} to {to_date_str}] ({vol_label})"

    # --- 3.1 Main Over Time Plot (combined)
    plt.figure(figsize=(12, 6))
    plt.plot(
        merged_df["TimeRecorded"], 
        merged_df["Point_Diff"], 
        marker="o",
        linestyle="none",
        markersize=2,
        alpha=0.5
    )
    plt.title(f"Point Diff Over Time - {common_title}", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Point Diff", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    buf_main = io.BytesIO()
    plt.savefig(buf_main, format='png')
    buf_main.seek(0)
    main_time_plot = "data:image/png;base64," + base64.b64encode(buf_main.getvalue()).decode()
    plt.close()

    # --- 3.2 Distribution Plots (normal, log, and outlier highlight)
    distribution_plots = {}

    # Normal
    plt.figure(figsize=(10, 6))
    plt.hist(merged_df["Point_Diff"].dropna(), bins=50, edgecolor="k", alpha=0.7)
    plt.title(f"Distribution of Point_Diff - {common_title}", fontsize=16)
    plt.xlabel("Point_Diff", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    buf_dist = io.BytesIO()
    plt.savefig(buf_dist, format='png')
    buf_dist.seek(0)
    distribution_plots["hist_normal"] = "data:image/png;base64," + base64.b64encode(buf_dist.getvalue()).decode()
    plt.close()

    # Log
    plt.figure(figsize=(10, 6))
    plt.hist(merged_df["Point_Diff"].dropna(), bins=50, edgecolor="k", alpha=0.7, log=True)
    plt.title(f"Distribution of Point_Diff (Log Scale) - {common_title}", fontsize=16)
    plt.xlabel("Point_Diff", fontsize=14)
    plt.ylabel("Frequency (log scale)", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    buf_log = io.BytesIO()
    plt.savefig(buf_log, format='png')
    buf_log.seek(0)
    distribution_plots["hist_log"] = "data:image/png;base64," + base64.b64encode(buf_log.getvalue()).decode()
    plt.close()

    # Outlier
    Q1 = merged_df["Point_Diff"].quantile(0.25)
    Q3 = merged_df["Point_Diff"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = merged_df[(merged_df["Point_Diff"] < lower_bound) | (merged_df["Point_Diff"] > upper_bound)]

    plt.figure(figsize=(12, 7))
    sns.histplot(
        merged_df["Point_Diff"], 
        bins=50, 
        color='skyblue', 
        edgecolor='black', 
        label='Data', 
        alpha=0.7
    )
    if not outliers.empty:
        sns.histplot(
            outliers["Point_Diff"], 
            bins=50, 
            color='red', 
            edgecolor='black', 
            label='Outliers', 
            alpha=0.7
        )
    plt.axvline(lower_bound, color='green', linestyle='--', linewidth=2, label=f'Lower Bound ({lower_bound:.2f})')
    plt.axvline(upper_bound, color='purple', linestyle='--', linewidth=2, label=f'Upper Bound ({upper_bound:.2f})')
    plt.title(f"Histogram of Point_Diff (Outliers Highlighted) - {common_title}", fontsize=18)
    plt.xlabel("Point_Diff", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    buf_outlier = io.BytesIO()
    plt.savefig(buf_outlier, format='png')
    buf_outlier.seek(0)
    distribution_plots["outlier"] = "data:image/png;base64," + base64.b64encode(buf_outlier.getvalue()).decode()
    plt.close()

    # --- 3.3 Create Hourly Plots => { 'YYYY-MM-DD': { '0': base64, '1': base64, ...} }
    #     We consider each unique date in the data, then each hour for that date
    merged_df['Date'] = merged_df['TimeRecorded'].dt.date.astype(str)  # 'YYYY-MM-DD'
    merged_df['Hour'] = merged_df['TimeRecorded'].dt.hour

    hourly_plots = {}
    unique_dates = merged_df['Date'].unique()
    for d in unique_dates:
        daily_df = merged_df[merged_df['Date'] == d]
        hourly_plots[d] = {}
        for hr in sorted(daily_df['Hour'].unique()):
            hour_df = daily_df[daily_df['Hour'] == hr]

            # Plot
            plt.figure(figsize=(12, 4))
            plt.plot(
                hour_df["TimeRecorded"], 
                hour_df["Point_Diff"],
                marker="o",
                linestyle="none",
                markersize=3,
                alpha=0.7
            )
            plt.title(f"Point Diff Over Time - {symbol} {d} Hour {hr:02d} ({vol_label})", fontsize=14)
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Point Diff", fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            buf_hr = io.BytesIO()
            plt.savefig(buf_hr, format='png')
            buf_hr.seek(0)
            hr_plot = "data:image/png;base64," + base64.b64encode(buf_hr.getvalue()).decode()
            plt.close()

            hourly_plots[d][str(hr)] = hr_plot

    # --- 4. Statistics
    desc_default = merged_df['Point_Diff'].describe().to_dict()
    desc_custom = merged_df['Point_Diff'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()

    stats_specific = {
        "Mean": merged_df['Point_Diff'].mean(),
        "Median": merged_df['Point_Diff'].median(),
        "Std": merged_df['Point_Diff'].std(),
        "Var": merged_df['Point_Diff'].var(),
        "Min": merged_df['Point_Diff'].min(),
        "Max": merged_df['Point_Diff'].max(),
        "Skewness": merged_df['Point_Diff'].skew(),
        "Kurtosis": merged_df['Point_Diff'].kurt()
    }

    outlier_info = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Number of Outliers": len(outliers)
    }

    statistics = {
        "describe_default": desc_default,
        "describe_custom": desc_custom,
        "specific_stats": stats_specific,
        "outlier_info": outlier_info
    }

    return {
        "main_time_plot": main_time_plot,
        "distribution_plots": distribution_plots,  # dict with hist_normal, hist_log, outlier
        "hourly_plots": hourly_plots,              # nested dict of { date_str: { hour: plot_url } }
        "statistics": statistics
    }