# PointSpread.py
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
import psutil  # Added for memory monitoring

# ------------------------------------------------
# 1) Lower memory limit to 12GB (example)
# ------------------------------------------------
MEMORY_LIMIT = 12 * 1024 ** 3  # 12 GB

def check_memory_usage():
    """
    Check the current process memory usage.
    Raises MemoryError if usage exceeds MEMORY_LIMIT.
    """
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss  # in bytes
    if mem_usage > MEMORY_LIMIT:
        raise MemoryError(f"Memory usage exceeded: {mem_usage / (1024 ** 3):.2f} GB")

def get_quote_data(date_from, date_to, symbol, use_ssh=False):
    """
    Fetch quote data for a specific date range and symbol from Alp_Quotes or local storage.
    If local .pkl files exist, load from them. Otherwise, fetch from DB (optionally via SSH).
    """
    try:
        # -------------------------------
        # 1. Transform symbol for SQL query and local file
        # -------------------------------
        symbol_transformed = symbol.replace('/', '')
        print(f"Transformed symbol: {symbol_transformed}")

        # -------------------------------
        # 2. Generate list of dates
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
            check_memory_usage()  # Memory check
            file_path = os.path.join('Data', f"{symbol_transformed}_{date_str}.pkl")
            if not os.path.exists(file_path):
                print(f"Local file missing: {file_path}")
                local_data_available = False
                break
            else:
                print(f"Loading local file: {file_path}")
                try:
                    df_local = pd.read_pickle(file_path)
                    check_memory_usage()  # Memory check
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
                check_memory_usage()  # Memory check
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
        # 4. Determine relevant partitions
        # -------------------------------
        month_map = {
            1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
            7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"
        }

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
        # 6. DB Connection Info
        # -------------------------------
        ssh_host = '18.133.184.11'
        ssh_user = 'ubuntu'
        ssh_key_file = '/Users/jackhan/Desktop/Alpfin/OneZero_Data.pem'

        db_host_direct = '127.0.0.1'  # Replace with your actual DB host
        db_host_via_ssh = '127.0.0.1'
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
        # 8. Connect to DB & fetch
        # -------------------------------
        if use_ssh:
            # SSH approach
            with SSHTunnelForwarder(
                (ssh_host, 22),
                ssh_username=ssh_user,
                ssh_pkey=ssh_key_file,
                remote_bind_address=(db_host_via_ssh, db_port),
                allow_agent=False,
                host_pkey_directories=[],
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
                    dataframes = []
                    for q in queries:
                        check_memory_usage()
                        print(f"Executing query:\n{q}")
                        cursor.execute(q)
                        rows = cursor.fetchall()
                        print(f"Fetched {len(rows)} rows from partition.")
                        df = pd.DataFrame(rows, columns=columns)
                        check_memory_usage()
                        dataframes.append(df)
                    
                    if dataframes:
                        result_df = pd.concat(dataframes, ignore_index=True)
                        check_memory_usage()

                        # # --------------------------------
                        # #  Add row sampling if large
                        # # --------------------------------
                        # MAX_ROWS = 1_000_000
                        # if len(result_df) > MAX_ROWS:
                        #     result_df = result_df.sample(n=MAX_ROWS, random_state=42)
                        #     print(f"DataFrame was sampled to {MAX_ROWS} rows to reduce memory usage.")

                        print(f"All data fetched. Total rows: {len(result_df)}")
                    else:
                        result_df = pd.DataFrame(columns=columns)
                        print("No data fetched from database.")
                    return result_df
                finally:
                    cursor.close()
                    connection.close()
                    print("Database connection via SSH closed.")
        else:
            # Direct connection
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
                dataframes = []
                for q in queries:
                    check_memory_usage()
                    print(f"Executing query:\n{q}")
                    cursor.execute(q)
                    rows = cursor.fetchall()
                    print(f"Fetched {len(rows)} rows from partition.")
                    df = pd.DataFrame(rows, columns=columns)
                    check_memory_usage()
                    dataframes.append(df)
                
                if dataframes:
                    result_df = pd.concat(dataframes, ignore_index=True)
                    check_memory_usage()

                    # # --------------------------------
                    # #  Add row sampling if large
                    # # --------------------------------
                    # MAX_ROWS = 1_000_000
                    # if len(result_df) > MAX_ROWS:
                    #     result_df = result_df.sample(n=MAX_ROWS, random_state=42)
                    #     print(f"DataFrame was sampled to {MAX_ROWS} rows to reduce memory usage.")

                    print(f"All data fetched. Total rows: {len(result_df)}")
                else:
                    result_df = pd.DataFrame(columns=columns)
                    print("No data fetched from database.")
                
                return result_df
            finally:
                cursor.close()
                connection.close()
                print("Direct database connection closed.")

    except MemoryError as mem_err:
        print(f"MemoryError: {str(mem_err)}")
        raise mem_err  # re-raise
    except Exception as e:
        print(f"ERROR fetching data for {symbol} from {date_from} to {date_to}: {str(e)}")
        return None

def PointSpreadDisplay(df_input, trade_vol, date_range, maker_id="Britannia", top_of_book=True, symbol="XAU/USD"):
    """
    Build plots and statistics for the point spread. Returns a dict with:
      "main_time_plot", "distribution_plots", "hourly_plots", "statistics".
    """
    try:
        check_memory_usage()
        df_loaded = df_input[df_input['MakerId'] == maker_id]
        check_memory_usage()

        # Convert TimeRecorded to datetime
        if not pd.api.types.is_datetime64_any_dtype(df_loaded['TimeRecorded']):
            df_loaded['TimeRecorded'] = pd.to_datetime(df_loaded['TimeRecorded'])
            check_memory_usage()

        df_loaded.sort_values(by="TimeRecorded", inplace=True)
        check_memory_usage()

        # Merge Depth 0..6 (sell & buy) side by side
        depth_dfs = {}
        for depth in range(7):
            check_memory_usage()
            # Sell side => Side=0
            sell_df = df_loaded[(df_loaded["Side"] == 0) & (df_loaded["Depth"] == depth)]
            sell_df = sell_df.rename(
                columns={
                    "Price": f"Sell_Price_Depth{depth}", 
                    "Size": f"Sell_Size_Depth{depth}"
                }
            )
            sell_df = sell_df[["CoreSymbol", "TimeRecorded", f"Sell_Price_Depth{depth}", f"Sell_Size_Depth{depth}"]]

            depth_dfs[f'sell_df_depth{depth}'] = sell_df

            # Buy side => Side=1
            buy_df = df_loaded[(df_loaded["Side"] == 1) & (df_loaded["Depth"] == depth)]
            buy_df = buy_df.rename(
                columns={
                    "Price": f"Buy_Price_Depth{depth}", 
                    "Size": f"Buy_Size_Depth{depth}"
                }
            )
            buy_df = buy_df[["CoreSymbol", "TimeRecorded", f"Buy_Price_Depth{depth}", f"Buy_Size_Depth{depth}"]]
            depth_dfs[f'buy_df_depth{depth}'] = buy_df

        merged_df = depth_dfs['sell_df_depth0']
        for depth in range(1, 7):
            check_memory_usage()
            merged_df = merged_df.merge(depth_dfs[f'sell_df_depth{depth}'], on=["CoreSymbol", "TimeRecorded"], how="outer")

        for depth in range(7):
            check_memory_usage()
            merged_df = merged_df.merge(depth_dfs[f'buy_df_depth{depth}'], on=["CoreSymbol", "TimeRecorded"], how="outer")

        merged_df.sort_values(by="TimeRecorded", inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        check_memory_usage()

        # Calculate "Point_Diff"
        if top_of_book:
            # Depth0 only
            merged_df["Point_Diff"] = merged_df["Buy_Price_Depth0"] - merged_df["Sell_Price_Depth0"]
        else:
            # Volume fill simulation
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
        
        check_memory_usage()

        # Build Plots
        from_date_str, to_date_str = date_range
        vol_label = "Top-of-Book" if top_of_book else f"Volume {trade_vol}"
        common_title = f"{symbol} [{from_date_str} to {to_date_str}] ({vol_label})"

        # 3.1 Main Over Time Plot
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
        check_memory_usage()

        # 3.2 Distribution Plots
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
        check_memory_usage()

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
        check_memory_usage()

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
        plt.axvline(lower_bound, color='green', linestyle='--', linewidth=2, 
                    label=f'Lower Bound ({lower_bound:.2f})')
        plt.axvline(upper_bound, color='purple', linestyle='--', linewidth=2, 
                    label=f'Upper Bound ({upper_bound:.2f})')
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
        check_memory_usage()

        # 3.3 Hourly Plots
        merged_df['Date'] = merged_df['TimeRecorded'].dt.date.astype(str)
        merged_df['Hour'] = merged_df['TimeRecorded'].dt.hour

        hourly_plots = {}
        unique_dates = merged_df['Date'].unique()
        for d in unique_dates:
            check_memory_usage()
            daily_df = merged_df[merged_df['Date'] == d]
            hourly_plots[d] = {}
            for hr in sorted(daily_df['Hour'].unique()):
                check_memory_usage()
                hour_df = daily_df[daily_df['Hour'] == hr]

                plt.figure(figsize=(12, 4))
                plt.plot(
                    hour_df["TimeRecorded"], 
                    hour_df["Point_Diff"],
                    marker="o",
                    linestyle="none",
                    markersize=3,
                    alpha=0.7
                )
                plt.title(f"Point Diff - {symbol} {d} Hour {hr:02d} ({vol_label})", fontsize=14)
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
                check_memory_usage()

                hourly_plots[d][str(hr)] = hr_plot

        # 4. Statistics
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

        check_memory_usage()

        return {
            "main_time_plot": main_time_plot,
            "distribution_plots": distribution_plots,
            "hourly_plots": hourly_plots,
            "statistics": statistics
        }
    
    except MemoryError as mem_err:
        print(f"MemoryError: {str(mem_err)}")
        raise mem_err
    except Exception as e:
        print(f"ERROR in PointSpreadDisplay: {str(e)}")
        return None  # Fix: was 'Nones' before