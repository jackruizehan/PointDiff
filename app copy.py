# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
from PointSpread import get_quote_data, PointSpreadDisplay

app = Flask(__name__)

# Home route
@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template

# Route to fetch and process data
@app.route('/get_data', methods=['POST'])
def get_data():
    try:
        # 1. Parse Form Inputs
        from_date_str = request.form.get('from_date')  # start date string (YYYY-MM-DD)
        to_date_str = request.form.get('to_date')      # end date string   (YYYY-MM-DD)
        symbol = request.form.get('symbol')            # from the dropdown
        maker_id = request.form.get('maker_id')        # from the dropdown
        
        # We are using a radio or logic that ensures only one is chosen:
        #   top_of_book='on' means top-of-book
        #   Otherwise user might have typed a volume
        top_of_book_str = request.form.get('top_of_book', 'off')  # 'on' or 'off'
        top_of_book = (top_of_book_str == 'on')
        
        trade_vol_str = request.form.get('trade_vol', '0')
        trade_vol = int(trade_vol_str) if trade_vol_str else 0
        
        # 2. Enforce "volume only for XAU/USD; otherwise top-of-book must be True"
        if symbol != 'XAU/USD':
            # Force top_of_book to True if symbol is not XAU/USD
            top_of_book = True
            trade_vol = 0
        
        # 3. Validate the date range. Make sure from_date <= to_date, and max 7 days
        if not from_date_str or not to_date_str:
            return jsonify({"status": "error", "message": "Please select both from_date and to_date."})
        
        from_date = pd.to_datetime(from_date_str)
        to_date = pd.to_datetime(to_date_str)
        if from_date > to_date:
            return jsonify({"status": "error", "message": "From date cannot be after To date."})
        
        # Check maximum 7-day range
        if (to_date - from_date).days > 7:
            return jsonify({"status": "error", "message": "Date range cannot exceed 7 days."})
        
        # 4. Collect Data for Each Day in the Range
        all_data_frames = []
        current_date = from_date
        while current_date <= to_date:
            date_str = current_date.strftime("%Y-%m-%d")
            # Fetch data for that single day
            df_day = get_quote_data(date_str, symbol)
            if df_day is not None and not df_day.empty:
                all_data_frames.append(df_day)
            current_date += pd.Timedelta(days=1)
        
        # Combine everything into a single DataFrame
        if not all_data_frames:
            return jsonify({"status": "error", "message": "No data found for selected date range."})
        
        df_combined = pd.concat(all_data_frames, ignore_index=True)
        
        # 5. Generate Plots and Statistics
        #    Pass the combined DataFrame but also the date range (for display) if needed
        result = PointSpreadDisplay(
            df_input=df_combined,
            trade_vol=trade_vol,
            date_range=(from_date_str, to_date_str),
            maker_id=maker_id,
            top_of_book=top_of_book,
            symbol=symbol
        )

        # 6. Return results as JSON
        return jsonify({
            "status": "success",
            "main_time_plot": result.get("main_time_plot"),
            "distribution_plots": result.get("distribution_plots"),  # normal + log + outliers
            "hourly_plots": result.get("hourly_plots"),  # nested dict of { date_str: { hour: plot_url } }
            "statistics": result["statistics"]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Set debug=False in production
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8000, debug=True)