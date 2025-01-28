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
        # Get form data
        date = request.form.get('date')  # string (YYYY-MM-DD)
        symbol = request.form.get('symbol')  # from the select
        maker_id = request.form.get('maker_id')  # from the select
        top_of_book = request.form.get('top_of_book') == 'on'  # True if checked

        # If not top_of_book, parse trade volume. Otherwise, pass None
        if top_of_book:
            trade_vol = None
        else:
            trade_vol = int(request.form.get('trade_vol') or 0)

        # Fetch data
        df = get_quote_data(date, symbol)
        if df is None or df.empty:
            return jsonify({"status": "error", "message": "No data found!"})

        # Generate plots and statistics
        result = PointSpreadDisplay(
            df_input=df,
            trade_vol=trade_vol,
            date=date,
            maker_id=maker_id,
            top_of_book=top_of_book,
            symbol=symbol
        )

        return jsonify({
            "status": "success",
            "plot1_url": result.get("plot1_url"),
            "plot2_url": result.get("plot2_url"),
            "plot3_url": result.get("plot3_url"),
            "plot4_url": result.get("plot4_url"),
            "plot5_url": result.get("plot5_url"),
            "statistics": result["statistics"]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)