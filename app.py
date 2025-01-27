# app.py

from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Ensure non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
from PointSpread import get_quote_data, PointSpreadDisplay  # Ensure correct import

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
        date = request.form.get('date')
        symbol = request.form.get('symbol')
        trade_vol = int(request.form.get('trade_vol'))
        maker_id = request.form.get('maker_id')

        # Fetch the data using the Python function
        df = get_quote_data(date, symbol)

        if df is None or df.empty:
            return jsonify({"status": "error", "message": "No data found!"})

        # Generate plot and statistics
        result = PointSpreadDisplay(df, trade_vol, date, maker_id)

        return jsonify({
            "status": "success",
            "plot1_url": result["plot1_url"],
            "plot2_url": result["plot2_url"],
            "statistics": result["statistics"]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)