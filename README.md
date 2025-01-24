# Quote Data Analysis Project

This project contains Jupyter notebooks designed to query, process, and analyze quote data for a specific symbol over a given date range. The analysis focuses on two key metrics: Top of Book (TOB) point spread and Volume-Weighted Average Fill Price point spread. 

## Files in the Project

### 1. `DataLoading.ipynb`
- **Functionality**: Queries quote data for a specific symbol and day from an AWS MySQL database and saves it locally in the `Data/` folder.
- **Notes**: The `Data/` folder is excluded from the repository via `.gitignore`.

### 2. `DataAnalysis.ipynb`
- **Functionality**: 
  - Loads data from the `Data/` folder.
  - Performs analysis of the Top of Book (TOB) point spread for a specific symbol and date range.
  - Supports analysis over a single day or multiple days (e.g., a week) by reading the relevant data.
  - Includes visualization and a summary of the TOB point spread.

### 3. `DataAnalysisDepth.ipynb`
- **Functionality**: 
  - Loads data from the `Data/` folder.
  - Performs analysis of the Volume-Weighted Average Fill Price point spread for a specific symbol and date range.
  - Supports analysis over a single day or multiple days (e.g., a week) by reading the relevant data.
  - Includes visualization and a summary of the Volume-Weighted Average Fill Price point spread.

## Usage Instructions

1. **Data Querying**:
   - Run `DataLoading.ipynb` to fetch relevant quote data for the desired symbols and dates.
   - Ensure the fetched data is saved in the `Data/` folder.

2. **Top of Book (TOB) Point Spread Analysis**:
   - Use `DataAnalysis.ipynb` to load data from the `Data/` folder.
   - Perform analysis, visualization, and summarization of the TOB point spread.

3. **Volume-Weighted Average Fill Price Point Spread Analysis**:
   - Use `DataAnalysisDepth.ipynb` to load data from the `Data/` folder.
   - Perform analysis, visualization, and summarization of the Volume-Weighted Average Fill Price point spread.

## Requirements

- AWS MySQL database access for data querying.
- Python and Jupyter Notebook environment.
- Necessary libraries (e.g., pandas, matplotlib) installed.

## Notes

- Make sure to run `DataLoading.ipynb` first to fetch the required data before proceeding with analysis.
- Ensure the `Data/` folder is correctly structured for the analysis notebooks to work.
