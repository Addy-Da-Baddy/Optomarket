# Optomarket - Customer Segmentation and Analysis Tool

![Optomarket Logo](logo.jpg/468x60?text=Optomarket+Logo)

## Overview

Optomarket is a cutting-edge customer segmentation and analysis tool designed for market analysts and business intelligence professionals. The application uses K-Means clustering to segment customer data and provides detailed insights and recommendations using advanced AI integration. Built with Streamlit, this tool offers interactive data visualization and analysis features, helping businesses understand customer segments and optimize marketing strategies.

## Features

- **Data Upload and Preview**: Easily upload and preview your customer data in CSV format.
- **Data Cleaning and Processing**: Automatically handles missing values, calculates age from dates of birth, and prepares data

## Overview

Optomarket is a customer segmentation and analysis tool designed for market analysts and business intelligence professionals. The application uses K-Means clustering to segment customer data and provides detailed insights and recommendations using advanced AI integration. Built with Streamlit, this tool offers interactive data visualization and analysis features, helping businesses understand customer segments and optimize marketing strategies.

## Features

- **Data Upload and Preview**: Easily upload and preview your customer data in CSV format.
- **Data Cleaning and Processing**: Automatically handles missing values, calculates age from dates of birth, and prepares data for clustering.
- **Clustering and Analysis**: Applies clustering to segment customers into distinct groups.
- **Visualization**: Generates interactive plots including scatter plots, box plots, histograms, and PCA plots to visualize clustering results.
- **Segment Analysis**: Integrates with the Gemini API to provide professional titles, detailed analyses, and tailored marketing recommendations for each customer segment.
- **Download Options**: Download segmented data and segment profiles for further analysis.

## Requirements

- Python 3.7 or higher
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Generative AI Client Library

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Addy-Da-Baddy/optomarket.git
   cd optomarket
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the required Python libraries. You may need to install the Google Generative AI Client Library and other dependencies if not already available.

## Configuration

### API Key:
Replace the placeholder API key with your actual Gemini API key in the code:

```python
api_key = 'YOUR_GEMINI_API_KEY'  # Replace with your actual Gemini API key
genai.configure(api_key=api_key)
```

### Logo:
Place your logo image (`logo.jpg`) in the same directory as the Streamlit app script for display.

## Running the Application

1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
   Replace `app.py` with the name of your Python script containing the Streamlit app code.

2. Access the Application:
   Open your web browser and go to `http://localhost:8501` to view the application.

## Usage

1. **Upload Data**: Use the sidebar to upload your CSV file. The data preview will be displayed on the main page.
2. **Process Data**: Click the "Process Data" button to start data cleaning, processing, and clustering.
3. **View Results**: Use the buttons to navigate between graphical analysis, recommendations, and segmented data views.
4. **Download Data**: Download segment profiles and segmented data using the provided download buttons.

## Troubleshooting

- **Quota Errors**: Ensure you have a valid API key and sufficient quota for the Gemini API. Check the API usage limits if you encounter errors.
- **File Upload Issues**: Ensure your CSV file format matches the expected structure. Handle missing values and column names as described in the application.
