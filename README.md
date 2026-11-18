# 🌊 Flood and Landslide Prediction System Using Machine Learning

## 📌 Project Overview

The **Flood and Landslide Prediction System** is a machine learning-based web application designed to predict potential flood and landslide risks using environmental and weather-related data.

The application uses a Flask web interface integrated with machine learning models to process input data and generate risk predictions. It also provides interactive visualizations and satellite imagery to help users understand environmental conditions.

## 🎯 Objectives

- Predict potential flood and landslide risks using machine learning.
- Analyze environmental factors such as rainfall, humidity and terrain information.
- Visualize prediction results through interactive charts and heatmaps.
- Display satellite images for environmental monitoring.
- Provide a simple and user-friendly web interface.

## 🚀 Features

- Flood risk prediction.
- Landslide risk prediction.
- Machine learning model integration.
- Interactive data visualization using Plotly.
- Satellite image visualization.
- Weather and environmental data processing.
- Flask-based web application.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Machine Learning | CNN, LSTM, Random Forest |
| Frontend | HTML, CSS, JavaScript, Bootstrap |
| Visualization | Plotly |
| Data Processing | Pandas, NumPy |
| Dataset | CSV datasets and satellite images |
| Tools | VS Code, Git, GitHub |

## 📂 Project Structure

```text
Flood-and-Landslide-Prediction-System/
│
├── app.py
├── model.pickle
├── requirements.txt
│
├── dataset_city_stats.csv
├── dataset_features.csv
├── dataset_summary.csv
│
├── satellite_images/
│
├── static/
│   ├── css/
│   ├── js/
│   ├── img/
│   └── lib/
│
├── templates/
│   ├── index.html
│   ├── plots.html
│   ├── heatmaps.html
│   ├── predicts.html
│   ├── satellite.html
│   └── chart.html
│
└── training/
    ├── cnn_lstm_rf.py
    ├── cnn_rnn_model.py
    ├── cnn_rnn_rf_model.py
    ├── merged_pipeline.py
    └── prediction.py
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sahith-27/Flood-and-Landslide-Prediction-System.git
```

### 2. Navigate to the Project Folder

```bash
cd Flood-and-Landslide-Prediction-System
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Required Packages

```bash
pip install -r requirements.txt
```

### 6. Run the Flask Application

```bash
python app.py
```

### 7. Open in Browser

```text
http://127.0.0.1:5000/
```

## 🧠 Machine Learning

The project uses a hybrid machine learning approach involving:

- **CNN (Convolutional Neural Network):** Used for extracting relevant features from image data.
- **LSTM (Long Short-Term Memory):** Used for learning sequential patterns in data.
- **Random Forest:** Used for prediction and classification.

The models are integrated into the application to support flood and landslide risk prediction.

## 📊 Project Output

The application provides:

- Flood and landslide prediction results.
- Interactive plots and visualizations.
- Environmental heatmaps.
- Satellite image displays.
- A web-based prediction interface.

## 🔮 Future Enhancements

- Integrate real-time weather API data.
- Improve prediction accuracy using larger datasets.
- Deploy the application on a cloud platform.
- Add location-based alerts for high-risk areas.
- Improve the user interface and prediction reporting.

## 👨‍💻 Developer

**Kukkala Sahith**

- GitHub: https://github.com/sahith-27
- Project Repository: https://github.com/sahith-27/Flood-and-Landslide-Prediction-System

---

⭐ If you find this project useful, consider giving the repository a star!

**Note:** This project is developed for educational and research purposes. Predictions should not be treated as official disaster warnings.
