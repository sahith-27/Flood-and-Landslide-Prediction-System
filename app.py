"""Web app — Flood & Landslide Predictor
Uses CNN+RNN (live NASA satellite images) + Weather RF for prediction.
"""
import sys
import flask
from flask import Flask, render_template, request
import pickle
import base64

sys.path.insert(0, 'training')
from training import prediction as pred_module
from merged_pipeline import MergedCNNLSTMRF, CNN2D, LSTMCell

app = flask.Flask(__name__)

data   = [{'name': 'Delhi',     'sel': 'selected'}, {'name': 'Mumbai',    'sel': ''},
          {'name': 'Kolkata',   'sel': ''},          {'name': 'Bangalore', 'sel': ''},
          {'name': 'Chennai',   'sel': ''}]
months = [{'name': 'May',  'sel': ''}, {'name': 'June', 'sel': ''},
          {'name': 'July', 'sel': 'selected'}]
cities = [
    {'name': 'Delhi',       'sel': 'selected'}, {'name': 'Mumbai',      'sel': ''},
    {'name': 'Kolkata',     'sel': ''},          {'name': 'Bangalore',   'sel': ''},
    {'name': 'Chennai',     'sel': ''},          {'name': 'New York',    'sel': ''},
    {'name': 'Los Angeles', 'sel': ''},          {'name': 'London',      'sel': ''},
    {'name': 'Paris',       'sel': ''},          {'name': 'Sydney',      'sel': ''},
    {'name': 'Beijing',     'sel': ''},
]

model = pickle.load(open("model.pickle", "rb"))


# ─────────────────────────────────────────────────────────────────────────────
# Static pages
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/plots.html")
def plots():
    return render_template("plots.html")

@app.route("/heatmaps.html")
def heatmaps():
    return render_template("heatmaps.html")

@app.route("/chart.html")
def chart():
    return render_template("chart.html")


# ─────────────────────────────────────────────────────────────────────────────
# Satellite viewer
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/satellite.html", methods=["GET", "POST"])
def satelliteimages():
    place  = request.form.get("place", "Delhi")
    date   = request.form.get("date",  "July")
    data_l = [{'name': 'Delhi', 'sel': ''}, {'name': 'Mumbai', 'sel': ''},
              {'name': 'Kolkata', 'sel': ''}, {'name': 'Bangalore', 'sel': ''},
              {'name': 'Chennai', 'sel': ''}]
    months_l = [{'name': 'May', 'sel': ''}, {'name': 'June', 'sel': ''},
                {'name': 'July', 'sel': ''}]
    for item in data_l:
        if item["name"] == place: item["sel"] = "selected"
    for item in months_l:
        if item["name"] == date:  item["sel"] = "selected"
    direc = "satellite_images/{}_{}.png".format(place, date)
    with open(direc, "rb") as f:
        image = base64.b64encode(f.read()).decode("utf-8")
    return render_template("satellite.html", data=data_l, image_file=image,
                           months=months_l, text=f"{place} in {date} 2024")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction — now uses CNN+RNN + live satellite images
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predicts.html", methods=["GET", "POST"])
def get_predicts():
    if request.method == "GET":
        return render_template("predicts.html", cities=cities,
                               cityname="Enter a city to get a prediction")

    try:
        cityname = request.form["city"].strip()
        if not cityname:
            raise ValueError("Empty city name")

        cities_local = [{'name': c['name'], 'sel': 'selected' if c['name'] == cityname else ''}
                        for c in cities]

        # ── Run full CNN+RNN + Weather RF pipeline ─────────────────────────
        result = pred_module.get_full_prediction(cityname, model)

        wf   = result["weather_features"]
        temp_c = round((wf[0] - 32) * 5/9, 1)
        maxt_c = round((wf[1] - 32) * 5/9, 1)

        return render_template(
            "predicts.html",
            # City
            cityname        = f"Prediction for {cityname}",
            cities          = cities_local,
            # Prediction
            pred            = result["pred"],
            confidence      = result["confidence"],
            method          = result["method"],
            flood_score     = result["flood_score"],
            landslide_score = result["landslide_score"],
            cnn_rnn_prob    = result["cnn_rnn_prob"],
            weather_prob    = result["weather_prob"],
            # Weather
            temp            = round(wf[0], 1),
            maxt            = round(wf[1], 1),
            temp_c          = temp_c,
            maxt_c          = maxt_c,
            wspd            = round(wf[2], 1),
            cloudcover      = round(wf[3], 1),
            percip          = round(wf[4], 1),
            humidity        = round(wf[5], 1),
            # Satellite images (base64 JPEG strings, empty string = not available)
            sat_available   = result["sat_available"],
            sat_img_1       = result["satellite_b64_imgs"][0],
            sat_img_2       = result["satellite_b64_imgs"][1],
            sat_img_3       = result["satellite_b64_imgs"][2],
            sat_date_1      = result["satellite_dates"][0],
            sat_date_2      = result["satellite_dates"][1],
            sat_date_3      = result["satellite_dates"][2],
        )

    except Exception as e:
        print(f"Error in get_predicts: {e}")
        return render_template("predicts.html", cities=cities,
                               cityname="Oops — could not retrieve data for that city.")


if __name__ == "__main__":
    app.run(debug=True)