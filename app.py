from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("burnout_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        typing_speed = int(request.form["typing_speed"])
        study_hours = int(request.form["study_hours"])
        breaks = int(request.form["breaks"])
        active_hours = int(request.form["active_hours"])

        prediction = model.predict(
            np.array([[typing_speed, study_hours, breaks, active_hours]])
        )[0]

        if prediction == 0:
            result = "Low Burnout Risk 🟢"
        elif prediction == 1:
            result = "Medium Burnout Risk 🟡"
        else:
            result = "High Burnout Risk 🔴"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)