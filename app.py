from flask import Flask, render_template, request, jsonify
import numpy as np
from main import score_predict
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    batting_team = request.form.get("batting_team")
    bowling_team = request.form.get("bowling_team")
    overs = float(request.form.get("overs"))
    runs = int(request.form.get("runs"))
    wickets = int(request.form.get("wickets"))
    runs_last_5 = int(request.form.get("runs_last_5"))
    wickets_last_5 = int(request.form.get("wickets_last_5"))
    score = score_predict("Royal Challengers Bangalore", "Mumbai Indians", 62, 1, 6, 43, 2)
    print(score)
    return render_template("predict.html", predicted_score=score)

if __name__ == "__main__":
    app.run(debug=True)

