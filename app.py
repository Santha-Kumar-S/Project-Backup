from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import uuid
from toneAnalyser import ToneAnalyser

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/asset/tone_test/"

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predictPage')
def predictPage():
    return render_template('index.html',result='')

@app.route('/backToHome')
def backToHome():
    return render_template('home.html',result='') 


@app.route('/skin_tone_predict', methods=["POST"])
def skin_tone_predict():
    try:
        if request.method == 'POST':
            file = request.files['image']
            if file.filename != '':
                filename = f"{str(uuid.uuid4())}.jpg"
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                analyzer = ToneAnalyser()
                model_output = analyzer.analyse_skin_tone(os.path.join(app.config["UPLOAD_FOLDER"], filename), "jpg")
                result = {
                    'imageFile': filename,
                    'label': model_output["label"],
                    'accuracy': model_output["accuracy"],
                    'skin_tone': model_output["skin_tone"],
                    'dominant_colors': model_output["dominant_colors"],
                }
                return render_template('index.html', result=result)
            else:
                return render_template('index.html', result='No file uploaded')
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', result='An error occurred during prediction')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
