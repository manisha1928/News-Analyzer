# from flask import Flask, request, render_template
# import joblib

# app = Flask(__name__)

# # Load the trained model and vectorizer
# model = joblib.load('model.pkl')  # Ensure this path is correct
# vectorizer = joblib.load('vectorizer.pkl')  # Ensure this path is correct

# @app.route('/')
# def home():
#     return render_template('index.html')  # Render the homepage

# @app.route('/predict', methods=['POST'])
# def predict():
#     news_input = request.form['news_input']  # Get the news input from the form
    
#     # Vectorize the input
#     vectorized_text = vectorizer.transform([news_input])  
    
#     # Make prediction
#     prediction = model.predict(vectorized_text)[0]
    
#     # Map the prediction to a readable format
#     if prediction == 1:
#         prediction_label = "Real"
#     else:
#         prediction_label = "Fake"
    
#     return render_template('result.html', news_input=news_input, prediction=prediction_label)  # Show prediction result
# # except Exception as e:
# # return render_template('index.html', prediction=f"Error: {str(e)}")  # Show error message

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('model.pkl')  # Ensure this path is correct
vectorizer = joblib.load('vectorizer.pkl')  # Ensure this path is correct

@app.route('/')
def home():
    return render_template('index.html')  # Render the homepage

@app.route('/predict', methods=['POST'])
def predict():
    news_input = request.form['news_input']  # Get the news input from the form
    
    try:
        # Vectorize the input
        vectorized_text = vectorizer.transform([news_input])  
        
        # Make prediction
        prediction = model.predict(vectorized_text)[0]
        
        # Map the prediction to a readable format
        prediction_label = "Real" if prediction == 1 else "Fake"
    
        return render_template('result.html', news_input=news_input, prediction=prediction_label)  # Show prediction result
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")  # Show error message

if __name__ == '__main__':
    app.run(debug=True)

