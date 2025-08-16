# Iris Flower Classification API

This project is a simple web API to predict the type of Iris flower using a trained machine learning model.

## How to use

1. Make sure you have `model.pkl` in the same folder.
2. Install Python packages (dependencies):
   pip install -r requirements.txt
3. Start the API server:
   uvicorn main:app --reload
4. Open your browser and go to:
   http://127.0.0.1:8000/docs
5. Test the API by entering the flower measurements:
   - sepal_length
   - sepal_width
   - petal_length
   - petal_width
6. The API will return the predicted flower type:
   - setosa
   - versicolor
   - virginica

## Files in this project

- main.py → FastAPI code
- model.pkl → Trained ML model
- requirements.txt → Needed Python packages
- README.md → This instructions file
