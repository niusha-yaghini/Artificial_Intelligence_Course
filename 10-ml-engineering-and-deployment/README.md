10-ml-engineering-and-deployment/
├── 01_end_to_end_ml_pipeline.ipynb
├── src/
│   ├── train.py
│   ├── predict.py
│   └── api.py
├── tests/
│   ├── test_prediction.py
│   └── test_api.py
├── artifacts/
│   └── .gitkeep
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md


In our project, we separate responsibilities:
- train.py: Training the model and saving it
- predict.py: Loading the model, validating input, and performing predictions
- api.py: Receiving HTTP requests, calling the prediction function, and returning the response
- 01_end_to_end_ml_pipeline.ipynb: Serves as our training and experimentation environment.

This separation is important, as the final service should not have to retrain the model from scratch every time it receives a request.

------------------------------------

Workflow:

1. Model Training — `train.py`

We loaded the `digits` dataset, split the data into training and testing sets, created a pipeline consisting of `StandardScaler` and `LogisticRegression`, and trained the model. Finally, we saved the model to a file named `digits_pipeline.joblib`. `train.py`

2. Building the API — `api.py`

We instantiated the application object using `FastAPI()` and loaded the saved model using `load_model()`. We also defined the `/`, `/health`, and `/predict` endpoints. `api.py`

3. Viewing Documentation — Swagger UI

By accessing `/docs`, we accessed the API's interactive interface, which allowed us to send requests directly from the browser.

4. Sending a Request via `httpx`

Using a Python command, we sent 64 pixel values—formatted as JSON—to `http://127.0.0.1:8000/predict`. This request was transmitted to the running API via HTTP, rather than simply calling the model function directly from a script.

5. Receiving the Response

The server returned a `200` status code, and the model predicted the digit `0` with a confidence score of approximately `0.999662...`. This indicates that the request was processed successfully and the model predicted the digit zero.

-----------------------------------------

Commands for the command line (CMD):

python -m pytest -v

This runs the test file.



python -m uvicorn src.api:app --reload

This launches FastAPI.
You can access the Swagger UI via the URL provided.



python -c "from sklearn.datasets import load_digits; import json; print(json.dumps({'features': load_digits().data[0].tolist()}))"

A command to generate a sample data point.



python -c "from sklearn.datasets import load_digits; import httpx; features=load_digits().data[0].tolist(); response=httpx.post('http://127.0.0.1:8000/predict', json={'features': features}); print('HTTP status:', response.status_code); print('Response:', response.json())"

A command to send a request directly via the API.