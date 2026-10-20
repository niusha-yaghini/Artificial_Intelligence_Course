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
- train.py: Receives data, trains the model, evaluates it, and saves the output files.
- predict.py: Loads the saved model and generates predictions for new input.
- 01_end_to_end_ml_pipeline.ipynb: Serves as our training and experimentation environment.

This separation is important, as the final service should not have to retrain the model from scratch every time it receives a request.