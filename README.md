#Lendable Take-Home Assignment

##Overview

This repository contains the solution for the Early Loan Settlement Prediction task. It includes a full data pipeline from preprocessing and model training in Jupyter Notebook to deployment using Docker.

---

##Project Structure

lendable-docker-api/
├── main.py                # FastAPI app for inference
├── Dockerfile             # Containerization setup
├── requirements.txt       # Dependencies
├── Untitled.ipynb         # Jupyter Notebook with EDA, preprocessing & modeling
└── README.md              # Documentation

---

##How to Run the API (Locally with Docker)

(https://www.docker.com/products/docker-desktop/) is installed and running.

1. **Build the Docker image**:

```bash
docker build -t lendable-api .

2. **Run the container**:

docker run -p 8000:8000 lendable-api

3. Access the API documentation:

open browser and goto:
http://localhost:8000/docs

4. API Endpoint
GET / → Health check
Response: { "message": "API is working!" }

⸻

5. Model Training Notebook

The notebook performs:
Feature preprocessing and encoding
Handling missing data with SimpleImputer
Scaling with StandardScaler
Model training (e.g., XGBoost)
Exporting predictions to CSV

6. Dependencies
pip install -r requiremnts.txt

7.Submission Files
kushal_reddy_malladi_predictions.csv
Dockerfile, API, and requirements
Jupyter notebook with all steps

⸻

8. Author

Kushal Reddy Malladi
Feel free to reach out for any clarifications.

9. Final output sample
loan_id  early_settlement_probability
123456   0.1042
987654   0.2789

---

#### 💾 Step 5: Save the file

Click File > Save or press Ctrl+S.

---

10. save the file
11. Push to GitHub

Now in terminal:

```bash
git add README.md
git commit -m "Add README.md with full assignment documentation"
git push origin main