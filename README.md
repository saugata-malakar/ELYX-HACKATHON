🌟 Sentiment Analysis using Machine Learning and Ensemble Methods
<p align="center"> <img src="https://img.shields.io/badge/Machine%20Learning-Sentiment%20Analysis-blue?style=for-the-badge&logo=python" /> <img src="https://img.shields.io/badge/Framework-Scikit--Learn-green?style=for-the-badge&logo=scikit-learn" /> <img src="https://img.shields.io/badge/Boosting-XGBoost%20|%20LightGBM%20|%20CatBoost-orange?style=for-the-badge" /> <img src="https://img.shields.io/badge/Deployment-ONNX%20|%20Pickle-red?style=for-the-badge" /> </p>
📖 Overview

This repository presents an end-to-end sentiment analysis pipeline that classifies text into positive, negative, or neutral categories.
We implemented multiple models ranging from Logistic Regression & SVM to XGBoost, LightGBM, and CatBoost, and finally built a Stacking Ensemble that achieved the best overall performance (~90% accuracy, 0.89 Macro-F1).

The project demonstrates:
✔️ Preprocessing with tokenization, stop-word removal, and TF-IDF vectorization
✔️ Model comparison across linear, tree-based, and boosting algorithms
✔️ Stacking Ensemble for maximum predictive power
✔️ Cross-validation leaderboard for robust evaluation
✔️ Deployment-ready models in Pickle (.pkl) and ONNX (.onnx) formats

📂 Project Structure
sentiment-analysis-ensemble/
│
├── data/                # Dataset or data fetching instructions
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Source code (preprocessing, models, utils)
│   ├── preprocessing.py
│   ├── models.py
│   ├── ensemble.py
│   └── utils.py
├── results/             # Saved models, plots, and leaderboard
│   ├── leaderboard.csv
│   ├── confusion_matrix.png
│   └── best_model.onnx
├── requirements.txt     # Dependencies
├── README.md            # Project description
└── main.py              # Entry point for training and evaluation

⚙️ Installation

Clone the repository and install dependencies:

git clone https://github.com/your-username/sentiment-analysis-ensemble.git
cd sentiment-analysis-ensemble
pip install -r requirements.txt

▶️ Usage
🔹 Train Models
python main.py --mode train --data ./data/dataset.csv

🔹 Evaluate Models
python main.py --mode evaluate --data ./data/test.csv

🔹 Predict Sentiment
python main.py --mode predict --text "This product is fantastic!"


Output:

Positive (confidence: 0.91)

🔹 Export Models
python main.py --mode export


Models are saved in both .pkl and .onnx formats under results/.

📊 Results
🏆 Leaderboard (Example)
Rank	Model	CV Macro-F1	Holdout Macro-F1
1	Stacking Ensemble	0.89	0.89
2	CatBoost	0.86	0.86
3	LightGBM	0.85	0.85
4	XGBoost	0.84	0.84
5	Linear SVM	0.81	0.81

📌 The Stacking Ensemble consistently outperformed individual models.

🔮 Future Directions

🌐 Integrate Neural Networks (LSTM, GRU, CNN)

🤖 Fine-tune BERT/DistilBERT for SOTA performance

🧾 Add explainability with SHAP/LIME

⚡ Deploy real-time API with FastAPI or Flask

🤝 Contributing

Contributions are welcome!

Fork the repo 🍴

Create a new branch 🌱 (feature-branch)

Commit your changes ✅

Open a Pull Request 🔥

📜 License

This project is licensed under the MIT License.
