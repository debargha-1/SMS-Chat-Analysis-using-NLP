# SMS Chat Analysis with NLP

## 📜 Overview
This project explores automated chat analysis using the [UCI SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). By applying natural language processing (NLP) techniques, the system extracts valuable insights such as detecting spam, understanding sentiments, and summarizing key points from SMS data.

---

## ✨ Features
- **Data Preprocessing**: Cleaning and preparing SMS text for analysis.
- **Spam Detection**: Binary classification to identify spam and non-spam messages.
- **Sentiment Analysis**: Determining the tone and sentiment of text messages.
- **Intention Classification**: Categorizing messages based on purpose.
- **Summarization**: Extracting concise insights from conversations.
- **Visualizations**: Charts and graphs for data insights.

---

## 📂 Dataset
The project uses the [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), which contains labeled SMS messages classified as "ham" (not spam) or "spam." This dataset is ideal for text classification and NLP experiments.

---

## 🛠️ Tools and Technologies
- **NLP Libraries**: NLTK
- **Machine Learning**: Scikit-learn, Random Forest, Logistic Regression
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Deployment (Future Scope)**: Flask, FastAPI

## ⚙️ Implementation
### 1. **Preprocessing**
- Tokenization, stop word removal, and stemming/lemmatization.

### 2. **Modeling**
- Experimenting with Logistic Regression, Random Forest, and Naive Bayes for classification.
- Fine-tuning hyperparameters using GridSearchCV.

### 3. **Evaluation**
- Metrics: Accuracy, Precision, Recall, F1-score
- Tools: Confusion Matrix, Classification Reports

---

## 📈 Current Results
**Best Model**: Random Forest  
**Accuracy**: 97.68%  
**Classification Report**:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.97      | 1.00   | 0.99     | 894     |
| 1     | 1.00      | 0.83   | 0.91     | 140     |
| **Accuracy** |       |        | 0.98     | 1034    |

---

## 📝 Future Enhancements that can be made:
- Integrating topic modeling with Gensim.
- Deploying the system via Flask or FastAPI

---

