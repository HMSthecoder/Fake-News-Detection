# 📰 Fake News Detection Project

A machine learning-powered web application that classifies news articles as either **Real** or **Fake** using advanced NLP techniques and multiple ML models.

## 🌟 Features

- **Dual Model Support**: Choose between Logistic Regression and Naive Bayes models
- **Interactive Web Interface**: Beautiful Streamlit-based UI with animated background
- **Real-time Analysis**: Instant classification with confidence scores
- **Text Preprocessing**: Advanced NLP preprocessing including tokenization, stemming, and TF-IDF vectorization
- **Visual Analytics**: Comprehensive EDA with confusion matrices and performance metrics

## 🏗️ Project Structure

```
Fake News Detection/
├── data/
│   ├── Fake.csv           # Fake news dataset
│   └── True.csv           # Real news dataset
├── model/
│   ├── lr_model.pkl       # Trained Logistic Regression model
│   ├── naive_bayes_model.pkl  # Trained Naive Bayes model
│   └── tfidf_vectorizer.pkl   # TF-IDF vectorizer
├── notebooks/
│   └── 01_EDA.ipynb       # Exploratory Data Analysis
├── src/
│   ├── app.py             # Main Streamlit application
│   └── animation.html     # Background animation for UI
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn nltk
```

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd "Fake News Detection"
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if running notebooks)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

### Running the Application

```bash
streamlit run src/app.py
```

## 📊 Dataset Information

- **Total Articles**: 44,898 news articles
- **Real News**: 21,417 articles
- **Fake News**: 23,481 articles
- **Features**: Title, Text, Subject, Date

## 🔬 Model Performance

### Logistic Regression

- **Accuracy**: ~99%
- **Precision**: High for both classes
- **Recall**: Excellent fake news detection

### Naive Bayes

- **Accuracy**: ~94%
- **Precision**: Good overall performance
- **Recall**: Strong classification capability

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline

1. **Text Cleaning**: Lowercasing, punctuation removal
2. **Stop Word Removal**: English stopwords filtering
3. **Tokenization**: Word-level tokenization using NLTK
4. **Stemming**: Porter Stemmer for word normalization
5. **Vectorization**: TF-IDF with 5000 max features

### Machine Learning Models

- **Logistic Regression**: Linear classification with regularization
- **Naive Bayes**: Multinomial Naive Bayes for text classification

### Web Application Features

- **Dynamic Background**: Animated news-related words
- **Model Selection**: Toggle between ML models
- **Confidence Scoring**: Probability-based confidence display
- **Responsive Design**: Clean, professional UI

## 📈 Usage Examples

### Analyzing News Articles

1. **Select Model**: Choose between Logistic Regression or Naive Bayes
2. **Input Text**: Paste news article or headline
3. **Get Results**: View classification with confidence score

### Example Results

- ✅ **Real News**: "Scientists discover new treatment..." (Confidence: 94.2%)
- 🚨 **Fake News**: "Breaking: Celebrity scandal revealed..." (Confidence: 87.6%)

## 🔍 Exploratory Data Analysis

The project includes comprehensive EDA covering:

- **Data Distribution**: Real vs Fake news counts
- **Text Length Analysis**: Article length distribution
- **Model Comparison**: Confusion matrices and ROC curves
- **Performance Metrics**: Precision, Recall, F1-score

## 🎨 UI Features

- **Animated Background**: Dynamic word cloud with news-related terms
- **Color-coded Results**: Green for real, red for fake news
- **Interactive Sidebar**: Model information and usage instructions
- **Responsive Layout**: Optimized for different screen sizes

## 🔧 Configuration

### Model Paths

Update file paths in `src/app.py` if needed:

### Animation Customization

Modify `src/animation.html` to customize:

- Word list for background animation
- Colors and timing
- Animation frequency

## 📝 Future Enhancements

- [ ] Deep learning models (LSTM, BERT)
- [ ] Real-time news feed integration
- [ ] Multilingual support
- [ ] API endpoint development
- [ ] Docker containerization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request


## 👨‍💻 Author

**Himanshu** - [GitHub Profile](https://github.com/HMSthecoder)

## 🙏 Acknowledgments

- Dataset providers for real and fake news articles
- Streamlit team for the amazing framework
- scikit-learn community for ML tools
- NLTK developers for NLP capabilities
- My ma'am for encouraging me to do this work

---

_Built with ❤️ using Python, Streamlit, and Machine Learning_
