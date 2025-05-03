🎧 **Grammar Scoring Engine for Audio Samples**
 **Problem Statement**
The goal of this project is to assign a grammar score between 1 and 5 to audio samples. The project involves preprocessing the audio data, extracting features, training multiple machine learning models, and combining their outputs to predict grammar scores.

**🧪 1. Preprocessing & Feature Extraction (Using librosa)**
Why librosa over scipy?
While both librosa and scipy are powerful libraries, librosa is specifically designed for audio processing tasks and provides more high-level functions optimized for extracting relevant audio features. The main reasons for choosing librosa over scipy include:

Audio-specific features: librosa has built-in functions to extract features like MFCC, chroma, and spectral contrast, which are essential for audio classification tasks.

Time-domain and frequency-domain transformations: librosa handles both, making it easier to extract diverse audio features such as zero-crossing rate (ZCR), MFCCs, and spectral contrast.

Efficiency: librosa provides a highly optimized API for feature extraction, which is better suited for real-time applications or large-scale datasets.

Feature Extraction:
MFCCs (Mel-Frequency Cepstral Coefficients):

python
Copy
Edit
mfccs_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=30)
mfccs_scaled = np.mean(mfccs_features.T, axis=0)
Extracts timbral features to capture the unique characteristics of the voice.

Zero Crossing Rate (ZCR):

python
Copy
Edit
zcr = np.mean(librosa.feature.zero_crossing_rate(data))
Measures the rate at which the signal changes sign, which can indicate vocal changes.

Chroma Feature:

python
Copy
Edit
chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sample_rate))
Captures harmonic content and pitch, helping with tonality-related information.

Spectral Contrast:

python
Copy
Edit
contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate))
Highlights spectral peaks and valleys, essential for identifying speech patterns.

Final Feature Vector:
python
Copy
Edit
full_feature = np.hstack([mfccs_scaled, zcr, chroma, contrast])
The final feature vector combines the extracted audio characteristics into a format suitable for machine learning models.

**2. Model Training & Hyperparameter Tuning**
Model Selection:
XGBoost: A powerful gradient boosting model known for its efficiency and high performance in many machine learning tasks.

Random Forest: A robust ensemble method using multiple decision trees to capture non-linear relationships.

Ridge Regression: A regularized linear model that helps in cases where collinearity is present in the features.

LightGBM: A gradient boosting model that is optimized for speed and large datasets.

Hyperparameter Tuning with RandomizedSearchCV:
To find the optimal parameters for each model, we used RandomizedSearchCV to search through a wide range of hyperparameters. This method is more efficient than Grid Search as it samples a fixed number of parameter combinations from a specified range.

Example:

python
Copy
Edit
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1]
}

model = XGBRegressor()
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10)
random_search.fit(X_train, y_train)
**
Model Ensembling (Bagging):**
After training the individual models, we used bagging to combine their predictions and reduce variance, leading to a more robust model. By taking the average prediction (for regression tasks), we aggregated the predictions of the best-performing models.

