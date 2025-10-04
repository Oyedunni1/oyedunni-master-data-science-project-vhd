# Data Flow Diagram

## VHD Detection - Complete Data Flow from Input to Prediction

This diagram illustrates the complete data flow through the VHD detection system, from raw audio input to final prediction output, showing all processing stages and data transformations.

```mermaid
flowchart TD
    A[Raw Audio Input<br/>WAV/MP3/FLAC File] --> B[Input Validation<br/>File Format Check<br/>Size: <50MB<br/>Duration: 10-60s]
    
    B -->|Valid| C[Audio Loading<br/>librosa.load<br/>Sample Rate: 2000 Hz]
    B -->|Invalid| Z[Error Response<br/>Invalid File Format]
    
    C --> D[Signal Preprocessing<br/>Band-pass Filter: 25-400 Hz<br/>Noise Reduction<br/>Normalization: -1 to 1]
    
    D --> E[Signal Segmentation<br/>Heart Cycle Detection<br/>S1-S2 Period Identification]
    
    E --> F[Feature Extraction Pipeline<br/>16 Optimal Features]
    
    subgraph "Fractal Feature Extraction (6)"
        F --> G[Higuchi Fractal Dimension<br/>Signal Complexity Analysis]
        F --> H[Sample Entropy<br/>Regularity Measurement]
        F --> I[Hurst Exponent<br/>Long-range Dependence]
        F --> J[Signal Complexity<br/>Variance of Differences]
        F --> K[Spectral Entropy<br/>Frequency Complexity]
        F --> L[Signal Standard Deviation<br/>Statistical Variation]
    end
    
    subgraph "Audio Feature Extraction (10)"
        F --> M[Mel-spectrogram Analysis<br/>Mean & Standard Deviation]
        F --> N[Spectral Energy<br/>Signal Power Analysis]
        F --> O[Spectral Centroid<br/>Frequency Center of Mass]
        F --> P[Spectral Bandwidth<br/>Frequency Spread]
        F --> Q[Zero Crossing Rate<br/>Signal Crossing Frequency]
        F --> R[Spectral Rolloff<br/>Frequency Rolloff Point]
        F --> S[Spectral Contrast<br/>Timbral Analysis]
        F --> T[MFCC Features<br/>2 Coefficients]
    end
    
    G --> U[Feature Integration<br/>Concatenate 16 Features<br/>Fractal + Audio Features]
    H --> U
    I --> U
    J --> U
    K --> U
    L --> U
    M --> U
    N --> U
    O --> U
    P --> U
    Q --> U
    R --> U
    S --> U
    T --> U
    
    U --> V[Data Preprocessing<br/>RobustScaler Normalization<br/>Feature Selection Validation]
    
    V --> W[Model Prediction<br/>Ensemble Learning System]
    
    subgraph "Ensemble Prediction"
        W --> X[Gradient Boosting<br/>34.3% Weight<br/>AUC: 81.8%]
        W --> Y[Random Forest<br/>34.3% Weight<br/>AUC: 80.9%]
        W --> AA[Logistic Regression<br/>31.4% Weight<br/>AUC: 78.5%]
    end
    
    X --> BB[Weighted Voting<br/>AUC-based Combination<br/>Performance-weighted Average]
    Y --> BB
    AA --> BB
    
    BB --> CC[Final Prediction<br/>Binary Classification<br/>Normal (0) / Abnormal (1)]
    
    CC --> DD[Confidence Calculation<br/>Maximum Probability<br/>Range: 0.0 - 1.0]
    
    DD --> EE[Result Interpretation<br/>VHD Detected/Not Detected<br/>Clinical Significance]
    
    EE --> FF[Performance Tracking<br/>Update Metrics<br/>Record Prediction]
    
    FF --> GG[Web Interface Display<br/>User-friendly Results<br/>Confidence Score Display]
    
    subgraph "Data Storage & Persistence"
        HH[Model Storage<br/>joblib.pkl files] --> II[Performance Metrics<br/>JSON-based storage]
        II --> JJ[Prediction History<br/>Real-time tracking]
        JJ --> KK[System Logs<br/>Error tracking & debugging]
    end
    
    subgraph "Performance Optimization"
        LL[Parallel Processing<br/>ThreadPoolExecutor] --> MM[Vectorized Operations<br/>NumPy optimizations]
        MM --> NN[Memory Management<br/>Batch processing]
        NN --> OO[Platform Optimization<br/>Auto-detection & tuning]
    end
    
    W --> HH
    FF --> HH
    LL --> F
    MM --> F
    NN --> F
    OO --> F
    
    style A fill:#e3f2fd
    style U fill:#f3e5f5
    style CC fill:#c8e6c9
    style GG fill:#fff3e0
    style Z fill:#ffebee
```

## Data Flow Stages:

### **1. Input Stage**
- **Raw Audio**: WAV/MP3/FLAC file upload
- **Validation**: Format, size, and duration checks
- **Error Handling**: Invalid file rejection with user feedback

### **2. Preprocessing Stage**
- **Audio Loading**: librosa library with 2000 Hz sample rate
- **Signal Filtering**: Band-pass filter (25-400 Hz)
- **Noise Reduction**: Signal cleaning and normalization
- **Segmentation**: Heart cycle detection (S1-S2 periods)

### **3. Feature Extraction Stage**
- **Fractal Features (6)**: Signal complexity analysis
- **Audio Features (10)**: Spectral and timbral analysis
- **Integration**: 16-dimensional feature vector
- **Validation**: Feature quality and completeness checks

### **4. Data Preprocessing Stage**
- **Normalization**: RobustScaler for outlier resistance
- **Feature Selection**: 16 optimal features validation
- **Data Validation**: NaN/Inf value checks

### **5. Model Prediction Stage**
- **Ensemble Learning**: 3-algorithm weighted voting
- **Weight Calculation**: AUC-based performance weights
- **Prediction**: Binary classification (Normal/Abnormal)
- **Confidence**: Probability score calculation

### **6. Output Stage**
- **Result Interpretation**: Clinical significance
- **Performance Tracking**: Metrics update
- **Web Display**: User-friendly interface
- **Data Persistence**: Results storage

## Technical Implementation:

### **Feature Extraction Pipeline**
```python
def extract_features_from_file(self, file_path: str):
    # Load and preprocess signal
    signal, sr = librosa.load(file_path, sr=2000)
    processed_signal = self.preprocessor.fast_preprocess(signal)
    
    # Extract fractal features (6)
    fractal_features = self.fractal_extractor.extract_all_fractal_features(processed_signal)
    
    # Extract audio features (10)
    deep_features = self.deep_extractor.extract_ensemble_features(processed_signal, sr)
    
    return fractal_features, deep_features
```

### **Ensemble Prediction**
```python
def predict_proba(self, X):
    """Ensemble prediction with weighted voting"""
    predictions = []
    for name, model in self.models.items():
        pred_proba = model.predict_proba(X)[:, 1]
        predictions.append(pred_proba)
    
    # Weighted average of predictions
    ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
    return ensemble_pred
```

### **Performance Optimization**
- **Parallel Processing**: ThreadPoolExecutor for concurrent operations
- **Vectorized Operations**: NumPy optimizations for mathematical computations
- **Memory Management**: Efficient batch processing and cleanup
- **Platform Optimization**: Automatic detection and resource tuning

## Data Transformations:

### **Input → Preprocessed Signal**
- Raw audio → Filtered, normalized signal
- Variable length → Standardized processing
- Multiple formats → Unified representation

### **Signal → Features**
- Time-domain signal → 16-dimensional feature vector
- Raw audio → Mathematical representations
- Continuous signal → Discrete feature values

### **Features → Prediction**
- 16 features → Binary classification
- Continuous values → Discrete decision
- Mathematical model → Clinical interpretation

## Performance Characteristics:

### **Processing Speed**
- **Real-time**: 10+ files/second processing rate
- **Parallel**: Multi-threaded feature extraction
- **Optimized**: Vectorized NumPy operations
- **Efficient**: Memory-optimized batch processing

### **Data Quality**
- **Validation**: Input format and quality checks
- **Error Handling**: Graceful failure management
- **Robustness**: Outlier-resistant preprocessing
- **Consistency**: Standardized feature extraction

### **System Integration**
- **Model Persistence**: Joblib serialization
- **Performance Tracking**: Real-time metrics
- **Web Interface**: User-friendly display
- **Data Storage**: JSON-based persistence
