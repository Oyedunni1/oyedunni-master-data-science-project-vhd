# Machine Learning Pipeline Flowchart

## VHD Detection - Complete ML Pipeline

This diagram illustrates the complete machine learning pipeline from raw audio input to final prediction, showing all processing steps and decision points.

```mermaid
flowchart TD
    A[Raw Heart Sound Audio<br/>WAV/MP3/FLAC] --> B{Audio Validation<br/>Duration: 10-60s<br/>Size: <50MB}
    
    B -->|Valid| C[Audio Preprocessing<br/>Load with librosa<br/>Sample Rate: 2000 Hz]
    B -->|Invalid| Z[Error: Invalid Audio]
    
    C --> D[Signal Filtering<br/>Band-pass: 25-400 Hz<br/>Noise Reduction]
    D --> E[Signal Normalization<br/>Amplitude Scaling<br/>-1 to 1 range]
    E --> F[Signal Segmentation<br/>Heart Cycle Detection<br/>S1-S2 Periods]
    
    F --> G[Feature Extraction<br/>16 Optimal Features]
    
    subgraph "Fractal Features (6)"
        G --> H[Higuchi Fractal Dimension<br/>Ultra-fast complexity]
        G --> I[Sample Entropy<br/>Ultra-fast regularity]
        G --> J[Hurst Exponent<br/>Long-range dependence]
        G --> K[Signal Complexity<br/>Variance of differences]
        G --> L[Spectral Entropy<br/>Frequency complexity]
        G --> M[Signal Standard Deviation<br/>Statistical variation]
    end
    
    subgraph "Audio Features (10)"
        G --> N[Mel-spectrogram<br/>Mean & Standard Deviation]
        G --> O[Spectral Energy<br/>Signal power analysis]
        G --> P[Spectral Centroid<br/>Frequency center of mass]
        G --> Q[Spectral Bandwidth<br/>Frequency spread]
        G --> R[Zero Crossing Rate<br/>Signal crossing frequency]
        G --> S[Spectral Rolloff<br/>Frequency rolloff point]
        G --> T[Spectral Contrast<br/>Timbral analysis]
        G --> U[MFCC Features<br/>2 coefficients]
    end
    
    H --> V[Feature Integration<br/>Concatenate 16 features]
    I --> V
    J --> V
    K --> V
    L --> V
    M --> V
    N --> V
    O --> V
    P --> V
    Q --> V
    R --> V
    S --> V
    T --> V
    U --> V
    
    V --> W[Data Preprocessing<br/>RobustScaler<br/>Feature Selection]
    W --> X[Model Prediction<br/>Ensemble Learning]
    
    subgraph "Ensemble Models"
        X --> Y1[Gradient Boosting<br/>34.3% Weight<br/>AUC: 81.8%]
        X --> Y2[Random Forest<br/>34.3% Weight<br/>AUC: 80.9%]
        X --> Y3[Logistic Regression<br/>31.4% Weight<br/>AUC: 78.5%]
    end
    
    Y1 --> AA[Weighted Voting<br/>AUC-based Weights]
    Y2 --> AA
    Y3 --> AA
    
    AA --> BB[Final Prediction<br/>Normal/Abnormal]
    BB --> CC[Confidence Score<br/>0.0 - 1.0]
    CC --> DD[Result Interpretation<br/>VHD Detected/Not Detected]
    
    DD --> EE[Performance Tracking<br/>Update Metrics]
    EE --> FF[Web Interface Display<br/>User-friendly Results]
    
    style A fill:#e3f2fd
    style G fill:#f3e5f5
    style X fill:#e8f5e8
    style BB fill:#c8e6c9
    style FF fill:#fff3e0
```

## Pipeline Stages:

### 1. **Input Validation**
- Audio format validation (WAV, MP3, FLAC)
- Duration check (10-60 seconds)
- File size validation (<50MB)

### 2. **Signal Preprocessing**
- Audio loading with librosa
- Band-pass filtering (25-400 Hz)
- Noise reduction and normalization
- Heart cycle segmentation

### 3. **Feature Extraction**
- **Fractal Features (6)**: Signal complexity analysis
- **Audio Features (10)**: Spectral and timbral analysis
- **Total**: 16 optimal features

### 4. **Data Preprocessing**
- RobustScaler for outlier resistance
- Feature selection and validation
- Data normalization

### 5. **Ensemble Prediction**
- 3-algorithm weighted voting
- Performance-based weights
- Final prediction with confidence score

### 6. **Result Processing**
- Prediction interpretation
- Performance tracking
- Web interface display

## Performance Metrics:
- **Processing Speed**: 10+ files/second
- **Accuracy**: 84.91% ± 1.23%
- **Features**: 16 optimal features
- **Models**: 3-algorithm ensemble
