# System Architecture Diagram

## VHD Detection System - Overall Architecture

This diagram shows the complete system architecture of the VHD Detection System, including all major components and their interactions.

```mermaid
graph TB
    subgraph "Data Layer"
        A[PhysioNet CinC 2016 Dataset<br/>3,000+ PCG Recordings] --> B[Raw Audio Files<br/>WAV Format]
        B --> C[Labels CSV<br/>Normal/Abnormal]
    end
    
    subgraph "Processing Layer"
        D[Data Acquisition<br/>PhysioNetDataManager] --> E[Signal Preprocessing<br/>HeartSoundPreprocessor]
        E --> F[Feature Extraction<br/>16 Optimal Features]
    end
    
    subgraph "Feature Engineering"
        F --> G[Fractal Features<br/>6 Features]
        F --> H[Audio Features<br/>10 Features]
        G --> I[Higuchi FD<br/>Sample Entropy<br/>Hurst Exponent<br/>Signal Complexity<br/>Spectral Entropy<br/>Standard Deviation]
        H --> J[Mel-spectrogram<br/>MFCC<br/>Spectral Energy<br/>Zero Crossing Rate<br/>Spectral Bandwidth<br/>Spectral Contrast]
    end
    
    subgraph "Machine Learning Layer"
        K[Data Preprocessing<br/>RobustScaler + Feature Selection] --> L[Model Training<br/>4 Algorithms]
        L --> M[Gradient Boosting<br/>34.3% Weight]
        L --> N[Random Forest<br/>34.3% Weight]
        L --> O[Logistic Regression<br/>31.4% Weight]
        L --> P[SVM<br/>Excluded - Lower Performance]
    end
    
    subgraph "Ensemble Learning"
        M --> Q[Weighted Voting<br/>AUC-based Weights]
        N --> Q
        O --> Q
        Q --> R[Final Prediction<br/>84.91% Accuracy]
    end
    
    subgraph "Application Layer"
        S[Web Application<br/>Streamlit Interface] --> T[Upload & Predict Tab]
        S --> U[Analysis Dashboard]
        S --> V[Model Performance Tab]
        S --> W[Detailed Metrics Tab]
        S --> X[About Tab]
    end
    
    subgraph "Performance Monitoring"
        Y[Performance Tracker<br/>Real-time Metrics] --> Z[Accuracy: 84.91%<br/>Precision: 80.66%<br/>Recall: 76.42%<br/>F1-Score: 78.48%<br/>AUC: 83.21%]
    end
    
    subgraph "Platform Optimization"
        AA[Platform Detector<br/>Auto-detection] --> BB[Resource Optimizer<br/>CPU/Memory/GPU]
        BB --> CC[Parallel Processing<br/>10+ files/second]
    end
    
    C --> D
    I --> K
    J --> K
    R --> S
    R --> Y
    AA --> D
    AA --> E
    AA --> F
    
    style A fill:#e1f5fe
    style R fill:#c8e6c9
    style S fill:#fff3e0
    style Y fill:#f3e5f5
    style AA fill:#e8f5e8
```

## Key Components:

1. **Data Layer**: PhysioNet CinC 2016 dataset with 3,000+ PCG recordings
2. **Processing Layer**: Signal preprocessing and feature extraction
3. **Feature Engineering**: 16 optimal features (6 fractal + 10 audio)
4. **Machine Learning**: Ensemble of 3 algorithms with weighted voting
5. **Application Layer**: Streamlit web interface with 5 main tabs
6. **Performance Monitoring**: Real-time metrics tracking
7. **Platform Optimization**: Automatic platform detection and optimization

## Performance Metrics:
- **Accuracy**: 84.91% ± 1.23%
- **Processing Speed**: 10+ files/second
- **Features**: 16 optimal features
- **Models**: 3-algorithm ensemble
