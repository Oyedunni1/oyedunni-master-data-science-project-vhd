# Web Application User Flow

## VHD Detection - Streamlit Web Application

This diagram illustrates the complete user experience flow through the Streamlit web application, showing all 5 main tabs and their functionality.

```mermaid
graph TB
    A[User Access<br/>http://localhost:8501] --> B[Streamlit Web Application<br/>VHD Detection System]
    
    B --> C[Main Navigation<br/>5 Core Tabs]
    
    subgraph "Tab 1: Upload & Predict"
        C --> D[File Upload Interface<br/>WAV/MP3/FLAC Support<br/>Max 50MB, 10-60s duration]
        D --> E[Audio Validation<br/>Format & Duration Check]
        E --> F[Feature Extraction<br/>16 Optimal Features]
        F --> G[Model Prediction<br/>Ensemble Learning]
        G --> H[Result Display<br/>Normal/Abnormal + Confidence]
        H --> I[Download Results<br/>CSV Export Available]
    end
    
    subgraph "Tab 2: Analysis Dashboard"
        C --> J[Real-time Analytics<br/>Live Performance Monitoring]
        J --> K[Prediction Statistics<br/>Total Predictions: 23<br/>Success Rate: 100%<br/>Average Confidence: 76.3%]
        K --> L[Performance Trends<br/>Historical Data Visualization]
        L --> M[System Health<br/>Processing Speed & Memory Usage]
    end
    
    subgraph "Tab 3: Model Performance"
        C --> N[Training Metrics<br/>Comprehensive Model Analysis]
        N --> O[Accuracy Metrics<br/>84.91% ± 1.23%<br/>Precision: 80.66%<br/>Recall: 76.42%<br/>F1-Score: 78.48%]
        O --> P[Confusion Matrix<br/>Visual Classification Results]
        P --> Q[Feature Importance<br/>16 Features Ranked]
        Q --> R[Model Comparison<br/>Individual vs Ensemble]
    end
    
    subgraph "Tab 4: Detailed Metrics"
        C --> S[Advanced Analytics<br/>Deep Performance Analysis]
        S --> T[Cross-Validation Results<br/>5-Fold CV Performance]
        T --> U[Train vs Test vs Validation<br/>Overfitting Analysis]
        U --> V[ROC Curve<br/>AUC: 83.21%]
        V --> W[Precision-Recall Curve<br/>Balanced Performance]
        W --> X[Feature Correlation<br/>Feature Relationships]
    end
    
    subgraph "Tab 5: About"
        C --> Y[System Information<br/>Project Overview]
        Y --> Z[Technical Specifications<br/>16 Features, 3 Models<br/>84.91% Accuracy]
        Z --> AA[Research Context<br/>PhysioNet CinC 2016<br/>3,000+ Samples]
        AA --> BB[Contact Information<br/>Support & Documentation]
    end
    
    subgraph "Backend Processing"
        CC[Model Loading<br/>Pre-trained Ensemble] --> DD[Feature Extraction<br/>Parallel Processing]
        DD --> EE[Prediction Engine<br/>Weighted Voting]
        EE --> FF[Performance Tracking<br/>Real-time Metrics]
        FF --> GG[Result Storage<br/>JSON-based Persistence]
    end
    
    subgraph "User Experience Features"
        HH[Responsive Design<br/>Mobile-friendly Interface] --> II[Real-time Updates<br/>Live Performance Monitoring]
        II --> JJ[Error Handling<br/>Graceful Failure Management]
        JJ --> KK[Progress Indicators<br/>Processing Status Display]
        KK --> LL[Export Functionality<br/>Results Download]
    end
    
    F --> CC
    G --> CC
    J --> FF
    N --> CC
    S --> CC
    
    style A fill:#e3f2fd
    style H fill:#c8e6c9
    style K fill:#e8f5e8
    style O fill:#fff3e0
    style V fill:#f3e5f5
    style Z fill:#e1f5fe
```

## Web Application Features:

### **Tab 1: Upload & Predict**
- **File Upload**: Support for WAV, MP3, FLAC formats
- **Validation**: File size (<50MB) and duration (10-60s) checks
- **Processing**: Real-time feature extraction and prediction
- **Results**: Clear Normal/Abnormal classification with confidence score
- **Export**: CSV download of prediction results

### **Tab 2: Analysis Dashboard**
- **Real-time Metrics**: Live performance monitoring
- **Statistics**: Total predictions, success rate, average confidence
- **Trends**: Historical performance visualization
- **System Health**: Processing speed and memory usage monitoring

### **Tab 3: Model Performance**
- **Training Metrics**: Comprehensive model analysis
- **Accuracy Display**: All performance metrics with confidence intervals
- **Confusion Matrix**: Visual classification results
- **Feature Importance**: 16 features ranked by importance
- **Model Comparison**: Individual vs ensemble performance

### **Tab 4: Detailed Metrics**
- **Advanced Analytics**: Deep performance analysis
- **Cross-Validation**: 5-fold CV results visualization
- **Overfitting Analysis**: Train/test/validation comparison
- **ROC Curve**: AUC performance visualization
- **Feature Correlation**: Feature relationship analysis

### **Tab 5: About**
- **System Information**: Project overview and specifications
- **Technical Details**: 16 features, 3 models, 84.91% accuracy
- **Research Context**: PhysioNet CinC 2016 dataset information
- **Documentation**: Support and contact information

## Technical Implementation:

### **Frontend (Streamlit)**
```python
# Main application structure
st.set_page_config(
    page_title="VHD Detection System",
    page_icon="🫀",
    layout="wide"
)

# Tab navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Upload & Predict", 
    "Analysis", 
    "Model Performance", 
    "Detailed Metrics", 
    "About"
])
```

### **Backend Processing**
- **Model Loading**: Pre-trained ensemble model loading
- **Feature Extraction**: Parallel processing of 16 features
- **Prediction Engine**: Weighted voting ensemble
- **Performance Tracking**: Real-time metrics collection
- **Result Storage**: JSON-based persistence

### **User Experience Features**
- **Responsive Design**: Mobile-friendly interface
- **Real-time Updates**: Live performance monitoring
- **Error Handling**: Graceful failure management
- **Progress Indicators**: Processing status display
- **Export Functionality**: Results download capability

## Performance Characteristics:

### **Processing Speed**
- **Real-time Processing**: 10+ files/second
- **Parallel Processing**: Multi-threaded feature extraction
- **Memory Optimization**: Efficient batch processing
- **Platform Optimization**: Automatic platform detection

### **User Interface**
- **5 Main Tabs**: Comprehensive functionality coverage
- **Real-time Analytics**: Live performance monitoring
- **Interactive Visualizations**: Charts and graphs
- **Export Capabilities**: Results download and sharing

### **System Integration**
- **Model Persistence**: Joblib serialization
- **Performance Tracking**: Real-time metrics collection
- **Platform Optimization**: Automatic resource management
- **Error Handling**: Robust failure management
