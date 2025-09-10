# OpenWildfire

An AI-powered wildfire risk assessment system that analyzes camera feeds, weather data, and geographic information to evaluate fire risk levels and generate emergency recommendations.

## Features

- **Multi-Camera Analysis**: Clusters and analyzes multiple camera feeds for fire and smoke detection  
- **Real-time Weather Integration**: Incorporates current weather conditions including humidity, wind speed, and temperature  
- **Geographic Risk Assessment**: Uses California fire hazard severity zones (FHSZ) and historical fire data  
- **AI-Powered Evaluation**: Leverages Hugging Face's GPT-OSS-20B model for intelligent risk assessment  
- **RAG-Enhanced Recommendations**: Uses Retrieval-Augmented Generation to provide context-aware emergency recommendations  
- **Automated Report Generation**: Creates detailed technical reports in JSON format  

## Project Structure

```
OpenWildfire/
├── data/                           # Data files and emergency documents
├── logs/                           # Application logs
├── reports/                        # Generated assessment reports
│   ├── read_reports/               # HTML report generator
│   └── final_reports/              # Save the final reports
├── src/                            # Source code
│   ├── prompts/                    # Prompts
│   ├── api_utils/                  # API utils
│   │   └── weather_fetcher.py      # Call weather API
│   ├── data_loader.py              # Data processing
│   ├── emergency_rag.py            # RAG system
│   ├── fire_risk_agent.py          # AI assessment agent
│   ├── main_realtime.py            # Main application
│   └── report_generator.py         # Report generation
├── LICENSE                         # MIT License
├── README.md                       # Documentation
└── requirements.txt                # Dependencies
```


## System Architecture

The system consists of several key components:

- **Data Loader**: Processes camera data, geographic information, and weather data  
- **Fire Risk Agent**: AI model that evaluates risk levels and generates recommendations  
- **Emergency RAG**: Enhances recommendations using professional emergency documents  
- **Report Generator**: Creates comprehensive assessment reports  

## Installation

### Prerequisites

- Python 3.8+  
- CUDA-compatible GPU (recommended) or CPU  
- 16GB+ RAM recommended  

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- **torch >= 2.0.0**: For deep learning inference and GPU acceleration  
- **transformers >= 4.35.0**: For loading and running Hugging Face models  
- **langchain**, **langchain-community**: For Retrieval-Augmented Generation (RAG) pipelines  
- **chromadb**: Lightweight vector database for document retrieval  
- **sentence-transformers**: For generating document embeddings  
- **pandas**, **numpy**, **requests**: For data processing and API access  

## Usage

### Basic Usage

#### Step 1: Generate Risk Assessment Reports

Run the main assessment pipeline:

```bash
python src/main_realtime.py
```

This will:

- Load camera monitoring data
- Cluster cameras by geographic proximity
- Analyze each cluster for fire/smoke detection
- Integrate weather and geographic data
- Generate AI risk assessments
- Create detailed JSON reports in the reports/ directory

#### Step 2: Generate HTML Report Visualization

After the assessment is complete, render the reports into HTML format:

```bash
python reports/read_reports/html_report.py
```

This will:

- Process all JSON reports from the `reports/` directory
- Generate interactive HTML visualizations
- Create a comprehensive dashboard view
- Save rendered reports to `reports/read_reports/final_reports/`


### Complete Workflow

```bash
# Run the complete workflow
python src/main_realtime.py
python reports/read_reports/html_report.py
```

## Input Data Format

See `data/README_data.md` for detailed data format specifications.


## Risk Levels

The system evaluates risk on a 5-level scale:

- **Level 1 (Low)**: High humidity, low wind, no detections
- **Level 2 (Low-Moderate)**: Moderate conditions, stable environment
- **Level 3 (Moderate)**: Some risk factors present (smoke detection or concerning weather)
- **Level 4 (Very High)**: Fire detected or multiple adverse conditions
- **Level 5 (Extreme)**: Multiple fires or extreme weather conditions


## Configuration

The system can be configured by modifying parameters in the source files:

- **Model settings**: Edit `fire_risk_agent.py` for AI model parameters
- **RAG settings**: Adjust `data/emergency_docs/config.json` for recommendation enhancement
- **Risk thresholds**: Modify prompt templates in `src/prompts/` for risk evaluation criteria


## Data Requirements

### Required Data Files

Place these files in the `data/` directory:

- `camera_monitoring_dataset.jsonl`: Camera detection data
- `output/nested_california_fire_risk_enhanced_dataset.jsonl`: Geographic and fire risk data
- `emergency_docs/pdfs/`: Emergency procedure documents (PDF format)

### Weather Data

The system automatically fetches current weather data using OpenWeatherMap API. Ensure you have proper API access configured.


## Output

The system generates detailed JSON reports including:

- Risk level and confidence scores
- AI reasoning and analysis
- Emergency recommendations (enhanced by RAG)
- Monitoring requirements
- Fire station resources
- Technical metadata

Reports are saved to the `reports/` directory with timestamps.


## System Requirements

### Minimum Requirements

- 8GB RAM
- Python 3.8+


## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Make your changes  
4. Add tests if applicable  
5. Submit a pull request  


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.


## Disclaimer

This system is designed for informational purposes only. AI-generated risk assessments should be verified by qualified fire safety professionals. In case of immediate fire danger, contact emergency services (911) immediately regardless of this assessment.


## Acknowledgments

- Built using OpenAI's GPT-OSS-20B model
- Weather data provided by OpenWeatherMap
- Geographic data based on California fire risk datasets
- Emergency procedures enhanced through RAG technology
