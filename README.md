# ExamPredictor - Competitive Exam Preparation Tool

ExamPredictor is a comprehensive web application that uses data analysis to predict the most probable questions for competitive exams in India, including technical examinations like GATE, IES/ESE, and PSU exams.

## Features

- **File Upload**: Upload syllabus and previous question papers (PDF, DOCX, TXT formats)
- **Text Extraction**: Automatically extracts text from uploaded documents
- **Intelligent Analysis**: Analyzes syllabus coverage, topic frequency, and difficulty distribution
- **Prediction Engine**: Generates predicted question papers based on historical data
- **Support for Technical Exams**: Specialized analysis for engineering subjects (Electrical, Civil, Mechanical, Computer Science, Electronics)
- **Interactive UI**: Modern, responsive design with real-time updates
- **Data Visualization**: Charts and graphs for analysis results
- **Export Functionality**: Download predictions as CSV files

## Tech Stack

### Frontend
- HTML5, CSS3, JavaScript
- Tailwind CSS for styling
- Chart.js for data visualization
- Font Awesome for icons

### Backend
- Python 3.8+
- Flask web framework
- PyPDF2 for PDF processing
- python-docx for DOCX processing
- scikit-learn for text analysis
- NumPy for numerical operations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- A modern web browser

### Step 1: Clone or Download the Repository

```bash
# If using git
git clone <repository-url>
cd exampredictor

# Or download and extract the ZIP file
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Backend Server

```bash
python app.py
```

The backend server will start on `http://localhost:5000`

### Step 4: Open the Frontend

1. Open `index.html` in your web browser
2. Or use a local server:

```bash
# Using Python's built-in HTTP server
python -m http.server 8000
```

Then navigate to `http://localhost:8000` in your browser.

## Usage

### 1. Upload Syllabus
- Click on the "Browse Files" button in the Syllabus section
- Select your syllabus file (PDF, DOCX, or TXT format)
- Wait for the upload to complete

### 2. Upload Question Papers
- Click on the "Browse Files" button in the Previous Question Papers section
- Select one or more previous question paper files
- Wait for all files to upload

### 3. Analyze Data
- Click the "Analyze Data" button
- The system will process the uploaded files and extract topics
- Analysis results will be displayed in the Analysis section

### 4. View Predictions
- After analysis, predictions are automatically generated
- View predicted questions in the Prediction section
- Each prediction shows the subject, topic, question type, and probability

### 5. Download Results
- Click "Download Full Predicted Paper (PDF)" to export predictions as CSV
- Use the data for your exam preparation

## API Endpoints

### POST /api/upload/syllabus
Upload syllabus file for analysis

**Request**: multipart/form-data with 'file' field
**Response**: JSON with success status and extracted subjects

### POST /api/upload/question-papers
Upload question paper files for analysis

**Request**: multipart/form-data with 'files' field (multiple files)
**Response**: JSON with success status and uploaded file count

### POST /api/analyze
Perform comprehensive analysis of uploaded data

**Response**: JSON with analysis results including:
- Syllabus coverage percentages
- Topic frequency
- Difficulty distribution

### POST /api/predict
Generate predicted question paper

**Response**: JSON with predictions including:
- Subject
- Topic
- Probability
- Question type

### GET /api/status
Get current status of uploaded data

**Response**: JSON with upload status

### POST /api/reset
Reset all uploaded data

**Response**: JSON with success status

## Supported Exams

### General Competitive Exams
- UPSC Civil Services
- SSC CGL
- IBPS PO/Clerk
- Railway Exams
- State PSC Exams

### Technical Exams
- GATE (All branches)
- IES/ESE (Engineering Services)
- PSU Exams (ONGC, BHEL, NTPC, etc.)
- DRDO, BARC Entrance Exams
- State AE & JE Exams

### Supported Engineering Subjects
- Electrical Engineering
- Civil Engineering
- Mechanical Engineering
- Computer Science & Engineering
- Electronics & Communication

## Project Structure

```
exampredictor/
├── index.html          # Frontend HTML file
├── app.py              # Backend Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── uploads/           # Uploaded files (created automatically)
├── analysis/          # Analysis results (created automatically)
└── predictions/       # Generated predictions (created automatically)
```

## How It Works

1. **Text Extraction**: The system extracts text from uploaded PDF, DOCX, or TXT files
2. **Topic Identification**: Uses pattern matching to identify subjects and topics
3. **Frequency Analysis**: Counts topic occurrences across question papers
4. **Syllabus Matching**: Compares topics with the uploaded syllabus
5. **Probability Calculation**: Calculates prediction probability based on:
   - Topic frequency in previous papers (60% weight)
   - Syllabus coverage (40% weight)
6. **Ranking**: Sorts predictions by probability and returns top results

## Limitations & Future Improvements

### Current Limitations
- Text-based analysis (doesn't analyze images or diagrams)
- Pattern matching approach (could be improved with ML)
- Simulated difficulty distribution (needs actual ML model)

### Planned Improvements
- Machine Learning model for better predictions
- OCR support for scanned documents
- Database integration for data persistence
- User authentication and personalization
- Advanced analytics and study recommendations
- Mobile application

## Troubleshooting

### Backend Not Starting
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 5000 is available
- Verify Python version (3.8+)

### File Upload Failing
- Check file format (PDF, DOCX, or TXT only)
- Ensure file size is under 16MB
- Verify backend server is running

### CORS Errors
- Make sure backend server is running on localhost:5000
- Check browser console for specific error messages

### Analysis Not Working
- Upload both syllabus and question papers
- Ensure files contain readable text (not just images)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Support

For support, email support@exampredictor.in or raise an issue in the repository.

## Disclaimer

This tool provides predictions based on historical data analysis. It should be used as a supplementary study aid and not as a replacement for comprehensive exam preparation.
