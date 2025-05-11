# Image2Pdf_Scanner
A document scanning application with advanced image processing capabilities and a modern dark theme interface.
## Features
- Document Detection : Automatically detects document boundaries in images
- Perspective Correction : Transforms skewed documents to a proper top-down view
- Multiple Enhancement Modes :
  - Normal : Preserves original colors while enhancing clarity
  - Adaptive : High contrast black and white for text documents
  - Otsu : Clear text rendering for printed documents
- Special Processing for Display Boards : Optimized for flight information displays and other LED/LCD screens
- Orientation Correction : Automatically detects and corrects document orientation
- Multiple Image Support : Process multiple images into a single PDF
- Drag & Drop Interface : Easy-to-use modern interface
- Dark Theme : Eye-friendly dark mode design
## Technologies Used
- Python 3.x
- OpenCV for image processing
- Flask for web interface
- PIL/Pillow for image handling
- img2pdf for PDF creation
- scikit-image for advanced image processing
## Installation
1. Clone the repository:
```
git clone https://github.com/VibeCipher/Image2Pdf_Scanner.git
cd document-scanner
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Run the application:
```
python web_app.py
```
4. Open your browser and navigate to:
```
http://localhost:5000
```
## Supported Image Formats
- JPEG/JPG
- PNG
- BMP
- TIFF/TIF
- GIF
- WebP
- HEIC/HEIF
## Requirements
- Python 3.6 or higher
- See requirements.txt for complete list of dependencies

