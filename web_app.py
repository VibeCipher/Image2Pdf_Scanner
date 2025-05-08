from flask import Flask, request, render_template, send_file, redirect, url_for
import os
from pathlib import Path
import uuid
from pdf_scanner import create_pdf
import shutil
from werkzeug.utils import secure_filename  # Add this import

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Configure upload folder
UPLOAD_FOLDER = Path('uploads')
RESULT_FOLDER = Path('results')
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)

# Create debug directory
Path("debug_output").mkdir(exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        return redirect(request.url)
    
    # Create a unique session ID
    session_id = str(uuid.uuid4())
    session_folder = UPLOAD_FOLDER / session_id
    session_folder.mkdir(exist_ok=True)
    
    # Save uploaded files
    image_paths = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.heic', '.heif')
    
    for file in files:
        if file and (file.filename.lower().endswith(valid_extensions) or file.mimetype.startswith('image/')):
            # Ensure filename is safe
            filename = secure_filename(file.filename)
            filepath = session_folder / filename
            file.save(filepath)
            image_paths.append(filepath)
    
    if not image_paths:
        return "No valid image files uploaded", 400
    
    # Process images
    output_path = RESULT_FOLDER.absolute() / f"{session_id}.pdf"
    enhance_mode = request.form.get('enhance_mode', 'adaptive')
    
    try:
        success = create_pdf(image_paths, output_path, enhance_mode=enhance_mode, debug=True)
        
        if success:
            return redirect(url_for('download_file', filename=f"{session_id}.pdf"))
        else:
            return "Failed to create PDF. Please check the server logs for details.", 500
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error creating PDF: {str(e)}\n{error_details}")
        return f"Error creating PDF: {str(e)}", 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(RESULT_FOLDER / filename, as_attachment=True)

@app.route('/cleanup/<session_id>')
def cleanup(session_id):
    # Clean up uploaded files after processing
    session_folder = UPLOAD_FOLDER / session_id
    if session_folder.exists():
        shutil.rmtree(session_folder)
    return "Cleanup complete"

@app.route('/test_pdf')
def test_pdf():
    try:
        from PIL import Image
        import img2pdf
        import io
        import tempfile
        
        # Print temp directory for debugging
        print(f"Temporary directory: {tempfile.gettempdir()}")
        
        # Create a simple image
        img = Image.new('RGB', (100, 100), color = 'red')
        
        # Save to BytesIO
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Convert to PDF
        test_pdf_path = RESULT_FOLDER.absolute() / "test.pdf"
        print(f"Attempting to create PDF at: {test_pdf_path}")
        
        with open(test_pdf_path, "wb") as f:
            f.write(img2pdf.convert(img_byte_arr.getvalue()))
        
        return f"Test PDF created successfully at {test_pdf_path}"
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in test_pdf: {str(e)}\n{error_details}")
        return f"Error creating test PDF: {str(e)}<br><pre>{error_details}</pre>"

if __name__ == '__main__':
    app.run(debug=True)