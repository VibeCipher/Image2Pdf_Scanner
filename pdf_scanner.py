import os
import io
import sys
import argparse
import cv2
import numpy as np
import img2pdf
from PIL import Image
from skimage.filters import threshold_local
from pathlib import Path
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentScanner:
    """
    A class that handles document scanning operations including
    boundary detection, perspective correction, and image enhancement.
    """
    
    def __init__(self, debug=False):
        """
        Initialize the document scanner.
        
        Args:
            debug (bool): If True, intermediate processing steps will be saved as images.
        """
        self.debug = debug
        self.debug_dir = None
        if debug:
            self.debug_dir = Path("debug_output")
            self.debug_dir.mkdir(exist_ok=True)
    
    def _save_debug_image(self, image, name):
        """Save intermediate images when in debug mode."""
        if self.debug:
            path = self.debug_dir / f"{name}.jpg"
            cv2.imwrite(str(path), image)
            logger.debug(f"Saved debug image: {path}")
    
    def preprocess_image(self, image):
        """
        Preprocess the image for document detection.
        
        Args:
            image: The input image.
            
        Returns:
            Preprocessed image ready for contour detection.
        """
        # Create a copy of the image
        orig = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._save_debug_image(gray, "1_grayscale")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self._save_debug_image(blurred, "2_blurred")
        
        # Apply edge detection
        edged = cv2.Canny(blurred, 75, 200)
        self._save_debug_image(edged, "3_edged")
        
        # Apply dilation and erosion to strengthen the edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edged, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        self._save_debug_image(eroded, "4_morphology")
        
        return eroded, orig
    
    def find_document_contour(self, processed_image, original_image):
        """
        Find the contour of the document in the image.
        
        Args:
            processed_image: The preprocessed image.
            original_image: The original image.
            
        Returns:
            The four corners of the document or None if not found.
        """
        # Find contours in the processed image
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Initialize document contour
        document_contour = None
        
        # Loop through contours to find the document
        for contour in contours:
            # Approximate the contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # If the contour has 4 points, we can assume it's the document
            if len(approx) == 4:
                document_contour = approx
                break
        
        # If no suitable contour is found, try with the largest contour
        if document_contour is None and contours:
            largest_contour = contours[0]
            perimeter = cv2.arcLength(largest_contour, True)
            document_contour = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)
            
            # If the approximation has too many points, simplify to 4 corners
            if len(document_contour) > 4:
                # Get the bounding rectangle
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                document_contour = np.int0(box)
        
        # Draw the contour on a copy of the original image for debugging
        if document_contour is not None and self.debug:
            contour_image = original_image.copy()
            cv2.drawContours(contour_image, [document_contour], -1, (0, 255, 0), 3)
            self._save_debug_image(contour_image, "5_document_contour")
        
        return document_contour
    
    def order_points(self, pts):
        """
        Order points in the order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: The four points representing the document corners.
            
        Returns:
            Ordered points.
        """
        # Initialize a list of coordinates
        rect = np.zeros((4, 2), dtype="float32")
        
        # The top-left point will have the smallest sum
        # The bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # The top-right point will have the smallest difference
        # The bottom-left point will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def four_point_transform(self, image, pts):
        """
        Apply a perspective transform to obtain a top-down view of the document.
        
        Args:
            image: The input image.
            pts: The four corners of the document.
            
        Returns:
            Warped image with corrected perspective.
        """
        # Obtain a consistent order of the points
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # Compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Construct the set of destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        self._save_debug_image(warped, "6_warped")
        
        return warped
    
    def enhance_image(self, image, enhance_mode="adaptive"):
        """
        Enhance the scanned document image for better readability.
        
        Args:
            image: The warped document image.
            enhance_mode: The enhancement mode to use ('adaptive', 'otsu', or 'normal').
            
        Returns:
            Enhanced document image.
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply different enhancement techniques based on mode
        if enhance_mode == "adaptive":
            # Apply adaptive thresholding with more conservative parameters
            block_size = 25  # Increased block size
            C = 15  # Increased constant subtracted from mean
            binary = threshold_local(gray, block_size, offset=C)
            enhanced = (gray > binary).astype("uint8") * 255
        elif enhance_mode == "otsu":
            # Apply Otsu's thresholding with Gaussian blur pre-processing
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, enhanced = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Apply basic enhancement - use CLAHE instead of simple histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
        # Save debug image
        self._save_debug_image(enhanced, "7_enhanced")
        
        # Return the original color image for "normal" mode to preserve color information
        if enhance_mode == "normal":
            return image
        
        # For binary modes, convert back to BGR for consistency
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def detect_text_orientation(self, image):
        """
        Detect and correct the orientation of text in the document.
        
        Args:
            image: The document image.
            
        Returns:
            Image with corrected orientation.
        """
        # This is a simplified approach. For more accurate results,
        # OCR or more advanced techniques would be needed.
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use Hough Line Transform to detect lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            logger.warning("No lines detected for orientation correction")
            return image
        
        # Calculate the angles of the lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Avoid division by zero
                continue
            angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
            angles.append(angle)
        
        if not angles:
            logger.warning("No valid angles detected for orientation correction")
            return image
        
        # Find the median angle
        median_angle = np.median(angles)
        
        # If the median angle is close to horizontal or vertical, no rotation needed
        if abs(median_angle) < 1 or abs(median_angle - 90) < 1 or abs(median_angle + 90) < 1:
            return image
        
        # Determine the rotation angle
        if abs(median_angle) < 45:
            rotation_angle = -median_angle
        elif median_angle > 45:
            rotation_angle = 90 - median_angle
        else:  # median_angle < -45
            rotation_angle = -90 - median_angle
        
        # Rotate the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        self._save_debug_image(rotated, "8_orientation_corrected")
        
        return rotated
    
    def scan_document(self, image_path, enhance_mode="adaptive"):
        """
        Process an image to scan a document.
        
        Args:
            image_path: Path to the input image.
            enhance_mode: The enhancement mode to use.
            
        Returns:
            Processed document image or None if processing fails.
        """
        try:
            # Read the image with OpenCV
            image = cv2.imread(str(image_path))
            
            # If OpenCV fails, try with PIL as fallback
            if image is None:
                logger.warning(f"OpenCV failed to read image: {image_path}, trying with PIL")
                try:
                    from PIL import Image
                    import numpy as np
                    pil_image = Image.open(str(image_path))
                    # Convert to RGB if it's not already
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    # Convert PIL image to numpy array for OpenCV
                    image = np.array(pil_image)
                    # Convert RGB to BGR (OpenCV format)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                except Exception as img_error:
                    logger.error(f"Both OpenCV and PIL failed to read image: {image_path}. Error: {str(img_error)}")
                    return None
            
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            
            # Check if image is a display board (bright text on dark background)
            is_display_board = self.is_display_board(image)
            
            # For display boards, use special processing
            if is_display_board and enhance_mode != "normal":
                logger.info(f"Detected display board image, using special processing")
                return self.process_display_board(image)
            
            # Resize large images to improve processing speed
            max_dimension = 1500
            h, w = image.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                logger.info(f"Resized image from {w}x{h} to {int(w*scale)}x{int(h*scale)}")
            
            # Save a copy of the original image
            original = image.copy()
            
            # If enhance_mode is "normal", skip document detection and just enhance
            if enhance_mode == "normal":
                enhanced = self.enhance_image(original, enhance_mode)
                return enhanced
            
            # Otherwise, try to detect and process the document
            try:
                # Preprocess the image
                processed = self.preprocess_image(image)
                
                # Find document contour
                document_contour = self.find_document_contour(processed, original)
                
                if document_contour is None:
                    logger.warning(f"No document contour found in {image_path}. Using the entire image.")
                    # If no contour is found, use the entire image
                    enhanced = self.enhance_image(original, enhance_mode)
                    return enhanced
                
                # Reshape contour to required format
                document_contour = document_contour.reshape(4, 2)
                
                # Apply perspective transform
                warped = self.four_point_transform(original, document_contour)
                
                # Enhance the image
                enhanced = self.enhance_image(warped, enhance_mode)
                
                return enhanced
                
            except Exception as inner_e:
                logger.warning(f"Error in document processing: {str(inner_e)}, falling back to original image")
                # Fallback to original image with basic enhancement
                enhanced = self.enhance_image(original, enhance_mode)
                return enhanced
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def is_display_board(self, image):
        """
        Detect if an image is likely a display board (bright text on dark background)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate average brightness
            avg_brightness = np.mean(gray)
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Normalize histogram
            hist = hist.flatten() / hist.sum()
            
            # Check if image has mostly dark pixels with some bright spots (typical for display boards)
            dark_ratio = np.sum(hist[:50]) / np.sum(hist)
            bright_spots = np.sum(hist[200:])
            
            # Display board typically has low average brightness but some bright spots
            return avg_brightness < 100 and dark_ratio > 0.6 and bright_spots > 0.01
        except Exception as e:
            logger.error(f"Error in is_display_board: {str(e)}")
            return False
            
    def process_display_board(self, image):
        """
        Special processing for display board images
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold to highlight text
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 21, 5
            )
            
            # Invert the image to get white text on black background
            thresh = cv2.bitwise_not(thresh)
            
            # Apply morphological operations to clean up the text
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to color for consistency with other processing paths
            result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            return result
        except Exception as e:
            logger.error(f"Error in process_display_board: {str(e)}")
            # Fallback to normal processing
            return self.enhance_image(image, "adaptive")

def create_pdf(image_paths, output_path, enhance_mode="adaptive", debug=False):
    """
    Create a PDF from one or more images.
    
    Args:
        image_paths: List of paths to input images.
        output_path: Path to save the output PDF.
        enhance_mode: The enhancement mode to use.
        debug: Whether to save debug images.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        start_time = time.time()
        logger.info(f"Processing {len(image_paths)} image(s)...")
        
        # Initialize the document scanner
        scanner = DocumentScanner(debug=debug)
        
        # Process each image
        processed_images = []
        for i, img_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
            
            # Scan the document
            processed = scanner.scan_document(img_path, enhance_mode)
            
            if processed is not None:
                # Save a debug copy of the processed image
                if debug:
                    debug_path = Path("debug_output") / f"processed_{i}.jpg"
                    cv2.imwrite(str(debug_path), processed)
                
                # Convert OpenCV image to PIL Image
                if len(processed.shape) == 2:  # If grayscale
                    # For grayscale, convert to RGB to ensure compatibility with img2pdf
                    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                    pil_image = Image.fromarray(processed_rgb)
                else:  # If color
                    # OpenCV uses BGR, PIL uses RGB
                    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(processed_rgb)
                
                # Save a debug copy of the PIL image
                if debug:
                    debug_pil_path = Path("debug_output") / f"pil_processed_{i}.jpg"
                    pil_image.save(str(debug_pil_path))
                
                processed_images.append(pil_image)
            else:
                # If processing fails, try to use the original image directly
                logger.warning(f"Processing failed for {img_path}, trying to use original image")
                try:
                    # Open with PIL directly
                    original_pil = Image.open(str(img_path))
                    # Convert to RGB if needed
                    if original_pil.mode != 'RGB':
                        original_pil = original_pil.convert('RGB')
                    processed_images.append(original_pil)
                    
                    if debug:
                        debug_orig_path = Path("debug_output") / f"original_fallback_{i}.jpg"
                        original_pil.save(str(debug_orig_path))
                except Exception as img_error:
                    logger.error(f"Failed to use original image: {img_path}. Error: {str(img_error)}")
        
        if not processed_images:
            logger.error("No images were successfully processed")
            return False
        
        # Create PDF
        logger.info(f"Creating PDF: {output_path}")
        
        # Try multiple methods to create the PDF
        success = False
        
        # METHOD 1: Use img2pdf with BytesIO
        if not success:
            try:
                import io
                pdf_bytes = io.BytesIO()
                
                # Convert PIL images to bytes
                img_bytes_list = []
                for img in processed_images:
                    img_byte = io.BytesIO()
                    img.save(img_byte, format='JPEG', quality=95)
                    img_byte.seek(0)
                    img_bytes_list.append(img_byte.getvalue())
                
                # Use img2pdf to create PDF
                pdf_data = img2pdf.convert(img_bytes_list)
                
                # Write to file
                with open(output_path, 'wb') as f:
                    f.write(pdf_data)
                
                success = True
                logger.info("PDF created successfully using method 1 (img2pdf with BytesIO)")
            except Exception as e:
                logger.error(f"Error with method 1: {str(e)}")
        
        # METHOD 2: Save images to temporary files and use img2pdf
        if not success:
            temp_image_paths = []
            try:
                for i, img in enumerate(processed_images):
                    temp_path = Path("debug_output") / f"temp_img_{i}.jpg"
                    img.save(str(temp_path), "JPEG", quality=95)
                    temp_image_paths.append(str(temp_path))
                
                # Use img2pdf with file paths
                with open(output_path, "wb") as f:
                    f.write(img2pdf.convert(temp_image_paths))
                success = True
                logger.info("PDF created successfully using method 2 (img2pdf with file paths)")
            except Exception as e:
                logger.error(f"Error with method 2: {str(e)}")
        
        # METHOD 3: Use PIL to create PDF directly
        if not success:
            try:
                # Save the first image as PDF and append the rest
                processed_images[0].save(
                    str(output_path),
                    "PDF",
                    resolution=100.0,
                    save_all=True,
                    append_images=processed_images[1:] if len(processed_images) > 1 else []
                )
                success = True
                logger.info("PDF created successfully using method 3 (PIL direct PDF creation)")
            except Exception as e2:
                logger.error(f"Error with method 3: {str(e2)}")
        
        # METHOD 4: Use reportlab as a last resort
        if not success:
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
                import io
                
                c = canvas.Canvas(str(output_path), pagesize=letter)
                
                for i, img in enumerate(processed_images):
                    # Save image to bytes
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)
                    
                    # Add a new page for each image except the first
                    if i > 0:
                        c.showPage()
                    
                    # Add image to PDF
                    img_width, img_height = img.size
                    # Scale to fit on page
                    page_width, page_height = letter
                    scale = min(page_width / img_width, page_height / img_height) * 0.9
                    c.drawImage(img_byte_arr, 
                                x=(page_width - img_width * scale) / 2,
                                y=(page_height - img_height * scale) / 2,
                                width=img_width * scale,
                                height=img_height * scale)
                
                c.save()
                success = True
                logger.info("PDF created successfully using method 4 (reportlab)")
            except Exception as e3:
                logger.error(f"Error with method 4: {str(e3)}")
        
        if success:
            elapsed_time = time.time() - start_time
            logger.info(f"PDF created successfully in {elapsed_time:.2f} seconds")
            return True
        else:
            logger.error("All PDF creation methods failed")
            return False
        
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to parse arguments and execute the PDF scanner."""
    parser = argparse.ArgumentParser(description="Convert images to high-quality PDF documents.")
    
    parser.add_argument("input", help="Input image file or directory containing images")
    parser.add_argument("-o", "--output", help="Output PDF file path", default="output.pdf")
    parser.add_argument("-e", "--enhance", choices=["adaptive", "otsu", "normal"], 
                        default="adaptive", help="Image enhancement mode")
    parser.add_argument("-d", "--debug", action="store_true", 
                        help="Save intermediate processing steps as images")
    
    args = parser.parse_args()
    
    # Check if input exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    # Get list of image files
    image_paths = []
    if input_path.is_file():
        # Single file
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            image_paths = [input_path]
        else:
            logger.error(f"Unsupported file format: {input_path}")
            return 1
    else:
        # Directory
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']:
            image_paths.extend(sorted(input_path.glob(ext)))
        
        if not image_paths:
            logger.error(f"No supported image files found in directory: {input_path}")
            return 1
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process images and create PDF
    success = create_pdf(image_paths, output_path, args.enhance, args.debug)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())