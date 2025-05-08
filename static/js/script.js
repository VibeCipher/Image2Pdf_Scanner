document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const submitButton = document.getElementById('submit-button');
    const browseButton = document.getElementById('browse-button');
    
    // Files array to store selected files
    let selectedFiles = [];
    
    // Add click event to browse button
    browseButton.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('dragover');
    }
    
    function unhighlight() {
        dropArea.classList.remove('dragover');
    }
    
    // Handle dropped files
    dropArea.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });
    
    function handleFiles(files) {
        const validFiles = Array.from(files).filter(file => {
            const extension = file.name.split('.').pop().toLowerCase();
            // Check both extension and MIME type
            const validExtension = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif', 'webp'].includes(extension);
            const validMimeType = file.type.startsWith('image/');
            return validExtension && validMimeType;
        });
        
        if (validFiles.length === 0) {
            alert('Please select valid image files (JPG, PNG, BMP, TIFF, etc.).');
            return;
        }
        
        // Add files to file list
        validFiles.forEach(file => {
            addFileToList(file);
        });
        
        // Update file input with selected files
        updateFileInput();
        
        // Enable submit button if files are selected
        updateSubmitButton();
    }
    
    function addFileToList(file) {
        // Check if file already exists in the list
        if (selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
            return;
        }
        
        // Add file to array
        selectedFiles.push(file);
        
        // Create file item element
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        // Add file icon
        const fileIcon = document.createElement('i');
        fileIcon.className = 'fas fa-file-image file-icon';
        fileItem.appendChild(fileIcon);
        
        // Add file name
        const fileName = document.createElement('span');
        fileName.className = 'file-name';
        fileName.textContent = file.name;
        fileItem.appendChild(fileName);
        
        // Add remove button
        const removeBtn = document.createElement('i');
        removeBtn.className = 'fas fa-times remove-file';
        removeBtn.addEventListener('click', function() {
            // Remove file from array
            selectedFiles = selectedFiles.filter(f => f !== file);
            
            // Remove file item from list
            fileItem.remove();
            
            // Update file input
            updateFileInput();
            
            // Update submit button
            updateSubmitButton();
        });
        fileItem.appendChild(removeBtn);
        
        // Add file item to list
        fileList.appendChild(fileItem);
    }
    
    function updateFileInput() {
        // Create a new DataTransfer object
        const dataTransfer = new DataTransfer();
        
        // Add selected files to DataTransfer object
        selectedFiles.forEach(file => {
            dataTransfer.items.add(file);
        });
        
        // Set files property of file input
        fileInput.files = dataTransfer.files;
    }
    
    function updateSubmitButton() {
        submitButton.disabled = selectedFiles.length === 0;
    }
});