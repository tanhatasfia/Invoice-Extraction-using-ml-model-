import os
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

# Path to Tesseract executable (change as needed)
pytesseract.pytesseract.tesseract_cmd = r'D:\Extraction of invoice\process-env\Scripts\tesseract.exe'  # Update this path to your Tesseract executable

# ---------------------------------------------------------------------------

input_path = r'D:\Extraction of invoice\process-env\new samples'
output_path = r'D:\Extraction of invoice\Images'

# ---------------------------------------------------------------------------

os.makedirs(output_path, exist_ok=True)

# ---------------------------------------------------------------------------

def perform_ocr(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

# ---------------------------------------------------------------------------

for i, file_name in tqdm(enumerate(sorted(os.listdir(input_path)))):
    full_input_path = os.path.join(input_path, file_name)

    if file_name.endswith((".png", ".jpeg", ".jpg", ".tif")):
        image = Image.open(full_input_path).convert('RGB')
       
        output_image_path = os.path.join(output_path, f'jpg_image_{str(i).zfill(3)}.jpg')
        image.save(output_image_path)
        # Perform OCR on the saved image
        ocr_text = perform_ocr(output_image_path)
        print(f'OCR Result for {output_image_path}:\n{ocr_text}\n')

    elif file_name.endswith(".pdf"):
        try:
            # Convert all pages of the PDF to a list of images
            images = convert_from_path(full_input_path, poppler_path=r'D:\Extraction of invoice\process-env\Lib\site-packages\poppler-24.02.0\Library\bin')
            for j, image in enumerate(images):
                output_image_path = os.path.join(output_path, f'{os.path.splitext(file_name)[0]}_page_{str(j).zfill(3)}.jpg')
                image.save(output_image_path, 'JPEG')
                # Perform OCR on the saved image
                ocr_text = perform_ocr(output_image_path)
                print(f'OCR Result for {output_image_path}:\n{ocr_text}\n')
        except Exception as e:
            print(f"Error converting PDF '{file_name}': {e}")

print(f"There are {len(os.listdir(output_path))} images after standardizing image format and dimension.")
cd ''