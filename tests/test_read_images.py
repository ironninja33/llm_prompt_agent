import pprint
import os
import json
from PIL import Image

def read_image_png(filepath):
    img = Image.open(filepath)
    info = img.info
    pprint.pprint(info["prompt"])

def _extract_from_jpg_exif(filepath: str) -> ParsedImageData | None:
    """Try to extract prompt from EXIF data."""
    try:
        img = Image.open(filepath)
        
        # Try to get EXIF data
        exif_data = img.getexif()
        img.close()
        
        if not exif_data:
            return None
        
        # UserComment tag (0x9286)
        user_comment = exif_data.get(0x9286)
        if user_comment:
            text = user_comment
            if isinstance(text, bytes):
                # Skip encoding prefix if present (ASCII, UNICODE, JIS, etc.)
                if text[:8] in (b"ASCII\x00\x00\x00", b"UNICODE\x00"):
                    text = text[8:]
                text = text.decode("utf-8", errors="ignore")
            
            return _parse_prompt_text(text, filepath)
        
        # ImageDescription tag (0x010E)
        description = exif_data.get(0x010E)
        if description:
            return _parse_prompt_text(str(description), filepath)
            
        return None
        
    except Exception as e:
        print(f"No EXIF data in {filepath}: {e}")
        return None


data_root = "/mnt/linux_extra/projects/llm_prompt_agent/data/examples/output/asa_akira/"
for filename in os.listdir(data_root):
    path = os.path.join(data_root, filename)
    read_image_png(path)
    break