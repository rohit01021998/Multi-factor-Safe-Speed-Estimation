#!/usr/bin/env python3
"""
Optimize PNG images for web display by resizing and compressing them.
This script reduces file sizes while maintaining good quality for README display.
"""

from PIL import Image, ImageOps
import os

def optimize_image(input_path, max_width=800, max_height=600, quality=85, output_format='PNG'):
    """
    Optimize an image by resizing and compressing it.
    
    Args:
        input_path (str): Path to the input image
        max_width (int): Maximum width in pixels
        max_height (int): Maximum height in pixels
        quality (int): JPEG quality (1-100, only used for JPEG)
        output_format (str): Output format ('PNG' or 'JPEG')
    """
    # Get file info
    file_size_before = os.path.getsize(input_path)
    
    with Image.open(input_path) as img:
        print(f"\nProcessing: {os.path.basename(input_path)}")
        print(f"  Original size: {img.size[0]}x{img.size[1]} pixels")
        print(f"  Original file size: {file_size_before/1024:.1f} KB")
        
        # Calculate new dimensions while maintaining aspect ratio
        original_width, original_height = img.size
        
        # Only resize if image is larger than max dimensions
        if original_width > max_width or original_height > max_height:
            # Calculate scaling factor
            width_ratio = max_width / original_width
            height_ratio = max_height / original_height
            scale_factor = min(width_ratio, height_ratio)
            
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Resize with high-quality resampling
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"  Resized to: {new_width}x{new_height} pixels")
        else:
            img_resized = img.copy()
            print(f"  No resizing needed (within {max_width}x{max_height})")
        
        # Determine output filename and format
        base_name = os.path.splitext(input_path)[0]
        
        if output_format.upper() == 'JPEG':
            output_path = f"{base_name}_optimized.jpg"
            # Convert to RGB if necessary (JPEG doesn't support transparency)
            if img_resized.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparent images
                white_bg = Image.new('RGB', img_resized.size, (255, 255, 255))
                if img_resized.mode == 'RGBA':
                    white_bg.paste(img_resized, mask=img_resized.split()[-1])
                else:
                    white_bg.paste(img_resized.convert('RGBA'), mask=img_resized.convert('RGBA').split()[-1])
                img_resized = white_bg
            
            # Save as JPEG with optimization
            img_resized.save(output_path, 'JPEG', quality=quality, optimize=True)
            
        else:  # PNG
            output_path = f"{base_name}_optimized.png"
            
            # Optimize PNG
            img_resized.save(output_path, 'PNG', optimize=True, compress_level=9)
        
        # Check new file size
        file_size_after = os.path.getsize(output_path)
        reduction_percent = ((file_size_before - file_size_after) / file_size_before) * 100
        
        print(f"  Output: {os.path.basename(output_path)}")
        print(f"  New file size: {file_size_after/1024:.1f} KB")
        print(f"  Size reduction: {reduction_percent:.1f}%")
        
        return output_path

def optimize_all_images():
    """Optimize all PNG images in the Images directory."""
    images_dir = "Images"
    
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} directory not found!")
        return
    
    # Find PNG files (excluding backups and already optimized files)
    png_files = [f for f in os.listdir(images_dir) 
                 if f.lower().endswith('.png') 
                 and not f.endswith('_backup.png') 
                 and not f.endswith('_optimized.png')]
    
    if not png_files:
        print(f"No PNG files found in {images_dir} directory!")
        return
    
    print("Image Optimization Settings:")
    print("- Max dimensions: 800x600 pixels")
    print("- PNG: Maximum compression")
    print("- JPEG alternative: 85% quality")
    print("=" * 60)
    
    results = []
    
    for png_file in png_files:
        file_path = os.path.join(images_dir, png_file)
        
        try:
            # Create both PNG and JPEG versions for comparison
            png_output = optimize_image(file_path, max_width=800, max_height=600, 
                                      output_format='PNG')
            jpeg_output = optimize_image(file_path, max_width=800, max_height=600, 
                                       quality=85, output_format='JPEG')
            
            # Compare file sizes and recommend the smaller one
            png_size = os.path.getsize(png_output)
            jpeg_size = os.path.getsize(jpeg_output)
            
            if jpeg_size < png_size * 0.7:  # If JPEG is significantly smaller
                print(f"  → Recommend JPEG version (70%+ smaller)")
                recommended = jpeg_output
            else:
                print(f"  → Recommend PNG version (better quality/transparency)")
                recommended = png_output
            
            results.append({
                'original': png_file,
                'png_optimized': png_output,
                'jpeg_optimized': jpeg_output,
                'recommended': recommended
            })
            
        except Exception as e:
            print(f"Error processing {png_file}: {e}")
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY:")
    print("=" * 60)
    
    for result in results:
        original_size = os.path.getsize(os.path.join(images_dir, result['original']))
        recommended_size = os.path.getsize(result['recommended'])
        savings = ((original_size - recommended_size) / original_size) * 100
        
        print(f"\n{result['original']}:")
        print(f"  Original: {original_size/1024:.1f} KB")
        print(f"  Recommended: {os.path.basename(result['recommended'])} ({recommended_size/1024:.1f} KB)")
        print(f"  Total savings: {savings:.1f}%")
    
    print(f"\nOptimized images are saved with '_optimized' suffix.")
    print("You can replace the originals with the optimized versions if satisfied with quality.")

if __name__ == "__main__":
    optimize_all_images()