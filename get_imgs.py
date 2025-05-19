import imageio
from PIL import Image
import os

# === Paths ===
raw_img_path = '/home/zimadmin/Documents/deconvolution/deform2self/2024_04_22_13_36_random_stim_duration_Anno_image_stack_1104px_43k_10_crop224px.avi'
output_dir = './data'
os.makedirs(output_dir, exist_ok=True)

# === Load video and convert ===
reader = imageio.get_reader(raw_img_path)

for i, frame in enumerate(reader):
    gray_frame = Image.fromarray(frame).convert('L')  # Force grayscale
    gray_frame.save(os.path.join(output_dir, f'{i:04d}.png'))

print(f"✅ Saved {i + 1} grayscale frames to {output_dir}")
