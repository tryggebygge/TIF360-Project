import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

def apply_sepia(img_array):
    filter = np.array([[0.393, 0.769, 0.189],  # Red Channel
                       [0.349, 0.686, 0.168],  # Green Channel
                       [0.272, 0.534, 0.131]]) # Blue Channel
    sepia_img = np.dot(img_array, filter.T)
    sepia_img = np.where(sepia_img > 255, 255, sepia_img)
    return sepia_img.astype(np.uint8)

def apply_noise(img_array, strength=30):
    mean = 0
    var = strength
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, img_array.shape)
    noisy_img = img_array + gaussian
    noisy_img = np.clip(noisy_img, 0, 255)
    return noisy_img.astype(np.uint8)

def apply_vignette(img_array):
    height, width = img_array.shape[:2]
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)
    center_x, center_y = width / 2, height / 2
    radius = np.sqrt(center_x**2 + center_y**2)
    vignette = ((radius - np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)) / radius)
    vignette_power = 0.5
    vignette = vignette ** vignette_power
    vignette = np.stack([vignette] * 3, axis=-1)
    vignette_img = img_array * vignette
    vignette_img = np.clip(vignette_img, 0, 255)
    return vignette_img.astype(np.uint8)

def draw_random_line(img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    start = (random.randint(0, width), random.randint(0, height))
    end = (random.randint(0, width), random.randint(0, height))
    draw.line([start, end], fill='black', width=3)
    return img

def remove_random_corner(img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    corner = random.choice([
        (0, 0, width//3, height//3),
        (2*width//3, 0, width, height//3),
        (0, 2*height//3, width//3, height),
        (2*width//3, 2*height//3, width, height)
    ])
    draw.rectangle(corner, fill='black')
    return img

def apply_damage_effect(img_array, effect_type='line'):
    img = Image.fromarray(img_array)
    if effect_type == 'line':
        img = draw_random_line(img)
    elif effect_type == 'corner':
        img = remove_random_corner(img)
    return np.array(img)


def apply_blur(img_array):
    # Convert NumPy array to PIL Image
    img = Image.fromarray(img_array.astype('uint8'))
    # Apply Gaussian blur
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=1))
    # Convert PIL Image back to NumPy array
    return np.array(blurred_img)

def apply_stains(img_array):
    img = Image.fromarray(img_array.astype('uint8'))
    draw = ImageDraw.Draw(img)
    num_stains = random.randint(1, 5)  # You can adjust the number of stains if needed
    for _ in range(num_stains):
        # Starting points for the ellipse
        x0 = random.randint(0, img.width)
        y0 = random.randint(0, img.height)
        # End points for the ellipse are now closer to the starting points for smaller stains
        x1 = x0 + random.randint(10, 30)  # Smaller range for width of stains
        y1 = y0 + random.randint(10, 30)  # Smaller range for height of stains
        draw.ellipse([x0, y0, x1, y1], fill='brown', outline='brown')
    return np.array(img)

def apply_scratches(img_array):
    img = Image.fromarray(img_array.astype('uint8'))  # Convert numpy array to PIL Image
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 10)):  # Random number of scratches
        start = (random.randint(0, img.width), random.randint(0, img.height))
        end = (random.randint(0, img.width), random.randint(0, img.height))
        draw.line([start, end], fill='white', width=random.randint(1, 3))
    return np.array(img)  # Convert back to numpy array if needed

def apply_fading(img_array):
    img = Image.fromarray(img_array.astype('uint8'), 'RGB')  # Convert numpy array to PIL Image
    converter = ImageEnhance.Color(img)
    img_faded = converter.enhance(0.5)  # Reduce color saturation by 50%
    return np.array(img_faded)  # Convert back to numpy array if needed


def apply_combined_effects(img_array, number_of_effects=5):
    # Apply random effects
    for _ in range(number_of_effects):
        # Randomly select an effect from all effects
        list_of_effects = ['sepia', 'noise', 'vignette', 'damage_line', 'damage_corner', 'fading', 'scratches', 'stains', 'blur']
        effect_name = random.choice(list_of_effects)
        effect_function, kwargs = effects[effect_name]
        img_array = effect_function(img_array, **kwargs)
    return img_array


def process_image(image_path, output_dir, effect_function, effect_name, **kwargs):
    image = Image.open(image_path)
    img_array = np.array(image)
    processed_img_array = effect_function(img_array, **kwargs)
    output_subfolder = os.path.join(output_dir, effect_name)
    os.makedirs(output_subfolder, exist_ok=True)
    output_path = os.path.join(output_subfolder, f"{effect_name}_{os.path.basename(image_path)}")
    Image.fromarray(processed_img_array).save(output_path)
    print(f"Saved {effect_name} image to {output_path}")


input_dir = '/Users/malteaqvist/Library/CloudStorage/OneDrive-Chalmers/Chalmers Maskin/Aktiva kurser/TIF360 - Advanced machine learning with neural networks/gans-egen/00000'
output_dir = 'path_to_output_folder'
effects = {
    "sepia": (apply_sepia, {}),
    "noise": (apply_noise, {'strength': 50}),
    "vignette": (apply_vignette, {}),
    "damage_line": (apply_damage_effect, {'effect_type': 'line'}),
    "damage_corner": (apply_damage_effect, {'effect_type': 'corner'}),
    "fading": (apply_fading, {}),
    "scratches": (apply_scratches, {}),
    "stains": (apply_stains, {}),
    "blur": (apply_blur, {}),
    "combined_effects": (apply_combined_effects, {})    # Randomly apply 5 effects
}

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        input_path = os.path.join(input_dir, filename)
        for effect_name, (effect_function, kwargs) in effects.items():
            process_image(input_path, output_dir, effect_function, effect_name, **kwargs)
