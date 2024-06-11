from flask import Flask, request, jsonify, render_template, url_for
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the custom TensorFlow model
model_path = "archive"  # Ensure this is the correct path
model = tf.saved_model.load(model_path)
print(f"Model loaded from {model_path}")

def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def save_image(image_tensor, output_path):
    img = image_tensor.numpy()[0]  # Update this line
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(output_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    generated_image = request.args.get('image')
    return render_template('result.html', generated_image=generated_image)

@app.route('/apply-style', methods=['POST'])
def apply_style():
    try:
        gender = request.form.get('gender')
        content_image = request.files.get('content_image')
        style_images = request.files.getlist('style_image')
        style_options = request.form.getlist('style_options[]')

        if not content_image or not style_images or not gender or not style_options:
            return jsonify({'success': False, 'message': 'Missing required fields'})

        # Save the content image
        content_image_path = os.path.join(UPLOAD_FOLDER, content_image.filename)
        content_image.save(content_image_path)
        print(f"Content image saved to {content_image_path}")

        # Load the content image
        content_image_tensor = load_image(content_image_path)
        print(f"Content image loaded and processed")

        # Process each style image and apply the style to the content image
        result_image_urls = []
        for style_image in style_images:
            style_image_path = os.path.join(UPLOAD_FOLDER, style_image.filename)
            style_image.save(style_image_path)
            print(f"Style image saved to {style_image_path}")

            style_image_tensor = load_image(style_image_path)
            print(f"Style image loaded and processed")

            # Convert tensors to float32 before applying the style transfer
            content_image_tensor = tf.cast(content_image_tensor, tf.float32)
            style_image_tensor = tf.cast(style_image_tensor, tf.float32)

            # Apply the style transfer
            outputs = model(tf.constant(content_image_tensor), tf.constant(style_image_tensor))
            stylized_image_tensor = outputs[0]
            print(f"Style transfer applied")

            # Save the stylized image
            result_image_filename = f'stylized_{style_image.filename}'
            result_image_path = os.path.join(RESULT_FOLDER, result_image_filename)
            save_image(stylized_image_tensor, result_image_path)
            print(f"Stylized image saved to {result_image_path}")

            # Append the result image URL
            result_image_url = url_for('static', filename=f'results/{result_image_filename}')
            result_image_urls.append(result_image_url)

        # Return the result
        return jsonify({'success': True, 'images': result_image_urls})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
