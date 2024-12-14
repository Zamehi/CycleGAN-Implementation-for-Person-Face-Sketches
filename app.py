# command: python -m flask --app .\app.py run






from flask import Flask, render_template, request, send_file
from PIL import Image
import torch
import io
import torchvision.transforms as transforms
from generator import Generator 

app = Flask(__name__)

# Load the pre-trained CycleGAN model
generator_A = Generator(input_nc=3, output_nc=3)  # For face-to-sketch
generator_B = Generator(input_nc=3, output_nc=3)  # For sketch-to-face

checkpoint = torch.load('cycleGAN_epoch_34.pth', map_location=torch.device('cpu'))
generator_A.load_state_dict(checkpoint['generator_G_state_dict'])  # Load weights for face-to-sketch
generator_B.load_state_dict(checkpoint['generator_F_state_dict'])  # Load weights for sketch-to-face
generator_A.eval()
generator_B.eval()

# Transformations for input and output images
transform = transforms.Compose([
    
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

reverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.ToPILImage()
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/convert", methods=["POST"])
def convert_image():
    try:
        # Get the image from the form
        file = request.files['file']
        image = Image.open(file).convert("RGB")

        # Apply the pre-processing transformation
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Choose the model based on the conversion type
        if request.form['type'] == 'face_to_sketch':
            converted_image = generator_A(image_tensor)  # Use generator_A for face-to-sketch
        else:
            converted_image = generator_B(image_tensor)  # Use generator_B for sketch-to-face
        
        # Post-process the output image
        converted_image = converted_image.squeeze(0).detach().cpu()
        converted_image = reverse_transform(converted_image)

        # Save or return the image as needed
        output_io = io.BytesIO()
        converted_image.save(output_io, 'PNG')
        output_io.seek(0)

        return send_file(output_io, mimetype='image/png')
    
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
