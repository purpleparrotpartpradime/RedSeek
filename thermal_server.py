from flask import Flask, request, send_file
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)

def apply_thermal_effect(image_bytes):
    # Decode the image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thermal colormap
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Encode the result to JPEG
    _, output = cv2.imencode('.jpg', heatmap)
    return output.tobytes()

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    processed = apply_thermal_effect(file.read())

    # Save to temporary file to send via send_file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp.write(processed)
    temp.flush()
    temp.seek(0)

    # Serve the file and then remove it
    response = send_file(temp.name, mimetype='image/jpeg')
    @response.call_on_close
    def cleanup():
        os.unlink(temp.name)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
