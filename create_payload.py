import base64
import json
import os

# Path to your test image
image_path = r"C:\Users\Administrator\deepfake_detection_project\Dataset\Test\Fake\fake_28.jpg"

# Output payload path
output_json_path = r"C:\Users\Administrator\deepfake_detection_project\payload.json"

def create_payload(image_path, output_json_path):
    with open(image_path, "rb") as img_file:
        b64_str = base64.b64encode(img_file.read()).decode('utf-8')

    payload = {
        "image_data": b64_str
    }

    with open(output_json_path, "w") as json_file:
        json.dump(payload, json_file)

    print(f"Payload JSON created at: {output_json_path}")

if __name__ == "__main__":
    if os.path.exists(image_path):
        create_payload(image_path, output_json_path)
    else:
        print(f"Image file does not exist: {image_path}")
