import os
import re
import time
import torch
import numpy as np
from PIL import Image
import io
import base64
import json
import requests as http_requests
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from google.oauth2.credentials import Credentials

# --- Configuration Logic (Modified to store GCP Project Info) ---
CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "uncut_vertex_config.json")

TOKEN_ENDPOINT_PATH = "/get_vertex_token"
DEFAULT_PORT = "9191"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception: pass
    return {"project_id": "your-project-id", "location": "us-central1", "server_ip": ""}

def save_config(project_id, location, server_ip=""):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"project_id": project_id, "location": location, "server_ip": server_ip}, f)
    except Exception as e:
        print(f"Warning: Could not save config: {e}")

def fetch_vertex_token(server_ip):
    host = server_ip.strip()
    if ":" not in host:
        host = f"{host}:{DEFAULT_PORT}"
    url = f"http://{host}{TOKEN_ENDPOINT_PATH}"
    try:
        resp = http_requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise Exception(f"No access_token in response: {data}")
        return token
    except Exception as e:
        raise Exception(f"Failed to fetch Vertex token from {url}: {e}")

def interpret_safety_error(error_message):
    SAFETY_ERROR_CODES = {
        "58061214": "Child", "17301594": "Child",
        "29310472": "Celebrity", "15236754": "Celebrity",
        "62263041": "Dangerous content",
        "57734940": "Hate", "22137204": "Hate",
        "74803281": "Other", "29578790": "Other", "42876398": "Other",
        "39322892": "People/Face",
        "92201652": "Personal information",
        "89371032": "Prohibited content", "49114662": "Prohibited content", "72817394": "Prohibited content",
        "90789179": "Sexual", "63429089": "Sexual", "43188360": "Sexual",
        "78610348": "Toxic",
        "61493863": "Violence", "56562880": "Violence",
        "32635315": "Vulgar",
        "64151117": "Celebrity or child"
    }
    
    err_str = str(error_message)
    match = re.search(r"Support codes?:?\s*([0-9\s,]+)", err_str, re.IGNORECASE)
    if match:
        codes_str = match.group(1)
        found_codes = re.findall(r"\d+", codes_str)
        reasons = []
        for code in found_codes:
            if code in SAFETY_ERROR_CODES:
                reasons.append(SAFETY_ERROR_CODES[code])
        
        if reasons:
            unique_reasons = list(set(reasons))
            return f"{err_str}\n\n[Safety Filter Triggered] Explicit Reason(s): {', '.join(unique_reasons)} (Codes: {', '.join(found_codes)})"
            
    return err_str

# Helper functions
def comfy_tensor_to_pil(tensor):
    if tensor is None:
        return None
    
    # Ensure tensor is on CPU
    tensor = tensor.cpu()
    
    # Handle single image without batch dim
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    pil_images = []
    for i in range(tensor.shape[0]):
        # Convert tensor to numpy, scale to 0-255, cast to uint8
        image_np = (tensor[i].numpy() * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        pil_images.append(image)
    return pil_images

def pil_to_comfy_tensor(pil_image):
    if isinstance(pil_image, list):
        images = pil_image
    else:
        images = [pil_image]
    
    tensors = []
    for image in images:
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(image_np))
    
    if tensors:
        return torch.stack(tensors, dim=0)
    else:
        # Return empty tensor if input is empty
        return torch.empty((0, 0, 0, 3))

class NanoBananaProNodeVertex:
    @classmethod
    def INPUT_TYPES(s):
        current_config = load_config()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Generate a fashion model..."}),
                "system_instruction": ("STRING", {"multiline": True, "default": "Professional fashion photography assistant."}),
                "project_id": ("STRING", {"default": current_config.get("project_id")}),
                "location": ("STRING", {"default": current_config.get("location", "us-central1")}),
                "server_ip": ("STRING", {"default": current_config.get("server_ip", "")}),
                "aspect_ratio": (["1:1", "2:3", "3:2", "4:3", "3:4", "9:16", "16:9"], {"default": "1:1"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "model": (["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"], {"default": "gemini-3-pro-image-preview"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {"reference_images": ("IMAGE",)}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate_nano"
    CATEGORY = "UncutNodes"

    def generate_nano(self, prompt, project_id, location, server_ip, aspect_ratio, resolution, seed, system_instruction, model, reference_images=None):
        save_config(project_id, location, server_ip)

        client_kwargs = dict(vertexai=True, project=project_id, location=location)

        if server_ip.strip():
            token = fetch_vertex_token(server_ip)
            client_kwargs["credentials"] = Credentials(token=token)

        client = genai.Client(**client_kwargs)

        processed_images = comfy_tensor_to_pil(reference_images) if reference_images is not None else []

        try:
            contents = []
            if processed_images:
                contents.extend(processed_images)
            contents.append(prompt)

            max_retries = 5
            retry_delay = 10  # Start with 10 seconds for 429 errors

            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_modalities=['IMAGE'],
                            image_config=types.ImageConfig(
                                aspect_ratio=aspect_ratio,
                                image_size=resolution 
                            ),
                            seed=seed % 2147483647,
                            # Vertex AI IAM allows setting thresholds to OFF
                            safety_settings=[
                                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                            ],
                        )
                    )
                    break # Success!
                except ClientError as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        print(f"Vertex AI quota exceeded (429). Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise Exception(interpret_safety_error(e)) # Re-raise interpreted error
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        # Fallback for other exceptions that might contain 429
                        print(f"Vertex AI quota exceeded (429) - Generic Exception. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise Exception(interpret_safety_error(e))

            # Output processing logic
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                
                # Check for finish reason indicating safety block
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    # Convert to string just in case it's an enum
                    reason = str(candidate.finish_reason)
                    if reason not in ["STOP", "FinishReason.STOP"]:
                        print(f"Vertex AI Finish Reason: {reason}")
                        if "SAFETY" in reason or "BLOCK" in reason or "PROHIBITED" in reason:
                             raise Exception(f"Image generation blocked by safety filters. Reason: {reason}")

                if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                    part = candidate.content.parts[0]
                    if part.inline_data:
                        # Debugging: Print data type and size
                        print(f"Received inline_data. Mime: {part.inline_data.mime_type}")
                        
                        raw_data = part.inline_data.data
                        
                        # Check if data is already bytes or needs decoding
                        if isinstance(raw_data, bytes):
                            # If it's bytes, it might be the raw image or base64 bytes
                            # Try to open as image directly first
                            try:
                                image = Image.open(io.BytesIO(raw_data))
                                image.verify() # Verify it's a valid image
                                image = Image.open(io.BytesIO(raw_data)) # Re-open after verify
                                print("Data was raw bytes.")
                                return (pil_to_comfy_tensor(image),)
                            except Exception:
                                # If not a valid image, try base64 decoding
                                try:
                                    img_bytes = base64.b64decode(raw_data)
                                    return (pil_to_comfy_tensor(Image.open(io.BytesIO(img_bytes))),)
                                except Exception as e:
                                    print(f"Failed to decode bytes: {e}")
                        
                        elif isinstance(raw_data, str):
                            # If string, it's likely base64
                            try:
                                img_bytes = base64.b64decode(raw_data)
                                return (pil_to_comfy_tensor(Image.open(io.BytesIO(img_bytes))),)
                            except Exception as e:
                                 print(f"Failed to decode string: {e}")
                    
                    # Handle text refusal (sometimes model returns text saying it can't generate)
                    if part.text:
                         print(f"Vertex AI Refusal Text: {part.text}")
                         raise Exception(f"Model refused to generate image: {part.text}")

            # If no candidates or no content, check prompt feedback if available
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                 print(f"Prompt Feedback: {response.prompt_feedback}")
                 raise Exception(f"Prompt blocked. Feedback: {response.prompt_feedback}")

            raise Exception("No image generated. Check console for safety block reasons.")

        except Exception as e:
            print(f"!!!! Vertex AI Error !!!!\n{e}\n")
            raise e

NODE_CLASS_MAPPINGS = {
    "NanoBananaProNodeVertex": NanoBananaProNodeVertex
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaProNodeVertex": "Uncut NBP (Vertex AI)"
}

# This explicitly tells ComfyUI what to export
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']