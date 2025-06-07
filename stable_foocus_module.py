import requests
import os
import datetime
import base64
import io
from PIL import Image
import json
from flask import url_for

def encode_image_to_base64(image: Image.Image) -> str:
    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')
    image_data_bytes = image_data.getvalue()
    encoded_image = base64.b64encode(image_data_bytes).decode('utf-8')
    return encoded_image

def decode_base64_to_image(base64_image: str) -> Image.Image:
    pil_image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
    return pil_image

url = "http://localhost:8888/v2/generation/image-inpaint-outpaint"
url2 = "http://localhost:8888/v1/tools/generate_mask"
headers = {"accept": "image/png", "Content-Type": "application/json"}
headers2 = {"accept": "application/json", "Content-Type": "application/json"}

def maskinglowerfunc(image: str, preset_params: dict) -> dict:
    preset_params["image"] = image
    preset_params["cloth_category"] = "lower"
    data = json.dumps(preset_params)
    response = requests.post(url=url2, data=data, headers=headers2)
    return response.json()

def maskingupperfunc(image: str, preset_params: dict) -> dict:
    preset_params["image"] = image
    preset_params["cloth_category"] = "upper"
    data = json.dumps(preset_params)
    response = requests.post(url=url2, data=data, headers=headers2)
    return response.json()

def inpaint(input_image: str, input_mask: str, cn_img: str, preset_params: dict):
    preset_params["input_image"] = input_image
    preset_params["input_mask"] = input_mask
    preset_params["require_base64"] = True
    preset_params["async_process"] = False
    preset_params["advanced_params"]["mixing_image_prompt_and_inpaint"] = True
    preset_params["image_prompts"][0]["cn_img"] = cn_img
    preset_params["image_prompts"][0]["cn_stop"] = 1
    preset_params["image_prompts"][0]["cn_weight"] = 1
    data = json.dumps(preset_params)
    response = requests.post(url=url, data=data, headers=headers)
    return response

def stableFoocus(img_path, prompt_img_path, prompt_img1_path):
    try:
        img = Image.open(img_path).convert('RGB')
        prompt_img = Image.open(prompt_img_path).convert('RGB')
        prompt_img1 = Image.open(prompt_img1_path).convert('RGB')

        img_64bytes = encode_image_to_base64(image=img)

        payload2 = {
            "image": "",
            "mask_model": "u2net_cloth_seg",
            "cloth_category": "",
            "dino_prompt_text": "",
            "sam_model": "vit_b",
            "box_threshold": 0.3,
            "text_threshold": 0.25,
            "sam_max_detections": 0,
            "dino_erode_or_dilate": 0,
            "dino_debug": False
        }

        masklower_bytes = maskinglowerfunc(image=img_64bytes, preset_params=payload2)
        masklower = decode_base64_to_image(masklower_bytes)

        masklower_64bytes, prompt_img_64bytes = encode_image_to_base64(image=masklower), encode_image_to_base64(image=prompt_img)

        payload = {
            "prompt": "",
            "negative_prompt": "",
            "style_selections": [
                "Fooocus V2",
                "Fooocus Enhance",
                "Fooocus Sharp"
            ],
            "performance_selection": "Speed",
            "aspect_ratios_selection": "1152*896",
            "image_number": 1,
            "image_seed": -1,
            "sharpness": 2,
            "guidance_scale": 4,
            "base_model_name": "juggernautXL_v8Rundiffusion.safetensors",
            "refiner_model_name": "None",
            "refiner_switch": 0.5,
            "loras": [
                {"enabled": True, "model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.1},
                {"enabled": True, "model_name": "None", "weight": 1},
                {"enabled": True, "model_name": "None", "weight": 1},
                {"enabled": True, "model_name": "None", "weight": 1},
                {"enabled": True, "model_name": "None", "weight": 1}
            ],
            "advanced_params": {
                "adaptive_cfg": 7,
                "adm_scaler_end": 0.3,
                "adm_scaler_negative": 0.8,
                "adm_scaler_positive": 1.5,
                "black_out_nsfw": False,
                "canny_high_threshold": 128,
                "canny_low_threshold": 64,
                "clip_skip": 2,
                "controlnet_softness": 0.25,
                "debugging_cn_preprocessor": False,
                "debugging_dino": False,
                "debugging_enhance_masks_checkbox": False,
                "debugging_inpaint_preprocessor": False,
                "dino_erode_or_dilate": 0,
                "disable_intermediate_results": False,
                "disable_preview": False,
                "disable_seed_increment": False,
                "freeu_b1": 1.01,
                "freeu_b2": 1.02,
                "freeu_enabled": False,
                "freeu_s1": 0.99,
                "freeu_s2": 0.95,
                "inpaint_advanced_masking_checkbox": True,
                "inpaint_disable_initial_latent": False,
                "inpaint_engine": "v2.6",
                "inpaint_erode_or_dilate": 0,
                "inpaint_respective_field": 1,
                "inpaint_strength": 1,
                "invert_mask_checkbox": False,
                "mixing_image_prompt_and_inpaint": True,
                "mixing_image_prompt_and_vary_upscale": False,
                "overwrite_height": -1,
                "overwrite_step": -1,
                "overwrite_switch": -1,
                "overwrite_upscale_strength": -1,
                "overwrite_vary_strength": -1,
                "overwrite_width": -1,
                "refiner_swap_method": "joint",
                "sampler_name": "dpmpp_2m_sde_gpu",
                "scheduler_name": "karras",
                "skipping_cn_preprocessor": False,
                "vae_name": "Default (model)"
            },
            "save_meta": True,
            "meta_scheme": "fooocus",
            "save_extension": "png",
            "save_name": "",
            "read_wildcards_in_order": False,
            "require_base64": False,
            "async_process": False,
            "webhook_url": "",
            "input_image": "",
            "input_mask": "",
            "inpaint_additional_prompt": "",
            "outpaint_selections": [],
            "outpaint_distance_left": -1,
            "outpaint_distance_right": -1,
            "outpaint_distance_top": -1,
            "outpaint_distance_bottom": -1,
            "image_prompts": [{
                "cn_img": "string",
                "cn_stop": 0,
                "cn_weight": 0,
                "cn_type": "ImagePrompt"
            }]
        }

        inpaint_output_bytes = inpaint(input_image=img_64bytes, input_mask=masklower_64bytes, cn_img=prompt_img_64bytes, preset_params=payload)
        half_output_path = os.path.join("uploads", "half_output.png")
        with open(half_output_path, "wb") as f:
            f.write(inpaint_output_bytes.content)

        loaded_image = Image.open(half_output_path)
        inpaint_output = encode_image_to_base64(loaded_image)

        maskupper_bytes = maskingupperfunc(image=inpaint_output, preset_params=payload2)
        maskupper = decode_base64_to_image(maskupper_bytes)

        maskupper_64bytes, prompt_img1_64bytes = encode_image_to_base64(image=maskupper), encode_image_to_base64(image=prompt_img1)

        inpaint_output_bytes1 = inpaint(input_image=inpaint_output, input_mask=maskupper_64bytes, cn_img=prompt_img1_64bytes, preset_params=payload)

        full_output_path = os.path.join("uploads", "full_output.png")
        with open(full_output_path, "wb") as f:
            f.write(inpaint_output_bytes1.content)

        return url_for('uploaded_file', filename=os.path.basename(full_output_path))

    except Exception as e:
        raise







def stableFoocusLowerOnly(img_path, prompt_img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        prompt_img = Image.open(prompt_img_path).convert('RGB')

        img_64bytes = encode_image_to_base64(image=img)

        payload2 = {
            "image": "",
            "mask_model": "u2net_cloth_seg",
            "cloth_category": "",
            "dino_prompt_text": "",
            "sam_model": "vit_b",
            "box_threshold": 0.3,
            "text_threshold": 0.25,
            "sam_max_detections": 0,
            "dino_erode_or_dilate": 0,
            "dino_debug": False
        }

        masklower_bytes = maskinglowerfunc(image=img_64bytes, preset_params=payload2)
        masklower = decode_base64_to_image(masklower_bytes)

        masklower_64bytes, prompt_img_64bytes = encode_image_to_base64(image=masklower), encode_image_to_base64(image=prompt_img)

        payload = {
            "prompt": "",
            "negative_prompt": "",
            "style_selections": [
                "Fooocus V2",
                "Fooocus Enhance",
                "Fooocus Sharp"
            ],
            "performance_selection": "Speed",
            "aspect_ratios_selection": "1152*896",
            "image_number": 1,
            "image_seed": -1,
            "sharpness": 2,
            "guidance_scale": 4,
            "base_model_name": "juggernautXL_v8Rundiffusion.safetensors",
            "refiner_model_name": "None",
            "refiner_switch": 0.5,
            "loras": [
                {"enabled": True, "model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.1},
                {"enabled": True, "model_name": "None", "weight": 1},
                {"enabled": True, "model_name": "None", "weight": 1},
                {"enabled": True, "model_name": "None", "weight": 1},
                {"enabled": True, "model_name": "None", "weight": 1}
            ],
            "advanced_params": {
                "adaptive_cfg": 7,
                "adm_scaler_end": 0.3,
                "adm_scaler_negative": 0.8,
                "adm_scaler_positive": 1.5,
                "black_out_nsfw": False,
                "canny_high_threshold": 128,
                "canny_low_threshold": 64,
                "clip_skip": 2,
                "controlnet_softness": 0.25,
                "debugging_cn_preprocessor": False,
                "debugging_dino": False,
                "debugging_enhance_masks_checkbox": False,
                "debugging_inpaint_preprocessor": False,
                "dino_erode_or_dilate": 0,
                "disable_intermediate_results": False,
                "disable_preview": False,
                "disable_seed_increment": False,
                "freeu_b1": 1.01,
                "freeu_b2": 1.02,
                "freeu_enabled": False,
                "freeu_s1": 0.99,
                "freeu_s2": 0.95,
                "inpaint_advanced_masking_checkbox": True,
                "inpaint_disable_initial_latent": False,
                "inpaint_engine": "v2.6",
                "inpaint_erode_or_dilate": 0,
                "inpaint_respective_field": 1,
                "inpaint_strength": 1,
                "invert_mask_checkbox": False,
                "mixing_image_prompt_and_inpaint": True,
                "mixing_image_prompt_and_vary_upscale": False,
                "overwrite_height": -1,
                "overwrite_step": -1,
                "overwrite_switch": -1,
                "overwrite_upscale_strength": -1,
                "overwrite_vary_strength": -1,
                "overwrite_width": -1,
                "refiner_swap_method": "joint",
                "sampler_name": "dpmpp_2m_sde_gpu",
                "scheduler_name": "karras",
                "skipping_cn_preprocessor": False,
                "vae_name": "Default (model)"
            },
            "save_meta": True,
            "meta_scheme": "fooocus",
            "save_extension": "png",
            "save_name": "",
            "read_wildcards_in_order": False,
            "require_base64": False,
            "async_process": False,
            "webhook_url": "",
            "input_image": "",
            "input_mask": "",
            "inpaint_additional_prompt": "",
            "outpaint_selections": [],
            "outpaint_distance_left": -1,
            "outpaint_distance_right": -1,
            "outpaint_distance_top": -1,
            "outpaint_distance_bottom": -1,
            "image_prompts": [{
                "cn_img": "string",
                "cn_stop": 0,
                "cn_weight": 0,
                "cn_type": "ImagePrompt"
            }]
        }

        inpaint_output_bytes = inpaint(input_image=img_64bytes, input_mask=masklower_64bytes, cn_img=prompt_img_64bytes, preset_params=payload)
        half_output_path = os.path.join("uploads", "half_output.png")
        with open(half_output_path, "wb") as f:
            f.write(inpaint_output_bytes.content)


        return url_for('uploaded_file', filename=os.path.basename(half_output_path))

    except Exception as e:
        raise




def stableFoocusOnlyhalf(img_path, prompt_img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        prompt_img = Image.open(prompt_img_path).convert('RGB')

        img_64bytes = encode_image_to_base64(image=img)

        payload2 = {
            "image": "",
            "mask_model": "u2net_cloth_seg",
            "cloth_category": "",
            "dino_prompt_text": "",
            "sam_model": "vit_b",
            "box_threshold": 0.3,
            "text_threshold": 0.25,
            "sam_max_detections": 0,
            "dino_erode_or_dilate": 0,
            "dino_debug": False
        }

        masklower_bytes = maskinglowerfunc(image=img_64bytes, preset_params=payload2)
        masklower = decode_base64_to_image(masklower_bytes)

        masklower_64bytes, prompt_img_64bytes = encode_image_to_base64(image=masklower), encode_image_to_base64(image=prompt_img)

        payload = {
            "prompt": "",
            "negative_prompt": "",
            "style_selections": [
                "Fooocus V2",
                "Fooocus Enhance",
                "Fooocus Sharp"
            ],
            "performance_selection": "Speed",
            "aspect_ratios_selection": "1152*896",
            "image_number": 1,
            "image_seed": -1,
            "sharpness": 2,
            "guidance_scale": 4,
            "base_model_name": "juggernautXL_v8Rundiffusion.safetensors",
            "refiner_model_name": "None",
            "refiner_switch": 0.5,
            "loras": [
                {"enabled": True, "model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.1},
                {"enabled": True, "model_name": "None", "weight": 1},
                {"enabled": True, "model_name": "None", "weight": 1},
                {"enabled": True, "model_name": "None", "weight": 1},
                {"enabled": True, "model_name": "None", "weight": 1}
            ],
            "advanced_params": {
                "adaptive_cfg": 7,
                "adm_scaler_end": 0.3,
                "adm_scaler_negative": 0.8,
                "adm_scaler_positive": 1.5,
                "black_out_nsfw": False,
                "canny_high_threshold": 128,
                "canny_low_threshold": 64,
                "clip_skip": 2,
                "controlnet_softness": 0.25,
                "debugging_cn_preprocessor": False,
                "debugging_dino": False,
                "debugging_enhance_masks_checkbox": False,
                "debugging_inpaint_preprocessor": False,
                "dino_erode_or_dilate": 0,
                "disable_intermediate_results": False,
                "disable_preview": False,
                "disable_seed_increment": False,
                "freeu_b1": 1.01,
                "freeu_b2": 1.02,
                "freeu_enabled": False,
                "freeu_s1": 0.99,
                "freeu_s2": 0.95,
                "inpaint_advanced_masking_checkbox": True,
                "inpaint_disable_initial_latent": False,
                "inpaint_engine": "v2.6",
                "inpaint_erode_or_dilate": 0,
                "inpaint_respective_field": 1,
                "inpaint_strength": 1,
                "invert_mask_checkbox": False,
                "mixing_image_prompt_and_inpaint": True,
                "mixing_image_prompt_and_vary_upscale": False,
                "overwrite_height": -1,
                "overwrite_step": -1,
                "overwrite_switch": -1,
                "overwrite_upscale_strength": -1,
                "overwrite_vary_strength": -1,
                "overwrite_width": -1,
                "refiner_swap_method": "joint",
                "sampler_name": "dpmpp_2m_sde_gpu",
                "scheduler_name": "karras",
                "skipping_cn_preprocessor": False,
                "vae_name": "Default (model)"
            },
            "save_meta": True,
            "meta_scheme": "fooocus",
            "save_extension": "png",
            "save_name": "",
            "read_wildcards_in_order": False,
            "require_base64": False,
            "async_process": False,
            "webhook_url": "",
            "input_image": "",
            "input_mask": "",
            "inpaint_additional_prompt": "",
            "outpaint_selections": [],
            "outpaint_distance_left": -1,
            "outpaint_distance_right": -1,
            "outpaint_distance_top": -1,
            "outpaint_distance_bottom": -1,
            "image_prompts": [{
                "cn_img": "string",
                "cn_stop": 0,
                "cn_weight": 0,
                "cn_type": "ImagePrompt"
            }]
        }

        inpaint_output_bytes = inpaint(input_image=img_64bytes, input_mask=masklower_64bytes, cn_img=prompt_img_64bytes, preset_params=payload)
        half_output_path = os.path.join("uploads", "half_output.png")
        with open(half_output_path, "wb") as f:
            f.write(inpaint_output_bytes.content)


        return url_for('uploaded_file', filename=os.path.basename(half_output_path))

    except Exception as e:
        raise
