import os
import torch
import time
import argparse
from fastapi import FastAPI, HTTPException, Body
import uvicorn
from diffusers import SanaPipeline

from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

from rembg import remove, new_session

class Removebg():
    def __init__(self, name="u2net"):
        '''
            name: rembg
        '''
        self.session = new_session(name)

    def __call__(self, rgb_img, force=False):
        '''
            inputs:
                rgb_img: PIL.Image, with RGB mode expected
                force: bool, input is RGBA mode
            return:
                rgba_img: PIL.Image with RGBA mode
        '''
        if rgb_img.mode == "RGBA":
            if force:
                rgb_img = rgb_img.convert("RGB")
            else:
                return rgb_img
        rgba_img = remove(rgb_img, session=self.session)
        return rgba_img

app = FastAPI()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lite", default=False, action="store_true")
    parser.add_argument("--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str)
    parser.add_argument("--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str)
    parser.add_argument("--text2image_path", default="weights/hunyuanDiT", type=str)
    parser.add_argument("--save_folder", default="outputs/", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--t2i_seed", default=0, type=int)
    parser.add_argument("--t2i_steps", default=25, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--gen_steps", default=50, type=int)
    parser.add_argument("--max_faces_num", default=90000, type=int)
    parser.add_argument("--save_memory", default=False, action="store_true")
    parser.add_argument("--do_texture_mapping", default=False, action="store_true")
    parser.add_argument("--do_render", default=False, action="store_true")
    parser.add_argument("--port", default=8093, type=int)
    return parser.parse_args()

args = get_args()


pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    torch_dtype=torch.float16,
    device_map="balanced",
)
pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)
pipe.transformer = pipe.transformer.to(torch.bfloat16)

def sana_gen_image(prompt,device="cuda:0", seed=23, steps=20):
    '''
        inputs:
            prompr: str
            seed: int
            steps: int
        return:
            rgb: PIL.Image
    '''
    # prompt = prompt + ' white background cartoon 3D three-dimensional'
    # prompt = prompt + ' white background 3D'
    # prompt = prompt + ', style: no shadows, white background cartoon 3D three-dimensional'
    # prompt = prompt + ', style: white background cartoon 3D three-dimensional, no shadows'
    # prompt = prompt + ' white background cartoon 3D three-dimensional, no shadows'
    # prompt = prompt + ', style: white background 3D three-dimensional, no shadows'
    # prompt = prompt + ', style: white background, 3D, no shadows'
    # prompt = prompt + ', style: white background, 3D, three-dimensional, no shadows'
    prompt = prompt + ', style: white background 3D three-dimensional, no shadows'
    rgb = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        num_inference_steps=steps,
        generator=torch.Generator(device=device).manual_seed(seed),
        return_dict=False
    )[0][0]
    torch.cuda.empty_cache()
    return rgb


# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()


# Initialize models globally
rembg_model = Removebg()

def process_image_to_3d(res_rgb_pil, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Stage 2: Remove Background
    res_rgba_pil = rembg_model(res_rgb_pil)
    res_rgb_pil.save(os.path.join(output_folder, "img_nobg.png"))

        # Run the pipeline
    outputs = pipeline.run(
        res_rgba_pil,
        seed=1,
        # Optional parameters
        # sparse_structure_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 7.5,
        # },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(os.path.join(output_folder, "mesh.glb"))
    glb.export(os.path.join(output_folder, "mesh.obj"))

@app.post("/generate_from_text")
async def text_to_3d(prompt: str = Body()):
    output_folder = os.path.join(args.save_folder, "text_to_3d")
    os.makedirs(output_folder, exist_ok=True)

    # Stage 1: Text to Image
    start = time.time()
    res_rgb_pil = sana_gen_image(
        prompt,
        seed=args.t2i_seed,
        steps=args.t2i_steps
    )
    res_rgb_pil.save(os.path.join(output_folder, "img.jpg"))

    process_image_to_3d(res_rgb_pil, output_folder)
    
    print(f"Successfully generated: {output_folder}")
    print(f"Generation time: {time.time() - start}")

    return {"success": True, "path": output_folder}

@app.post("/generate_from_image")
async def image_to_3d(image_path: str):
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="Image file not found")

    start = time.time()
    output_folder = os.path.join(args.save_folder, "image_to_3d")
    os.makedirs(output_folder, exist_ok=True)

    # Load Image
    res_rgb_pil = Image.open(image_path)
    process_image_to_3d(res_rgb_pil, output_folder)
    
    print(f"Successfully generated: {output_folder}")
    print(f"Generation time: {time.time() - start}")

    return {"message": "3D model generated successfully from image", "output_folder": output_folder}

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)