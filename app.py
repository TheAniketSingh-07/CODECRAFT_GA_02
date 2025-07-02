import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

def generate(prompt):
    image = pipe(prompt).images[0]
    return image

gr.Interface(fn=generate, inputs="text", outputs="image", title="AI Image Generator").launch()
