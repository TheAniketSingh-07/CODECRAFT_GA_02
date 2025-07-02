import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")


def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image


interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Enter your image prompt here...", label="Prompt"),
    outputs=gr.Image(type="pil"),
    title="🪐 AI Image Generator - Stable Diffusion",
    description="Type a prompt, click 'Generate', and see your AI-generated image!"
)
