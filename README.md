
# 🧠  AI Image Generator

> **Internship Role**: Generative AI Intern  
> **Company**: CodeCraft  
> **Task Title**: Image Generation using Pre-trained Models  
> **Tech Stack**: Google Colab, Hugging Face Diffusers, Stable Diffusion, Gradio, Python

---

## 🎯 Objective

The goal of this task was to build an **AI-powered image generator** using **pre-trained generative models** such as `DALL·E-mini` or `Stable Diffusion`. The model should accept text prompts and generate corresponding images using a clean user interface.

---
[Open In Colab](https://colab.research.google.com/drive/1lChhVmmFiXYykIdIHQoVzoMDfyYh_G_h?usp=sharing)

## 🚀 Project Summary

I successfully developed an **AI Image Generator** web app using the **Stable Diffusion v1-4** model from Hugging Face. The app is deployed using **Gradio** and hosted for free via **Google Colab** with a shareable link for public use.

---

## 🌐 Live Demo

🔗 **Try the Image Generator App Here**:  
👉 [Click to Launch AI Image Generator](https://29eb9964131f2fef8c.gradio.live/)



---

## 🛠️ Tools & Technologies Used

| Tool | Purpose |
|------|---------|
| `Python` | Programming language |
| `Hugging Face diffusers` | To load and run Stable Diffusion model |
| `Gradio` | For building the UI and API |
| `Google Colab` | To run the model using free GPU |
| `safetensors` | Safe model checkpointing and loading |

---

✅ Cell Code for Google Colab Image Generator - I Used


# 🚀 Install necessary libraries
!pip install -q gradio diffusers transformers accelerate safetensors

# 🧠 Import libraries
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# ⚙️ Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📦 Load Stable Diffusion v1-4 model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
).to(device)

# 🎨 Define generation function
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# 🌐 Launch Gradio app with public link
gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=1, placeholder="Try: A castle floating in the sky surrounded by waterfalls"),
    outputs="image",
    title="🎨 AI Image Generator",
    description="Enter a text prompt and generate a high-quality AI image using Stable Diffusion 1.4"
).launch(share=True)





🧠 How It Works:
Code Part	Purpose
!pip install ...	Installs required Python libraries
from diffusers import ...	Loads the model and image generation pipeline
device = "cuda" ...	Uses GPU if available (Colab usually provides it)
pipe(...)	Downloads and loads the Stable Diffusion model
generate_image()	Defines how to generate the image from prompt
.launch(share=True)	Starts the Gradio interface with a public link



## 🧪 Features

- ✅ Generate high-quality images from text prompts
- ✅ Simple, clean Gradio UI
- ✅ Public link to share with users
- ✅ Uses `CompVis/stable-diffusion-v1-4` for realistic results
- ✅ Runs on GPU (via Colab) with no login/token required

---

## 📚 Learning Outcomes

During this task, I learned:

- How to work with **pre-trained generative models**
- How to use **Hugging Face Diffusers and Transformers**
- How to build and share **Gradio web interfaces**
- How to handle deployment challenges (e.g., GPU limitations on Hugging Face)
- How to deploy apps on **Google Colab** with `share=True` for public use
- Fundamentals of **text-to-image generation** using deep learning

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `app.py` | Main Python file containing the image generation logic |
| `requirements.txt` | Python libraries needed to run the app |
| `README.md` | Project overview and documentation |

---

## 💡 Future Improvements

- Allow image size and style customization
- Add multiple image generation (num_images=2,3,...)
- Add gallery to view generated history
- Deploy permanently on Hugging Face (when GPU access available)

---

## 👏 Special Thanks

To the team at **CodeCraft** for this incredible opportunity and guidance throughout the internship!

---

> Built with ❤️ by Aniket Singh  
> Generative AI Intern, CodeCraft  
> [LinkedIn](https://linkedin.com/in/aniket-singh7as) | 

