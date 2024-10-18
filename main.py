from diffusers import StableDiffusionPipeline
import torch

# Cargar el modelo
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Usa GPU para acelerar el proceso

# Configura tu prompt
prompt = "A cup of coffee in a high-quality photographic and cinematic style"

# Genera la imagen
image = pipe(prompt, height=512, width=512, num_inference_steps=50).images[0]

# Guarda la imagen
image.save("taza_de_cafe.png")
