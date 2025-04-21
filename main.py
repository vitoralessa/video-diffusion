from diffusers import StableVideoDiffusionPipeline
import torch
from PIL import Image

# Carrega a pipeline (pode demorar)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# Abre a imagem base
image = Image.open("input.jpg").convert("RGB").resize((512, 512))

# Gera o vídeo
video_frames = pipe(image, num_frames=14, decode_chunk_size=4).frames[0]

# Salva como GIF (ou depois converte para .mp4 com ffmpeg)
video_frames[0].save("output.gif", save_all=True, append_images=video_frames[1:], duration=100, loop=0)