import itertools
import os
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image


def main(
    prompt_label: str,
    prompt: str,
    negative_prompt: str,
    use_negative: bool,
    n_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    guidance_rescale: float,
    output_folder: str,
):
    print(f"\nRunning: prompt={prompt_label}, neg={use_negative}, "
          f"steps={n_steps}, size={height}x{width}, gs={guidance_scale}, "
          f"gr={guidance_rescale}")

    # -------------------------------------------------------
    # Load SDXL BASE + IP-Adapter
    # -------------------------------------------------------
    print("Loading SDXL Base + IP-Adapter...")

    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
        variant="fp16",
        torch_dtype=torch.float16,
    )

    base.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.safetensors",
    )

    base.set_ip_adapter_scale(0.8)
    base.enable_model_cpu_offload()

    # -------------------------------------------------------
    # Load REFINER (same pattern as your LoRA script)
    # -------------------------------------------------------
    print("Loading SDXL Refiner...")

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        use_safetensors=True,
        variant="fp16",
        torch_dtype=torch.float16,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
    )

    refiner.enable_model_cpu_offload()

    # Load the reference image ONCE globally outside loops
    ip_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png"
    )


    # -------------------------------------------------------
    # BASE stage → latent only
    # -------------------------------------------------------
    print("BASE stage running (latent generation)...")

    base_args = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=n_steps,
        denoising_end=0.8,                 # fixed like your original example
        guidance_scale=guidance_scale,
        guidance_rescale=guidance_rescale,
        ip_adapter_image=ip_image,
        generator=torch.Generator("cuda").manual_seed(42),
        output_type="latent",
    )

    if use_negative:
        base_args["negative_prompt"] = negative_prompt

    with torch.no_grad():
        latents = base(**base_args).images

    # -------------------------------------------------------
    # REFINER stage → final pixels
    # -------------------------------------------------------
    print("REFINER stage running (final image)...")

    refiner_args = dict(
        prompt=prompt,
        image=latents,
        height=height,
        width=width,
        num_inference_steps=n_steps,
        denoising_start=0.8,
        guidance_scale=guidance_scale,
        guidance_rescale=guidance_rescale,
        generator=torch.Generator("cuda").manual_seed(42),
        output_type="pil",
    )

    if use_negative:
        refiner_args["negative_prompt"] = negative_prompt

    with torch.no_grad():
        final_image = refiner(**refiner_args).images[0]

    # -------------------------------------------------------
    # Save results (same as your LoRA script)
    # -------------------------------------------------------
    os.makedirs(output_folder, exist_ok=True)

    neg_flag = "negON" if use_negative else "negOFF"

    filename = (
        f"sdxl_ipadapter_{prompt_label}_h{height}_w{width}_steps{n_steps}_"
        f"gs{guidance_scale}_gr{guidance_rescale}_{neg_flag}.png"
    )

    filepath = os.path.join(output_folder, filename)
    final_image.save(filepath)
    print(f"Saved: {filepath}")


# ======================================================================
# MAIN LOOP — identical structure to your reference script
# ======================================================================
if __name__ == '__main__':

    # Same parameters as your example
    STEPS = [50, 100]
    HEIGHT_WIDTH = [256, 512, 1024]
    GUIDANCE_SCALE = [3.0, 5.0, 7.0]
    GUIDANCE_RESCALE = [0.5, 0.7, 0.9]

    ROOT_FOLDER = "sdxl_ip_refiner_outputs"

    # Difficulty-level prompts
    prompts = {
        "easy": "A small cactus in a ceramic pot on a windowsill, gentle morning light",
        "medium": "A vintage lantern hanging in a forest at dusk, soft glowing fireflies around it",
        "difficult": "A massive ancient tree with a village built inside its branches, glowing runes, misty enchanted forest atmosphere"
    }

    negative_prompt = "deformed, ugly, low quality, bad anatomy, blur"

    # Loop: difficulty × negON/negOFF × parameter combinations
    for prompt_label, prompt_text in prompts.items():

        difficulty_folder = os.path.join(ROOT_FOLDER, prompt_label)

        for use_negative in [True, False]:

            for n_steps, size, gs, gr in itertools.product(
                STEPS, HEIGHT_WIDTH, GUIDANCE_SCALE, GUIDANCE_RESCALE
            ):
                main(
                    prompt_label,
                    prompt_text,
                    negative_prompt,
                    use_negative,
                    n_steps,
                    size,
                    size,
                    gs,
                    gr,
                    difficulty_folder,
                )