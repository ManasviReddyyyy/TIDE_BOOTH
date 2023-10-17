from __future__ import annotations

import math
import random
import gradio as gr
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline
# import cv2

example_instructions = [
    "Make it a picasso painting",
    "as if it were by modigliani",
    "convert to a bronze statue",
    "Turn it into an anime.",
    "have it look like a graphic novel",
    "make him gain weight",
    "what would he look like bald?",
    "Have him smile",
    "Put him in a cocktail party.",
    "move him at the beach.",
    "add dramatic lighting",
    "Convert to black and white",
    "What if it were snowing?",
    "Give him a leather jacket",
    "Turn him into a cyborg!",
    "make him wear a beanie",
]

model_id = "timbrooks/instruct-pix2pix"
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", safety_checker=None) if torch.cuda.is_available() else StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
    pipe = pipe.to(device)

    # def load_example(
    #     steps: int,
    #     randomize_seed: bool,
    #     seed: int,
    #     randomize_cfg: bool,
    #     text_cfg_scale: float,
    #     image_cfg_scale: float,
    # ):
    #     example_instruction = random.choice(example_instructions)
    #     return [example_instruction] + generate(
    #         None,  # Pass None for input_image when loading example
    #         example_instruction,
    #         steps,
    #         randomize_seed,
    #         seed,
    #         randomize_cfg,
    #         text_cfg_scale,
    #         image_cfg_scale,
    #     )

    def generate(
        input_image: Image.Image,
        instruction: str,
        steps: int,
        randomize_seed: bool,
        seed: int,
        randomize_cfg: bool,
        text_cfg_scale: float,
        image_cfg_scale: float,
    ):
        # if input_image is None:
        #     # Capture an image from the webcam using OpenCV
        #     cap = cv2.VideoCapture(0)
        #     ret, frame = cap.read()
        #     cap.release()
        #     if not ret:
        #         return ["Webcam Error"]
        #     input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        seed = random.randint(0, 100000) if randomize_seed else seed
        text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
        image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale

        width, height = input_image.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        if instruction == "":
            return [input_image, seed]

        generator = torch.manual_seed(seed)
        edited_image = pipe(
            instruction, image=input_image,
            guidance_scale=text_cfg_scale, image_guidance_scale=image_cfg_scale,
            num_inference_steps=steps, generator=generator,
        ).images[0]
        return [seed, text_cfg_scale, image_cfg_scale, edited_image]

    # def reset():
    #     return [0, "Randomize Seed", 1371, "Fix CFG", 7.5, 1.5, None]

    with gr.Blocks() as demo:
        # with gr.Row():
        #     with gr.Column(scale=1):
        #         gr.HTML("""<div >
        #         <img src='https://th.bing.com/th/id/OIP.gU-LyqDD4sApdSO1OvZWowHaHa?pid=ImgDet&rs=1' alt='image One' width="75" height="75">
        #         </div>""")
        #     with gr.Column(scale=3):
        #         gr.HTML("""
        #         <h1 style="font-weight: 900; margin-bottom: 7px;">
        #             GenNextSelfie: Click a Selfie and Follow Image Editing Instructions
        #          </h1>""")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Image(label="TIDE", value="TIDE_logo.png", type="filepath", show_share_button=False, show_download_button=False)
            with gr.Column(scale=3):
                gr.Textbox(label="GenNextSelfie",value="Click a Selfie and Follow Image Editing Instructions", interactive=False)
                # gr.HTML("""
                # <h1 style="font-weight: 900; margin-bottom: 7px;">
                #     GenNextSelfie: Click a Selfie and Follow Image Editing Instructions
                #  </h1>""")

        with gr.Row():
            steps = gr.Number(value=25, precision=0, label="Steps", interactive=True)
            randomize_seed = gr.Radio(
                ["Fix Seed", "Randomize Seed"],
                value="Randomize Seed",
                type="index",
                show_label=True,
                interactive=False,
                visible=False
            )
            seed = gr.Number(value=1371, precision=0, label="Seed", interactive=False, visible=False)
            randomize_cfg = gr.Radio(
                ["Fix CFG", "Randomize CFG"],
                value="Randomize CFG",
                type="index",
                show_label=True,
                interactive=False,
                visible=False
            )
            text_cfg_scale = gr.Number(value=7.5, label=f"Text CFG", interactive=False, visible=False)
            image_cfg_scale = gr.Number(value=1.5, label=f"Image CFG", interactive=False, visible=False)

        # with gr.Row():
        gr.HTML("""
            <h1 style="font-weight: 900; margin-bottom: 7px;">
                Change Background
             </h1>""")
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                instruction = gr.Textbox(lines=1, label="Edit Instruction", interactive=True)
            with gr.Column(scale=1, min_width=100):
                change_button = gr.Button("Change")
            
        with gr.Row():
            input_image = gr.Webcam(label="Capture Image", type="pil", interactive=True, height=512, width=512)
            edited_image = gr.Image(label=f"Edited Image", type="pil", interactive=False, height=512, width=512)
            # input_image.style(height=512, width=512)
            # edited_image.style(height=512, width=512)

        # with gr.Row():
        #     steps = gr.Number(value=50, precision=0, label="Steps", interactive=True)
        #     randomize_seed = gr.Radio(
        #         ["Fix Seed", "Randomize Seed"],
        #         value="Randomize Seed",
        #         type="index",
        #         show_label=True,
        #         interactive=False,
        #     )
        #     seed = gr.Number(value=1371, precision=0, label="Seed", interactive=False)
        #     randomize_cfg = gr.Radio(
        #         ["Fix CFG", "Randomize CFG"],
        #         value="Randomize CFG",
        #         type="index",
        #         show_label=True,
        #         interactive=False,
        #     )
        #     text_cfg_scale = gr.Number(value=7.5, label=f"Text CFG", interactive=False)
        #     image_cfg_scale = gr.Number(value=1.5, label=f"Image CFG", interactive=False)
  
        change_button.click(
            fn=generate,
            inputs=[
                input_image,
                instruction,
                steps,
                randomize_seed,
                seed,
                randomize_cfg,
                text_cfg_scale,
                image_cfg_scale,
            ],
            outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image],
        )

        # with gr.Row():
        gr.HTML("""
            <h1 style="font-weight: 900; margin-bottom: 7px;">
                Morph into an Avatar
             </h1>""")
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                instruction = gr.Textbox(lines=1, label="Edit Instruction", interactive=True)
            with gr.Column(scale=1, min_width=100):
                morph_button = gr.Button("Morph")
            
        with gr.Row():
            input_image = gr.Webcam(label="Capture Image", type="pil", interactive=True, height=512, width=512)
            edited_image = gr.Image(label=f"Edited Image", type="pil", interactive=False, height=512, width=512)
            # input_image.style(height=512, width=512)
            # edited_image.style(height=512, width=512)

        # with gr.Row():
        #     steps = gr.Number(value=50, precision=0, label="Steps", interactive=True)
        #     randomize_seed = gr.Radio(
        #         ["Fix Seed", "Randomize Seed"],
        #         value="Randomize Seed",
        #         type="index",
        #         show_label=True,
        #         interactive=False,
        #     )
        #     seed = gr.Number(value=1371, precision=0, label="Seed", interactive=False)
        #     randomize_cfg = gr.Radio(
        #         ["Fix CFG", "Randomize CFG"],
        #         value="Randomize CFG",
        #         type="index",
        #         show_label=True,
        #         interactive=False,
        #     )
        #     text_cfg_scale = gr.Number(value=7.5, label=f"Text CFG", interactive=False)
        #     image_cfg_scale = gr.Number(value=1.5, label=f"Image CFG", interactive=False)
  
        morph_button.click(
            fn=generate,
            inputs=[
                input_image,
                instruction,
                steps,
                randomize_seed,
                seed,
                randomize_cfg,
                text_cfg_scale,
                image_cfg_scale,
            ],
            outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image],
        )

        # with gr.Row():
        gr.HTML("""
            <h1 style="font-weight: 900; margin-bottom: 7px;">
                Add a character
             </h1>""")
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                instruction = gr.Textbox(lines=1, label="Edit Instruction", interactive=True)
            with gr.Column(scale=1, min_width=100):
                add_button = gr.Button("Add")
            
        with gr.Row():
            input_image = gr.Webcam(label="Capture Image", type="pil", interactive=True, height=512, width=512)
            edited_image = gr.Image(label=f"Edited Image", type="pil", interactive=False, height=512, width=512)
            # input_image.style(height=512, width=512)
            # edited_image.style(height=512, width=512)
 
        # with gr.Row():
        #     steps = gr.Number(value=50, precision=0, label="Steps", interactive=True)
        #     randomize_seed = gr.Radio(
        #         ["Fix Seed", "Randomize Seed"],
        #         value="Randomize Seed",
        #         type="index",
        #         show_label=True,
        #         interactive=False,
        #     )
        #     seed = gr.Number(value=1371, precision=0, label="Seed", interactive=False)
        #     randomize_cfg = gr.Radio(
        #         ["Fix CFG", "Randomize CFG"],
        #         value="Randomize CFG",
        #         type="index",
        #         show_label=True,
        #         interactive=False,
        #     )
        #     text_cfg_scale = gr.Number(value=7.5, label=f"Text CFG", interactive=False)
        #     image_cfg_scale = gr.Number(value=1.5, label=f"Image CFG", interactive=False)
  
        add_button.click(
            fn=generate,
            inputs=[
                input_image,
                instruction,
                steps,
                randomize_seed,
                seed,
                randomize_cfg,
                text_cfg_scale,
                image_cfg_scale,
            ],
            outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image],
        )

    demo.queue(concurrency_count=1)
    demo.launch(share=False)


if __name__ == "__main__":
    main()