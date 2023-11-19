from typing import Optional
import torch
import os
from typing import List
import numpy as np
from PIL import Image
import cv2
import time
import sys

from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionControlNetInpaintPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from controlnet_aux import (
    HEDdetector,
    OpenposeDetector,
    MLSDdetector,
    CannyDetector,
    LineartDetector,
    MidasDetector
)
from controlnet_aux.util import ade_palette
# from midas_hack import MidasDetector
from consistencydecoder import ConsistencyDecoder, save_image
from compel import Compel
from transformers import pipeline

def resize_image(image, max_width, max_height):
    """
    Resize an image to a specific height while maintaining the aspect ratio and ensuring
    that neither width nor height exceed the specified maximum values.

    Args:
        image (PIL.Image.Image): The input image.
        max_width (int): The maximum allowable width for the resized image.
        max_height (int): The maximum allowable height for the resized image.

    Returns:
        PIL.Image.Image: The resized image.
    """
    # Get the original image dimensions
    original_width, original_height = image.size

    # Calculate the new dimensions to maintain the aspect ratio and not exceed the maximum values
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height

    # Choose the smallest ratio to ensure that neither width nor height exceeds the maximum
    resize_ratio = min(width_ratio, height_ratio)

    # Calculate the new width and height
    new_width = int(original_width * resize_ratio)
    new_height = int(original_height * resize_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image

def sort_dict_by_string(input_string, your_dict):
    if not input_string or not isinstance(input_string, str):
        # Return the original dictionary if the string is empty or not a string
        return your_dict

    order_list = [item.strip() for item in input_string.split(',')]

    # Include keys from the input string that are present in the dictionary
    valid_keys = [key for key in order_list if key in your_dict]

    # Include keys from the dictionary that are not in the input string
    remaining_keys = [key for key in your_dict if key not in valid_keys]

    sorted_dict = {key: your_dict[key] for key in valid_keys}
    sorted_dict.update({key: your_dict[key] for key in remaining_keys})

    return sorted_dict


AUX_IDS = {
    # "depth": "fusing/stable-diffusion-v1-5-controlnet-depth",
    # "scribble": "fusing/stable-diffusion-v1-5-controlnet-scribble",
    'lineart': "ControlNet-1-1-preview/control_v11p_sd15_lineart",
    'tile': "lllyasviel/control_v11f1e_sd15_tile",
    'brightness': "ioclab/control_v1p_sd15_brightness",
    "inpainting": "lllyasviel/control_v11p_sd15_inpaint",
}

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "KLMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
}


SD15_WEIGHTS = "weights"
CONTROLNET_CACHE = "controlnet-cache"
PROCESSORS_CACHE = "processors-cache"
MISSING_WEIGHTS = []

if not os.path.exists(CONTROLNET_CACHE) or not os.path.exists(PROCESSORS_CACHE):
    print(
        "controlnet weights missing, use `cog run python script/download_weights` to download"
    )
    MISSING_WEIGHTS.append("controlnet")

if not os.path.exists(SD15_WEIGHTS):
    print(
        "sd15 weights missing, use `cog run python` and then load and save_pretrained('weights')"
    )
    MISSING_WEIGHTS.append("sd15")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        if len(MISSING_WEIGHTS) > 0:
            print("skipping setup... missing weights: ", MISSING_WEIGHTS)
            return

        print("Loading pipeline...")
        st = time.time()

        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD15_WEIGHTS, torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

        self.controlnets = {}
        for name in AUX_IDS.keys():
            self.controlnets[name] = ControlNetModel.from_pretrained(
                os.path.join(CONTROLNET_CACHE, name),
                torch_dtype=torch.float16,
                local_files_only=True,
            ).to("cuda")
            
        self.tile_pipe= StableDiffusionControlNetPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=self.pipe.safety_checker,
                feature_extractor=self.pipe.feature_extractor,
                controlnet=self.controlnets['tile'],
            )

        self.canny = CannyDetector()

        # Depth + Normal
        self.midas = MidasDetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=PROCESSORS_CACHE
        )

        self.hed = HEDdetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=PROCESSORS_CACHE
        )

        # Hough
        self.mlsd = MLSDdetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=PROCESSORS_CACHE
        )

        self.seg_processor = AutoImageProcessor.from_pretrained(
            "openmmlab/upernet-convnext-small", cache_dir=PROCESSORS_CACHE
        )
        self.seg_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small", cache_dir=PROCESSORS_CACHE
        )

        self.pose = OpenposeDetector.from_pretrained(
            "lllyasviel/Annotators", cache_dir=PROCESSORS_CACHE
        )
        
        self.lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
        
        self.compel_proc = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder)
        
        # self.consistency_decoder = ConsistencyDecoder(
        #     device="cuda:0", download_root="/src/consistencydecoder-cache"
        # )
        
        self.depth_estimator = pipeline('depth-estimation')

        print("Setup complete in %f" % (time.time() - st))

    def depth_preprocess(self, img):
        image = self.depth_estimator(img)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    def scribble_preprocess(self, img):
        return self.hed(img, scribble=True)
    
    def lineart_preprocess(self, img):
        return self.lineart(img)

    def tile_preprocess(self, img):
        return img
    
    def brightness_preprocess(self, img):
        return img
    
    def make_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        # Convert the torch tensor back to a Pillow image
        # image_pil = Image.fromarray((image.squeeze().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

        return image

    def resize_for_condition_image(self, input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img

    def build_pipe(
        self, inputs, max_width, max_height, guess_mode=False
    ):
        control_nets = []
        processed_control_images = []
        conditioning_scales = []
        w, h = max_width, max_height
        inpainting = False
        #image and mask for inpainting
        mask= None
        init_image= None
        got_size= False
        for name, [image, conditioning_scale, mask_image] in inputs.items():
            if image is None:
                continue
            print(name)
            image = Image.open(image)
            if not got_size:
                image= resize_image(image, max_width, max_height)
                w, h= image.size
                got_size= True
            else:
                image= image.resize((w,h))

            if name=="inpainting" and mask_image:
                inpainting = True
                mask_image= Image.open(mask_image)
                mask= mask_image.resize((w,h))
                img= self.make_inpaint_condition(image, mask)
                init_image= image
            else:
                img = getattr(self, "{}_preprocess".format(name))(image)
            
            if not inpainting:
                img= img.resize((w,h))
            control_nets.append(self.controlnets[name])
            processed_control_images.append(img)
            conditioning_scales.append(conditioning_scale)

        if len(control_nets) == 0:
            pipe = self.pipe
            kwargs = {}
        else:
            if inpainting:
                pipe = StableDiffusionControlNetInpaintPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    tokenizer=self.pipe.tokenizer,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    safety_checker=self.pipe.safety_checker,
                    feature_extractor=self.pipe.feature_extractor,
                    controlnet=control_nets,
                )
                kwargs = {
                    "image": init_image,
                    "mask_image": mask,
                    "control_image": processed_control_images,
                    "controlnet_conditioning_scale": conditioning_scales,
                    "guess_mode": guess_mode,
                }
            else:
                pipe = StableDiffusionControlNetPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=self.pipe.safety_checker,
                feature_extractor=self.pipe.feature_extractor,
                controlnet=control_nets,
                )
                kwargs = {
                    "image": processed_control_images,
                    "controlnet_conditioning_scale": conditioning_scales,
                    "guess_mode": guess_mode,
                }

        return pipe, kwargs

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt - using compel, use +++ to increase words weight:: doc: https://github.com/damian0815/compel/tree/main/doc || https://invoke-ai.github.io/InvokeAI/features/PROMPTS/#attention-weighting",),
        lineart_image: Path = Input(
            description="Control image for canny controlnet", default=None
        ),
        lineart_conditioning_scale: float = Input(
            description="Conditioning scale for canny controlnet",
            default=1,
        ),
        # depth_image: Path = Input(
        #     description="Control image for depth controlnet", default=None
        # ),
        # depth_conditioning_scale: float = Input(
        #     description="Conditioning scale for depth controlnet",
        #     default=1,
        # ),
        # scribble_image: Path = Input(
        #     description="Control image for scribble controlnet", default=None
        # ),
        # scribble_conditioning_scale: float = Input(
        #     description="Conditioning scale for scribble controlnet",
        #     default=1,
        # ),
        tile_image: Path = Input(
            description="Control image for tile controlnet", default=None
        ),
        tile_conditioning_scale: float = Input(
            description="Conditioning scale for tile controlnet",
            default=1,
        ),
        brightness_image: Path = Input(
            description="Control image for brightness controlnet", default=None
        ),
        brightness_conditioning_scale: float = Input(
            description="Conditioning scale for brightness controlnet",
            default=1,
        ),
        inpainting_image: Path = Input(
            description="Control image for inpainting controlnet", default=None
        ),
        mask_image: Path = Input(
            description="mask image for inpainting controlnet", default=None
        ),
        inpainting_conditioning_scale: float = Input(
            description="Conditioning scale for brightness controlnet",
            default=1,
        ),
        num_outputs: int = Input(
            description="Number of images to generate",
            ge=1,
            le=10,
            default=1,
        ),
        max_width: int = Input(
            description="Max width/Resolution of image",
            default=512,
        ),
        max_height: int = Input(
            description="Max height/Resolution of image",
            default=512,
        ),
        # consistency_decoder: bool = Input(
        #     description="Enable consistency decoder",
        #     default=True,
        # ),
        scheduler: str = Input(
            default="DDIM",
            choices=SCHEDULERS.keys(),
            description="Choose a scheduler.",
        ),
        num_inference_steps: int = Input(description="Steps to run denoising", default=20),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.0,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(description="Seed", default=None),
        eta: float = Input(
            description="Controls the amount of noise that is added to the input data during the denoising diffusion process. Higher value -> more noise",
            default=0.0,
        ),
        negative_prompt: str = Input(  # FIXME
            description="Negative prompt - using compel, use +++ to increase words weight",
            default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        ),
        guess_mode: bool = Input(
            description="In this mode, the ControlNet encoder will try best to recognize the content of the input image even if you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.",
            default=False,
        ),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=False
        ),
        sorted_controlnets: str = Input(
            description="Comma seperated string of controlnet names, list of names: tile, inpainting, lineart,depth ,scribble , brightness /// example value: tile, inpainting, lineart ", default="tile, inpainting, lineart"
        ),
        low_res_fix: bool = Input(
            description="Using controlnet tile- after image generation", default=False
        ),
        low_res_fix_resolution: int = Input(
            description="controlnet tile resolution- after image generation", default=768
        ),
        low_res_fix_steps: int = Input(
            description="controlnet tile resolution- after image generation", default=10
        ),
        low_res_fix_prompt: str = Input(
            description="", default="best quality"
        ),
        low_res_fix_negative_prompt: str = Input(
            description="", default="blur, lowres, bad anatomy, bad hands, cropped, worst quality"
        ),
        low_res_fix_guess_mode: bool = Input(
            description="low res fix - guess mode", default=False
        ),
    ) -> List[Path]:
        if len(MISSING_WEIGHTS) > 0:
            raise Exception("missing weights")
        
        control_inputs= {
                "brightness": [brightness_image, brightness_conditioning_scale, None],
                "tile": [tile_image, tile_conditioning_scale, None],
                "lineart": [lineart_image, lineart_conditioning_scale, None],
                "inpainting": [inpainting_image, inpainting_conditioning_scale, mask_image],
                # "scribble": [scribble_image, scribble_conditioning_scale, None],
                # "depth": [depth_image, depth_conditioning_scale, None],
            }
        sorted_control_inputs= sort_dict_by_string(sorted_controlnets, control_inputs)

        pipe, kwargs = self.build_pipe(
            sorted_control_inputs,
            max_width=max_width,
            max_height=max_height,
            guess_mode=guess_mode,
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        if disable_safety_check:
            pipe.safety_checker = None

        output_paths= []
        for idx in range(num_outputs):
            this_seed = seed + idx
            generator = torch.Generator("cuda").manual_seed(this_seed)

            output = pipe(
                prompt_embeds=self.compel_proc(prompt),
                negative_prompt_embeds=self.compel_proc(negative_prompt),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                num_images_per_prompt=1,
                generator=generator,
                output_type="pil",
                **kwargs,
            )

            if output.nsfw_content_detected and output.nsfw_content_detected[0]:
                continue
            
            output_path = f"/tmp/seed-{this_seed}.png"
            output.images[0].save(output_path)
            output_paths.append(Path(output_path))
            print("size: ", output.images[0].size)
            if low_res_fix:
                print("Running low res fix...")
                start = time.time()
                seed_= int.from_bytes(os.urandom(2), "big")
                condition_image = self.resize_for_condition_image(output.images[0], low_res_fix_resolution)
                print("condition image resize took", time.time() - start, "seconds")
                tile_output= self.tile_pipe(
                    prompt_embeds=self.compel_proc(low_res_fix_prompt),
                    negative_prompt_embeds=self.compel_proc(low_res_fix_negative_prompt),
                    image= condition_image,
                    num_inference_steps= low_res_fix_steps,
                    width=condition_image.size[0],
                    height=condition_image.size[1],
                    controlnet_conditioning_scale=2.0,
                    generator=generator,
                    guess_mode= low_res_fix_guess_mode,
                )

                path = f"/tmp/low-res-{seed_}.png"
                print("low-res-fix took", time.time() - start, "seconds")
                tile_output.images[0].save(path)
                output_paths.append(Path(path))
            # else:
                # if consistency_decoder:
                #     print("Running consistency decoder...")
                #     start = time.time()
                #     sample = self.consistency_decoder(
                #         output.images[0].unsqueeze(0) / self.pipe.vae.config.scaling_factor
                #     )
                #     print("Consistency decoder took", time.time() - start, "seconds")
                #     save_image(sample, output_path)
                # else:
                

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
