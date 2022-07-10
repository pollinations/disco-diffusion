# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from email.policy import default

from cog import BasePredictor, Input, Path
from pydotted import pydot
from typing import List
import os,sys,tempfile,glob


PROJECT_DIR = os.path.abspath(os.getcwd())

sys.path.append(PROJECT_DIR)
import dd, dd_args

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Predictor(BasePredictor):
    def setup(self):
        sys.argv = [sys.argv[0]]
        self.pargs = dd_args.arg_configuration_loader(pydot({"model_path": "/root/.cache/disco-diffusion"}))
        self.folders = dd.setupFolders(is_colab=False, PROJECT_DIR=PROJECT_DIR, pargs=self.pargs)
        dd.loadModels(self.folders)
        self.device = dd.getDevice(self.pargs)

    def predict(
        self,
        steps: int = Input(description="Number of steps", default=100),
        prompt: str = Input(description="Text Prompt", default="A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."),
        ViTB32: bool = Input(description="Use ViTB32 model", default=True),
        ViTB16: bool = Input(description="Use ViTB16 model", default=True),
        ViTL14: bool = Input(description="Use ViTB14 model", default=False),
        ViTL14_336: bool = Input(description="Use ViTL14_336 model", default=False),
        RN50: bool = Input(description="Use RN50 model", default=True),
        RN50x4: bool = Input(description="Use RN50x4 model", default=False),
        RN50x16: bool = Input(description="Use RN50x16 model", default=False),
        RN50x64: bool = Input(description="Use RN50x64 model", default=False),
        RN50x101: bool = Input(description="Use RN50x101 model", default=False),
        diffusion_model: str = Input(description="Diffusion Model", default = "512x512_diffusion_uncond_finetune_008100", choices=[
            "512x512_diffusion_uncond_finetune_008100",
            "256x256_diffusion_uncond",
            "pixel_art_diffusion_hard_256",
            "pixel_art_diffusion_soft_256",
            "256x256_openai_comics_faces_by_alex_spirin_084000",
            "lsun_uncond_100M_1200K_bs128",
            # "vit_b_16_plus_240-laion400m_e31-8fb26589",
        ]),
        use_secondary_model: bool = Input(description="Use secondary model", default=True),
        diffusion_sampling_mode: str = Input(description="Diffusion Sampling Mode", default="ddim", choices=["plms", "ddim"]),
        width: int = Input(description="Width", default=1280),
        height: int = Input(description="Height", default=768),
        clip_guidance_scale: int = Input(description="CLIP Guidance Scale", default=5000),
        tv_scale: int = Input(description="TV Scale", default=0),
        range_scale: int = Input(description="Range Scale", default=150),
        sat_scale: int = Input(description="Saturation Scale", default=0),
        cutn_batches: int = Input(description="Cut Batches", default=4),
        skip_augs: bool = Input(description="Skip Augmentations", default=False),
        init_image: Path = Input(description="Init Image", default=None),
        target_image: Path = Input(description="Target Image", default=None),
        init_scale: int = Input(description="Init Scale", default=1000),
        target_scale: int = Input(description="Target Scale", default=20000),
        skip_steps: int = Input(description="Skip Steps", default=10),
        seed: int = Input(description="Seed (leave empty to use a random seed)", default=None, le=(2**32-1), ge=0),
    ) -> List[Path]:
        """Run a single prediction on the model"""        
        outdir = tempfile.mkdtemp('disco')
        self.pargs.images_out = outdir
        self.pargs.steps = steps
        self.pargs.text_prompts= { 0: [ prompt ] }
        self.pargs.ViTB32=ViTB32
        self.pargs.ViTB16=ViTB16
        self.pargs.ViTL14=ViTL14
        self.pargs.ViTL14_336=ViTL14_336
        self.pargs.RN50=RN50
        self.pargs.RN50x4=RN50x4
        self.pargs.RN50x16=RN50x16
        self.pargs.RN50x64=RN50x64
        self.pargs.RN50x101=RN50x101
        self.pargs.diffusion_model = diffusion_model
        self.pargs.use_secondary_model = use_secondary_model
        self.pargs.diffusion_sampling_mode = diffusion_sampling_mode
        self.pargs.width_height = [width, height]
        self.pargs.clip_guidance_scale = clip_guidance_scale
        self.pargs.tv_scale = tv_scale
        self.pargs.range_scale = range_scale
        self.pargs.sat_scale = sat_scale
        self.pargs.cutn_batches = cutn_batches
        self.pargs.skip_augs = skip_augs
        self.pargs.init_image = init_image
        self.pargs.target_image = target_image
        self.pargs.init_scale = init_scale
        self.pargs.target_scale = target_scale
        self.pargs.skip_steps = skip_steps
        if seed:
            self.pargs.set_seed = seed

        dd.start_run(pargs=self.pargs, folders=self.folders, device=self.device, is_colab=False)
        yield [Path(image) for image in glob.glob(outdir+"/*.png")]
