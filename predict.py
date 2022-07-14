# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from email.policy import default

from cog import BasePredictor, Input, Path
from pydotted import pydot
from typing import List, Iterator
import os,sys,tempfile,glob
import queue, threading, uuid

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
        steps: int = Input(description="Number of steps, higher numbers will give more refined output but will take longer", default=100),
        prompt: str = Input(description="Text Prompt", default="A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."),
        width: int = Input(description="Width of the output image, higher numbers will take longer", default=1280),
        height: int = Input(description="Height of the output image, higher numbers will take longer", default=768),
        diffusion_model: str = Input(description="Diffusion Model", default = "512x512_diffusion_uncond_finetune_008100", choices=[
            "512x512_diffusion_uncond_finetune_008100",
            "256x256_diffusion_uncond",
            "pixel_art_diffusion_hard_256",
            "pixel_art_diffusion_soft_256",
            "pixelartdiffusion_expanded",
            "pixelartdiffusion4k",
            "PADexpanded",
            "watercolordiffusion",
            "watercolordiffusion_2",
            "PulpSciFiDiffusion",
            "256x256_openai_comics_faces_by_alex_spirin_084000",
            "lsun_uncond_100M_1200K_bs128",
            "ukiyoe_diffusion_256_022000.pt",
            "liminal_diffusion",
        ]),
        diffusion_sampling_mode: str = Input(description="Diffusion Sampling Mode", default="ddim", choices=["plms", "ddim"]),
        ViTB32: bool = Input(description="Use ViTB32 model", default=True),
        ViTB16: bool = Input(description="Use ViTB16 model", default=True),
        ViTL14: bool = Input(description="Use ViTB14 model", default=False),
        ViTL14_336: bool = Input(description="Use ViTL14_336 model", default=False),
        RN50: bool = Input(description="Use RN50 model", default=True),
        RN50x4: bool = Input(description="Use RN50x4 model", default=False),
        RN50x16: bool = Input(description="Use RN50x16 model", default=False),
        RN50x64: bool = Input(description="Use RN50x64 model", default=False),
        RN50x101: bool = Input(description="Use RN50x101 model", default=False),
        RN101: bool = Input(description="Use RN101 model", default=False),
        ViTB32_laion2b_e16: bool = Input(description="Use ViTB32_laion2b_e16 model", default=False),
        ViTB32_laion400m_e31: bool = Input(description="Use ViTB32_laion400m_e31 model", default=False),
        ViTB32_laion400m_e32: bool = Input(description="Use ViTB32_laion400m_e32 model", default=False),
        ViTB32quickgelu_laion400m_e31: bool = Input(description="Use ViTB32quickgelu_laion400m_e31 model", default=False),
        ViTB32quickgelu_laion400m_e32: bool = Input(description="Use ViTB32quickgelu_laion400m_e32 model", default=False),
        ViTB16_laion400m_e31:bool = Input(description="Use ViTB16_laion400m_e31 model", default=False),
        ViTB16_laion400m_e32:bool = Input(description="Use ViTB16_laion400m_e32 model", default=False),
        RN50_yffcc15m:bool = Input(description="Use RN50_yffcc15m model", default=False),
        RN50_cc12m:bool = Input(description="Use RN50_cc12m model", default=False),
        RN50_quickgelu_yfcc15m:bool = Input(description="Use RN50_quickgelu_yfcc15m model", default=False),
        RN50_quickgelu_cc12m:bool = Input(description="Use RN50_quickgelu_cc12m model", default=False),
        RN101_yfcc15m:bool = Input(description="Use RN101_yfcc15m model", default=False),
        RN101_quickgelu_yfcc15m:bool = Input(description="Use RN101_quickgelu_yfcc15m model", default=False),
        use_secondary_model: bool = Input(description="Use secondary model", default=True),        
        clip_guidance_scale: int = Input(description="CLIP Guidance Scale", default=5000),
        tv_scale: int = Input(description="TV Scale", default=0),
        range_scale: int = Input(description="Range Scale", default=150),
        sat_scale: int = Input(description="Saturation Scale", default=0),
        cutn_batches: int = Input(description="Cut Batches", default=4),
        skip_augs: bool = Input(description="Skip Augmentations", default=False),
        init_image: Path = Input(description="Initial image to start generation from", default=None),
        target_image: Path = Input(description="Target image to generate towards, similarly to the text prompt", default=None),
        init_scale: int = Input(description="Initial Scale", default=1000),
        target_scale: int = Input(description="Target Scale", default=20000),
        skip_steps: int = Input(description="Skip Steps", default=10),        
        display_rate: int = Input(description="Steps between outputs, lower numbers may slow down generation.", default=20),
        seed: int = Input(description="Seed (leave empty to use a random seed)", default=None, le=(2**32-1), ge=0),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""                
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
        self.pargs.RN101=RN101
        self.pargs.ViTB32_laion2b_e16=ViTB32_laion2b_e16
        self.pargs.ViTB32_laion400m_e31=ViTB32_laion400m_e31
        self.pargs.ViTB32_laion400m_e32=ViTB32_laion400m_e32
        self.pargs.ViTB32quickgelu_laion400m_e31=ViTB32quickgelu_laion400m_e31
        self.pargs.ViTB32quickgelu_laion400m_e32=ViTB32quickgelu_laion400m_e32
        self.pargs.ViTB16_laion400m_e31=ViTB16_laion400m_e31
        self.pargs.ViTB16_laion400m_e32=ViTB16_laion400m_e32
        self.pargs.RN50_yffcc15m=RN50_yffcc15m
        self.pargs.RN50_cc12m=RN50_cc12m
        self.pargs.RN50_quickgelu_yfcc15m=RN50_quickgelu_yfcc15m
        self.pargs.RN50_quickgelu_cc12m=RN50_quickgelu_cc12m
        self.pargs.RN101_yfcc15m=RN101_yfcc15m
        self.pargs.RN101_quickgelu_yfcc15m=RN101_quickgelu_yfcc15m
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
        self.pargs.init_image = str(init_image)
        self.pargs.target_image = str(target_image)
        self.pargs.init_scale = init_scale
        self.pargs.target_scale = target_scale
        self.pargs.skip_steps = skip_steps
        self.pargs.display_rate = display_rate
        if (seed):
            self.pargs.set_seed = seed

        id = str(uuid.uuid4())
        self.pargs.uuid = id                          
        self.pargs.n_batches = 1
        self.pargs.images_out = "images_out"
        self.pargs.init_images = "init_images"        
        self.pargs.batch_name = id
        self.folders = dd.setupFolders(is_colab=False, PROJECT_DIR=PROJECT_DIR, pargs=self.pargs)
        self.pargs.batchFolder = self.folders.batch_folder
        self.pargs.batchNum = 0
        dd.progress_fn = lambda img: output.put(img)
        self.device = dd.getDevice(self.pargs)
        output = queue.SimpleQueue()
        t = threading.Thread(target=self.worker, daemon=True)
        t.start()
        while t.is_alive():
            try:
                image = output.get(block=True, timeout=5)
                yield Path(image)
            except: {}

        yield Path(glob.glob(f'{self.folders.batch_folder}/*.png')[0])
        
    def worker(self):
        dd.start_run(pargs=self.pargs, folders=self.folders, device=self.device)
