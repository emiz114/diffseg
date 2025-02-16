import tensorflow as tf
from PIL import Image
import nltk
from transformers import AutoProcessor, TFBlipForConditionalGeneration
from keras_cv.src.models.stable_diffusion.image_encoder import ImageEncoder
from third_party.keras_cv.stable_diffusion import StableDiffusion 
from third_party.keras_cv.diffusion_model import SpatialTransformer
from diffseg.utils import process_image, augmenter, vis_without_label, semantic_mask
from diffseg.segmentor import DiffSeg

is_noun = lambda pos: pos[:2] == 'NN'
!nvidia-smi
nltk.download('all')