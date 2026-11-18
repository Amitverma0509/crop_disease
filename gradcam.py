import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import colormaps

# Layer names inside YOUR trained model (from model.h5):
# input -> sequential(augmentation) -> rescaling -> mobilenetv2_1.00_224 -> GAP -> dropout -> dense
RESCALE = "rescaling"
BACKBONE = "mobilenetv2_1.00_224"   # nested model; its output is "out_relu" (7x7x1280)
GAP = "global_average_pooling2d"
DENSE = "dense"


def make_gradcam_heatmap(model, img_batch, class_index=None):
    """img_batch: float32 array (1, 224, 224, 3) with RAW pixel values 0-255
    (the model rescales to [-1, 1] by itself). Returns (heatmap in [0,1], probs)."""
    rescale = model.get_layer(RESCALE)
    backbone = model.get_layer(BACKBONE)
    gap = model.get_layer(GAP)
    dense = model.get_layer(DENSE)

    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        x = rescale(x)
        conv_out = backbone(x, training=False)       # last conv feature maps (1, 7, 7, 1280)
        tape.watch(conv_out)
        probs = dense(gap(conv_out))                 # dropout is identity at inference
        if class_index is None:
            class_index = int(tf.argmax(probs[0]))
        score = probs[:, class_index]

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))          # importance of each channel
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)  # weighted sum of feature maps
    heatmap = tf.nn.relu(heatmap)                            # keep positive evidence only
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), probs[0].numpy()


def overlay_heatmap(original_img: Image.Image, heatmap, alpha=0.45):
    """Resize heatmap to the original image, colour it, blend. Returns (overlay, colored_heatmap)."""
    w, h = original_img.size
    hm = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BICUBIC)
    hm = np.array(hm) / 255.0
    colored = (colormaps["jet"](hm)[..., :3] * 255).astype(np.uint8)
    base = np.array(original_img.convert("RGB"))
    overlay = (base * (1 - alpha) + colored * alpha).astype(np.uint8)
    return Image.fromarray(overlay), Image.fromarray(colored)
