import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
import json
import numpy as np
import streamlit as st
import tensorflow as tf
import base64, io
from PIL import Image

from gradcam import make_gradcam_heatmap, overlay_heatmap

MODEL_PATH = "model.h5"
LABELS_PATH = "labels.json"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="wide")


@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except TypeError as e:
        if "optional" not in str(e):
            raise
        class FixedInputLayer(tf.keras.layers.InputLayer):
            def __init__(self, *args, **kwargs):
                kwargs.pop("optional", None)
                super().__init__(*args, **kwargs)
        model = tf.keras.models.load_model(
            MODEL_PATH, compile=False, custom_objects={"InputLayer": FixedInputLayer})
    with open(LABELS_PATH) as f:
        classes = json.load(f)
    return model, classes


def pretty(name: str):
    parts = name.split("___")
    crop = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else name
    return crop, disease

# import base64, io

def show_image(col, img, caption):
    small = img.copy()
    small.thumbnail((700, 700))            # keeps the page light
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    col.markdown(
        f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px">'
        f'<p style="text-align:center;opacity:0.7;font-size:0.9rem">{caption}</p>',
        unsafe_allow_html=True,
    )
st.title("🌿 Crop Disease Detection with Explainable AI")
st.caption("MobileNetV2 + Grad-CAM: see which part of the leaf drove the prediction.")

model, classes = load_assets()

with st.sidebar:
    st.header("Settings")
    alpha = st.slider("Heatmap intensity", 0.1, 0.9, 0.45, 0.05)
    explain = st.selectbox("Explain which prediction?", ["Top-1 (predicted)", "Top-2", "Top-3"])
    st.markdown("Use a clear photo of a single leaf for best results.")

file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
cam = st.camera_input("...or take a photo") if file is None else None
source = file or cam

if source:
    img = Image.open(source).convert("RGB")
    # Model has its own Rescaling layer -> feed RAW 0-255 pixels (no preprocess_input!)
    batch = np.expand_dims(np.array(img.resize(IMG_SIZE), dtype=np.float32), 0)

    probs = model.predict(batch, verbose=0)[0]
    top3 = probs.argsort()[::-1][:3]
    target = int(top3[int(explain.split("-")[1][0]) - 1])

    heatmap, _ = make_gradcam_heatmap(model, batch, class_index=target)
    overlay, colored = overlay_heatmap(img, heatmap, alpha=alpha)

    crop, disease = pretty(classes[top3[0]])
    conf = float(probs[top3[0]])

    c1, c2, c3 = st.columns(3)
    show_image(c1, img, "Original")
    show_image(c2, colored, "Grad-CAM heatmap")
    show_image(c3, overlay, "Overlay: red = most influential region")

    st.subheader("Result")
    if conf < CONFIDENCE_THRESHOLD:
        st.warning(f"Low confidence ({conf:.1%}). The image may not be a supported leaf, "
                   "or the photo quality is poor. Please retake it.")
    if "healthy" in disease.lower():
        st.success(f"**{crop}** - Healthy ({conf:.1%})")
    else:
        st.error(f"**{crop}** - {disease} ({conf:.1%})")

    st.markdown("**Top 3 predictions**")
    for i in top3:
        c, d = pretty(classes[i])
        st.progress(float(probs[i]), text=f"{c} - {d}: {probs[i]:.1%}")
