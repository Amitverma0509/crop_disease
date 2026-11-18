# Crop Disease Detection - MobileNetV2 + Grad-CAM (38 PlantVillage classes)

## Run locally
Use Python 3.10 or 3.11 (TensorFlow 2.15 does not support 3.12+).
```
python -m venv venv
venv\Scripts\activate        # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Files
- model.h5        - trained Keras model (used by the app; Grad-CAM needs gradients)
- model.tflite    - lightweight copy for a future mobile app (cannot do Grad-CAM)
- labels.json     - 38 class names, same order as training
- gradcam.py      - Grad-CAM heatmap + overlay
- app.py          - Streamlit UI

## Model facts (useful for viva)
Input 224x224x3 raw pixels -> augmentation (inactive at inference) -> Rescaling to [-1,1]
-> MobileNetV2 (fine-tuned) -> GlobalAveragePooling -> Dropout 0.3 -> Dense(38, softmax).
Grad-CAM uses the final conv activation of MobileNetV2 (out_relu, 7x7x1280).

## Deploy (free)
Push to GitHub (model.h5 is ~22 MB, fine) and deploy on Streamlit Community Cloud
or Hugging Face Spaces. Set Python 3.11 in the platform settings.
