!pip install gradio -q
import gradio as gr, tensorflow as tf, numpy as np, json
from PIL import Image

drive_path = '/content/drive/MyDrive'
chest_model = tf.keras.models.load_model(f'{drive_path}/chest_model.keras')
brain_model  = tf.keras.models.load_model(f'{drive_path}/brain_model.keras')
skin_model   = tf.keras.models.load_model(f'{drive_path}/skin_model.keras')
with open(f'{drive_path}/skin_classes.json') as f:
    skin_labels = {v:k for k,v in json.load(f).items()}

SKIN_NAMES = {
    'akiec':'Actinic Keratosis', 'bcc':'Basal Cell Carcinoma',
    'bkl':'Benign Keratosis',    'df':'Dermatofibroma',
    'mel':'Melanoma',            'nv':'Normal (Melanocytic Nevi)',
    'vasc':'Vascular Lesion'
}
print("✅ Models loaded!")

def predict(img, scan_type):
    img = img.convert('RGB').resize((128,128))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)

    if scan_type == 'Chest X-Ray':
        pred  = chest_model.predict(arr, verbose=0)[0][0]
        label = 'Pneumonia Detected' if pred > 0.5 else 'Normal — No Disease'
        conf  = pred if pred > 0.5 else 1 - pred

    elif scan_type == 'Brain MRI':
        pred  = brain_model.predict(arr, verbose=0)[0][0]
        label = 'Tumor Detected' if pred > 0.5 else 'Normal — No Tumor'
        conf  = pred if pred > 0.5 else 1 - pred

    else:
        pred  = skin_model.predict(arr, verbose=0)[0]
        key   = skin_labels[int(pred.argmax())]
        label = SKIN_NAMES.get(key, key)
        conf  = float(pred.max())

    return {
        "Diagnosis":  label,
        "Confidence": f"{round(float(conf)*100, 1)}%",
        "Scan Type":  scan_type
    }

gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type='pil', label="Upload medical image"),
        gr.Radio(['Chest X-Ray','Brain MRI','Skin Lesion'],
                 label="Scan type", value='Chest X-Ray')
    ],
    outputs=gr.JSON(label="Result"),
    title="MediScan AI",
    description="Upload a chest X-ray, brain MRI, or skin image for AI diagnosis.",
    theme=gr.themes.Soft()
).launch(share=True)