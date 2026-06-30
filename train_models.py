import os, json, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (128, 128)
BATCH    = 64

def build_model(num_classes):
    base = MobileNetV2(input_shape=(128,128,3),
                       include_top=False, weights='imagenet')
    base.trainable = False
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    loss       = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'
    inp = tf.keras.Input(shape=(128,128,3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation=activation)(x)
    m   = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=loss, metrics=['accuracy'])
    return m

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=3,
    restore_best_weights=True, verbose=1
)

# ── CHEST ──────────────────────────────────────────
print("Training CHEST...")
cg = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                         horizontal_flip=True, zoom_range=0.1)
ct = cg.flow_from_directory('/content/chest/chest_xray/train',
     target_size=IMG_SIZE, batch_size=BATCH,
     class_mode='binary', subset='training')
cv = cg.flow_from_directory('/content/chest/chest_xray/train',
     target_size=IMG_SIZE, batch_size=BATCH,
     class_mode='binary', subset='validation')
cw = compute_class_weight('balanced', classes=np.array([0,1]), y=ct.classes)
print("Classes:", ct.class_indices, "Weights:", cw)

chest_model = build_model(1)
chest_model.fit(ct, validation_data=cv, epochs=7,
                class_weight={0:cw[0], 1:cw[1]},
                callbacks=[early_stop])
chest_model.save('/content/drive/MyDrive/chest_model.keras')
print("✅ Chest saved!")

# ── BRAIN ──────────────────────────────────────────
print("\nTraining BRAIN...")
bg = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                         horizontal_flip=True, zoom_range=0.1)
bt = bg.flow_from_directory('/content/brain',
     classes=['no','yes'],
     target_size=IMG_SIZE, batch_size=BATCH,
     class_mode='binary', subset='training')
bv = bg.flow_from_directory('/content/brain',
     classes=['no','yes'],
     target_size=IMG_SIZE, batch_size=BATCH,
     class_mode='binary', subset='validation')
bw = compute_class_weight('balanced', classes=np.array([0,1]), y=bt.classes)
print("Classes:", bt.class_indices, "Weights:", bw)

brain_model = build_model(1)
brain_model.fit(bt, validation_data=bv, epochs=7,
                class_weight={0:bw[0], 1:bw[1]},
                callbacks=[early_stop])
brain_model.save('/content/drive/MyDrive/brain_model.keras')
print("✅ Brain saved!")

# ── SKIN ───────────────────────────────────────────
print("\nTraining SKIN...")
skin_df = pd.read_csv('/content/skin/HAM10000_metadata.csv')
skin_df['image_path'] = skin_df['image_id'].apply(
    lambda x: f'/content/skin/ham10000_images_part_1/{x}.jpg'
    if os.path.exists(f'/content/skin/ham10000_images_part_1/{x}.jpg')
    else f'/content/skin/ham10000_images_part_2/{x}.jpg')
skin_df['label'] = skin_df['dx']
skin_df = skin_df[skin_df['image_path'].apply(os.path.exists)].reset_index(drop=True)
skin_df = skin_df.groupby('label').apply(
    lambda x: x.sample(min(len(x), 300), random_state=42)
).reset_index(drop=True)
print(f"Skin size: {len(skin_df)}\n", skin_df['label'].value_counts())

sg = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                         horizontal_flip=True, zoom_range=0.1)
st = sg.flow_from_dataframe(skin_df, x_col='image_path', y_col='label',
     target_size=IMG_SIZE, batch_size=BATCH,
     class_mode='categorical', subset='training')
sv = sg.flow_from_dataframe(skin_df, x_col='image_path', y_col='label',
     target_size=IMG_SIZE, batch_size=BATCH,
     class_mode='categorical', subset='validation')
sw  = compute_class_weight('balanced',
      classes=np.unique(st.classes), y=st.classes)

skin_model = build_model(len(st.class_indices))
skin_model.fit(st, validation_data=sv, epochs=7,
               class_weight=dict(enumerate(sw)),
               callbacks=[early_stop])
skin_model.save('/content/drive/MyDrive/skin_model.keras')

with open('/content/drive/MyDrive/skin_classes.json', 'w') as f:
    json.dump(st.class_indices, f)
print("✅ Skin saved!")
print("\n🎉 All 3 models done!")