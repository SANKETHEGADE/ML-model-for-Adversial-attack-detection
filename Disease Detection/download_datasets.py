from google.colab import drive
import os, shutil, zipfile

drive.mount('/content/drive', force_remount=True)
os.makedirs('/root/.kaggle', exist_ok=True)
shutil.copy('/content/drive/MyDrive/kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 0o600)
print("✅ Ready!")

for name, slug, folder in [
    ('chest', 'paultimothymooney/chest-xray-pneumonia',              '/content/chest'),
    ('brain', 'navoneel/brain-mri-images-for-brain-tumor-detection', '/content/brain'),
    ('skin',  'kmader/skin-cancer-mnist-ham10000',                   '/content/skin'),
]:
    os.makedirs(folder, exist_ok=True)
    if len(os.listdir(folder)) == 0:
        print(f"Downloading {name}...")
        os.system(f'kaggle datasets download -d {slug} -p {folder}')
        for f in os.listdir(folder):
            if f.endswith('.zip'):
                with zipfile.ZipFile(f'{folder}/{f}') as z:
                    z.extractall(folder)
        print(f"✅ {name} done!")
    else:
        print(f"✅ {name} already exists")