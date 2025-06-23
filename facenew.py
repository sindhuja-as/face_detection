import streamlit as st
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import joblib
from torchvision import transforms

# 🔹 Load FaceNet model (embedding)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# 🔹 Load MTCNN (for face detection)
mtcnn = MTCNN(keep_all=True, device='cpu')

# 🔹 Load classifier (SVM trained on FaceNet embeddings)
clf = joblib.load("face_classifier.pkl")



# 🔹 Load LFW dataset to get target names
from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.5)

target_names = np.load("model/target_names.npy", allow_pickle=True)


# 🔹 Streamlit UI
st.title("Face Recognition System")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])


if uploaded_file:
    # Load and show image
    img = Image.open(uploaded_file).convert("RGB")
    
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Detect faces
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        for i, box in enumerate(boxes):
            # Crop and preprocess face
            face_crop = img.crop(box).resize((160, 160))
            face_tensor = transforms.ToTensor()(face_crop).unsqueeze(0)
            face_tensor = (face_tensor - 0.5) / 0.5  # Normalize to [-1, 1]

            # Get FaceNet embedding
            with torch.no_grad():
                embedding = facenet(face_tensor).squeeze().numpy()

            # Predict identity
            pred = clf.predict([embedding])[0]
            prob = clf.predict_proba([embedding])[0][pred]
            name = target_names[pred]

            # Show result
            st.success(f"Face {i+1}: **{name}** ({prob*100:.2f}% confidence)")
    else:
        st.warning("No faces detected in the image.")
