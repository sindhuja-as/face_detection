from sklearn.datasets import fetch_lfw_people
import cv2 as cv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN


# Download the dataset (you can set min_faces_per_person to reduce classes)
lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.5)

X = lfw_people.images  # image data (shape: n_samples, h, w)
y = lfw_people.target  # numerical labels
target_names = lfw_people.target_names  # list of person names
print(f"Images shape: {X.shape}, Classes: {len(target_names)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define augmentation pipeline


# Match the original shape (62 height, 47 width)
augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3),
    transforms.Resize((62, 47)),  # Ensure it matches original size
    transforms.ToTensor()
])


# Apply augmentation to all training images
augmented_images = []
augmented_labels = []

for img, label in zip(X_train, y_train):
    for _ in range(2):  # augment twice per image
        aug_img = augment(img)  # `img` is a NumPy array, handled by ToPILImage
        augmented_images.append(aug_img.numpy())
        augmented_labels.append(label)

# Convert back to NumPy
X_train_aug = np.concatenate([X_train[:, np.newaxis, :, :],np.array(augmented_images)], axis=0)
y_train_aug = np.concatenate([y_train, np.array(augmented_labels)], axis=0)

print(f"Augmented Training Set: {X_train_aug.shape}")
from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Initialize MTCNN detector
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)



# 1. Choose an image from X_train
img_array = X_train[1]  # shape: (62, 47), grayscale

# 2. Convert to 3-channel RGB and upscale (MTCNN likes higher-res)
img = Image.fromarray((img_array * 255).astype(np.uint8)).convert("RGB")
img = img.resize((188, 248))  # Upscale to ~3x original size

# 3. Initialize MTCNN
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# 4. Detect faces
boxes, probs = mtcnn.detect(img)
label_idx = y_train[1]  # assuming X_train[0] was used
person_name = target_names[label_idx]
# 5. Visualize
plt.imshow(img)
if boxes is not None:
    for box in boxes:
        plt.gca().add_patch(plt.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor='lime', facecolor='none'
        ))
    plt.title(f"{person_name} – {len(boxes)} face(s) detected")
else:
    plt.title("No faces detected")
plt.axis('off')
plt.show()

from facenet_pytorch import InceptionResnetV1

# Load pretrained FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval()

from torchvision import transforms

# Convert grayscale image to 3-channel RGB and resize to 160x160 (FaceNet input)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

embeddings = []
labels = []

for img, label in zip(X_train, y_train):
    img_rgb = np.stack([img] * 3, axis=-1)  # Convert to RGB
    face = preprocess(img_rgb)
    face = face.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        emb = facenet(face).squeeze().numpy()
    embeddings.append(emb)
    labels.append(label)

X_embeddings = np.array(embeddings)
y_labels = np.array(labels)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create and train classifier (SVM with standard scaling)
clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
clf.fit(X_embeddings, y_labels)
joblib.dump(clf, "face_classifier.pkl")
test_embeddings = []
test_labels = []

for img, label in zip(X_test, y_test):
    img_rgb = np.stack([img] * 3, axis=-1)
    face = preprocess(img_rgb)
    face = face.unsqueeze(0)
    with torch.no_grad():
        emb = facenet(face).squeeze().numpy()
    test_embeddings.append(emb)
    test_labels.append(label)

# Predict
y_pred = clf.predict(test_embeddings)

# Accuracy


print("Accuracy:", accuracy_score(test_labels, y_pred))
print(classification_report(test_labels, y_pred, target_names=target_names))