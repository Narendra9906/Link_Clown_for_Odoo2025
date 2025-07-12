import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import mediapipe as mp
import numpy as np
import cv2

# Load ResNet50 once globally
resnet = resnet50(pretrained=True)
resnet.eval()

# Transform pipeline
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_image_features(img_pil):
    img_tensor = img_transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor)
    return features.squeeze().numpy()

def estimate_user_chest(image_pil):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    h, w, _ = image.shape
    x1, y1 = int(left.x * w), int(left.y * h)
    x2, y2 = int(right.x * w), int(right.y * h)

    chest_pixels = np.linalg.norm([x2 - x1, y2 - y1])
    return round(chest_pixels, 2)

def predict_fit(user_img_pil, garment_img_pil, garment_chest_inch=None):
    user_feat = extract_image_features(user_img_pil)
    garment_feat = extract_image_features(garment_img_pil)
    sim = cosine_similarity([user_feat], [garment_feat])[0][0]

    media_pipe_estimate = ""
    if garment_chest_inch:
        user_chest_pixels = estimate_user_chest(user_img_pil)
        if user_chest_pixels:
            pixel_per_inch = user_chest_pixels / garment_chest_inch
            estimated_user_chest = user_chest_pixels / pixel_per_inch
            media_pipe_estimate = f"(Estimated user chest: {estimated_user_chest:.1f} in)"

    if sim >= 0.85:
        fit = "✅ Good Fit"
    elif sim >= 0.75:
        fit = "⚠️ May Be Tight/Loose"
    else:
        fit = "❌ Poor Fit"

    return {
        "fit": fit,
        "similarity": float(sim),
        "info": media_pipe_estimate
    }