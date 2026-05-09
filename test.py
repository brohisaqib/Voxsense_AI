from ultralytics import YOLO
import cv2

def test_model():
    try:
        print("📦 Loading YOLO model...")

        model = YOLO("yolov8n.pt")  # ensure file exists or auto-download will happen

        print("✅ Model loaded successfully!")

        # Test image (optional)
        img = cv2.imread("test.jpg")  # apni koi image rakh lena same folder me

        if img is None:
            print("⚠️ No test image found, skipping detection test")
            return

        print("🔍 Running inference...")

        results = model(img)

        print("✅ Detection complete!")
        print(f"📊 Results count: {len(results)}")

        for r in results:
            print(r)

    except Exception as e:
        print("❌ Error loading model:", e)

if __name__ == "__main__":
    test_model()