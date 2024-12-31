import cv2
import pyperclip


def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)


def web_cam_capture():
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Cannot open webcam")
        return None
    
    path = "webcam.jpg"
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to capture image from webcam")
        cam.release()
        return None
    
    cv2.imwrite(path, frame)
    cam.release()
    return path

def get_clipboard():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("No hay clipboard copiado")
        return None
