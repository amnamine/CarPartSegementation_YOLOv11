import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os

class CarPartSegApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Part Segmentation")
        self.root.geometry("900x650")
        self.root.configure(bg="#f4f6f9")

        # Load YOLO model
        self.model = YOLO("carpartseg.pt")

        # UI Elements
        self.title_label = tk.Label(root, text="Car Part Segmentation (YOLO11n-Seg)",
                                    font=("Arial", 18, "bold"), bg="#f4f6f9", fg="#2c3e50")
        self.title_label.pack(pady=15)

        self.canvas = tk.Canvas(root, width=800, height=450, bg="white", highlightthickness=2, highlightbackground="#bdc3c7")
        self.canvas.pack(pady=20)

        self.button_frame = tk.Frame(root, bg="#f4f6f9")
        self.button_frame.pack(pady=10)

        self.load_btn = tk.Button(self.button_frame, text="Load Image", command=self.load_image,
                                  font=("Arial", 12), bg="#3498db", fg="white", width=12)
        self.load_btn.grid(row=0, column=0, padx=10)

        self.predict_btn = tk.Button(self.button_frame, text="Predict", command=self.predict,
                                     font=("Arial", 12), bg="#27ae60", fg="white", width=12, state="disabled")
        self.predict_btn.grid(row=0, column=1, padx=10)

        self.reset_btn = tk.Button(self.button_frame, text="Reset", command=self.reset,
                                   font=("Arial", 12), bg="#e74c3c", fg="white", width=12, state="disabled")
        self.reset_btn.grid(row=0, column=2, padx=10)

        self.image_path = None
        self.tk_img = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not self.image_path:
            return

        img = Image.open(self.image_path)
        img = img.resize((800, 450))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        self.predict_btn.config(state="normal")
        self.reset_btn.config(state="normal")

    def predict(self):
        if not self.image_path:
            return

        # Run YOLO prediction
        results = self.model.predict(self.image_path, imgsz=640, conf=0.25)
        res_img = results[0].plot()  # Draw masks & boxes

        # Convert to Tkinter format
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        res_img = Image.fromarray(res_img).resize((800, 450))
        self.tk_img = ImageTk.PhotoImage(res_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def reset(self):
        self.canvas.delete("all")
        self.image_path = None
        self.tk_img = None
        self.predict_btn.config(state="disabled")
        self.reset_btn.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = CarPartSegApp(root)
    root.mainloop()
