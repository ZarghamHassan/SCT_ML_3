import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import joblib
import numpy as np
from skimage.transform import resize
from skimage.io import imread

def preprocess_image(image_path):
    img = imread(image_path)
    img_resized = resize(img, (64, 64, 3), preserve_range=True)
    img_normalized = img_resized / 255.0
    return img_normalized.flatten()

def open_file():
    global model  # Ensure model is accessible within this function
    try:
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if filepath:
            img = Image.open(filepath)
            img_resized = img.resize((500, 500))
            img_display = ImageTk.PhotoImage(img_resized)
            img_label.config(image=img_display)
            img_label.image = img_display
            
            result_label.config(text="Processing...", fg="blue")
            root.update_idletasks()  # Update the GUI
            
            image_data = np.array([preprocess_image(filepath)])
            prediction = model.predict(image_data)

            # Generate simple output based on the prediction
            if prediction == 0:
                result_text = "It is a cat."
            else:
                result_text = "It is a dog."

            result_label.config(text=result_text, fg="green")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Load model at startup
try:
    model = joblib.load('svm_cats_dogs_model.pkl')
except Exception as e:
    messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
    exit(1)  # Exit if model fails to load

# Initialize Tkinter GUI
root = tk.Tk()
root.title("Cats vs Dogs Classifier")
root.geometry("900x900")
root.configure(bg="#f0f0f0")

# Use frames for layout
main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(pady=20)

load_button = tk.Button(main_frame, text="Load Image", command=open_file, bg="#4CAF50", fg="white", font=("Arial", 18), relief="flat")
load_button.grid(row=0, column=0, pady=20, padx=10)

img_label = tk.Label(main_frame, bg="#f0f0f0")
img_label.grid(row=1, column=0)

result_label = tk.Label(main_frame, text="Prediction: ", font=("Roboto", 24), bg="#f0f0f0")
result_label.grid(row=2, column=0, pady=20)

exit_button = tk.Button(main_frame, text="Exit", command=root.quit, bg="#f44336", fg="white", font=("Arial", 18), relief="flat")
exit_button.grid(row=3, column=0, pady=20, padx=10)

root.mainloop()
