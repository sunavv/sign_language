import os
import cv2
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import threading

class DataCollectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Data Collection Tool")
        
        # Configuration variables
        self.data_dir = './data'
        self.number_of_classes = 3
        self.dataset_size = 100
        self.camera_index = 0
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Setup UI
        self.setup_ui()
        
        # Camera capture setup
        self.cap = None
        self.current_class = 0
        self.current_counter = 0
        self.collecting = False
        
    def setup_ui(self):
        # Frame for configuration
        config_frame = tk.Frame(self.master, padx=10, pady=10)
        config_frame.pack(fill=tk.X)
        
        # Number of classes
        tk.Label(config_frame, text="Number of Classes:").grid(row=0, column=0, sticky='w')
        self.classes_entry = tk.Entry(config_frame)
        self.classes_entry.insert(0, str(self.number_of_classes))
        self.classes_entry.grid(row=0, column=1, padx=5)
        
        # Dataset size
        tk.Label(config_frame, text="Images per Class:").grid(row=1, column=0, sticky='w')
        self.size_entry = tk.Entry(config_frame)
        self.size_entry.insert(0, str(self.dataset_size))
        self.size_entry.grid(row=1, column=1, padx=5)
        
        # Camera index
        tk.Label(config_frame, text="Camera Index:").grid(row=2, column=0, sticky='w')
        self.camera_entry = tk.Entry(config_frame)
        self.camera_entry.insert(0, str(self.camera_index))
        self.camera_entry.grid(row=2, column=1, padx=5)
        
        # Class selection
        tk.Label(config_frame, text="Current Class:").grid(row=3, column=0, sticky='w')
        self.class_var = tk.StringVar()
        self.class_dropdown = ttk.Combobox(config_frame, textvariable=self.class_var, state="readonly")
        self.class_dropdown.grid(row=3, column=1, padx=5, sticky='w')
        
        # Preview frame
        self.preview_label = tk.Label(self.master)
        self.preview_label.pack(pady=10)
        
        # Progress bar
        self.progress_frame = tk.Frame(self.master)
        self.progress_frame.pack(pady=5, fill=tk.X, padx=20)
        
        tk.Label(self.progress_frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=500, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Button frame
        button_frame = tk.Frame(self.master)
        button_frame.pack(pady=10)
        
        # Buttons
        self.start_button = tk.Button(button_frame, text="Start Preview", command=self.start_preview)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.collect_button = tk.Button(button_frame, text="Collect for Class", command=self.start_collection, state=tk.DISABLED)
        self.collect_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_collection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(self.master, text="Ready to start", fg="green")
        self.status_label.pack(pady=10)
        
    def update_class_dropdown(self):
        # Update the class dropdown menu
        values = [f"Class {i}" for i in range(self.number_of_classes)]
        self.class_dropdown['values'] = values
        if values:
            self.class_dropdown.current(0)
    
    def start_preview(self):
        # Update configuration
        try:
            self.number_of_classes = int(self.classes_entry.get())
            self.dataset_size = int(self.size_entry.get())
            self.camera_index = int(self.camera_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
            return
        
        # Update class dropdown
        self.update_class_dropdown()
        
        # Start camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot access camera at index {self.camera_index}")
            return
        
        # Enable collection button
        self.collect_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)
        
        # Start preview thread
        self.preview_thread = threading.Thread(target=self.update_preview, daemon=True)
        self.preview_thread.start()
        
    def update_preview(self):
        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert frame for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame for preview
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to PhotoImage
            from PIL import Image, ImageTk
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            
            # Update preview label
            self.preview_label.config(image=photo)
            self.preview_label.image = photo
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    def start_collection(self):
        # Get selected class
        if not self.class_var.get():
            messagebox.showerror("Error", "Please select a class")
            return
        
        class_index = int(self.class_var.get().split()[-1])
        
        # Disable buttons during collection
        self.collect_button.config(state=tk.DISABLED)
        self.class_dropdown.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Create all class directories if they don't exist
        for i in range(self.number_of_classes):
            class_dir = os.path.join(self.data_dir, str(i))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
        
        # Reset progress bar
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = self.dataset_size
        
        # Start collection thread
        self.collecting = True
        self.collection_thread = threading.Thread(
            target=self.collect_data_for_class, 
            args=(class_index,), 
            daemon=True
        )
        self.collection_thread.start()
        
    def collect_data_for_class(self, class_index):
        # Create class directory
        class_dir = os.path.join(self.data_dir, str(class_index))
        
        # Count existing images in the directory
        existing_files = os.listdir(class_dir) if os.path.exists(class_dir) else []
        existing_images = [f for f in existing_files if f.endswith('.jpg')]
        
        # Determine start counter (use 0 if starting fresh)
        counter = 0
        
        # Update status
        self.status_label.config(text=f"Collecting data for Class {class_index}")
        self.master.update_idletasks()
        
        # Collect images
        while counter < self.dataset_size and self.cap and self.cap.isOpened() and self.collecting:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Save image
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            counter += 1
            
            # Update progress bar and status
            self.progress_bar["value"] = counter
            self.status_label.config(text=f"Class {class_index}: Collected {counter}/{self.dataset_size} images")
            self.master.update_idletasks()
        
        # Collection completed or stopped
        if counter >= self.dataset_size:
            self.status_label.config(text=f"Completed collection for Class {class_index}")
        else:
            self.status_label.config(text=f"Stopped collection for Class {class_index}")
        
        # Re-enable UI
        self.collect_button.config(state=tk.NORMAL)
        self.class_dropdown.config(state="readonly")
        self.stop_button.config(state=tk.DISABLED)
        self.collecting = False
        
    def stop_collection(self):
        # Stop the current collection
        self.collecting = False
        
    def close_app(self):
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        self.master.destroy()

def main():
    root = tk.Tk()
    root.geometry("800x700")
    app = DataCollectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()

if __name__ == "__main__":
    main()