import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import nibabel as nib
import numpy as np
from utilities.utility import process_mri_volume
import matplotlib.cm as cm  

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("MRI Viewer")
root.geometry("800x600")
root.resizable(False, False)

frame1 = ctk.CTkFrame(root)
frame2 = ctk.CTkFrame(root)
sidebar = ctk.CTkFrame(root, width=150)

mri_data = None
current_slice = 0
frame2_mri_data = None
new_mri_loaded = 0  # New global variable to track if a new MRI is loaded

img_label = ctk.CTkLabel(frame1, text="")
img_label.pack(expand=True)

def update_image(slice_index):
    global mri_data
    if mri_data is not None:
        slice_data = mri_data[:, :, slice_index]
        img = Image.fromarray((slice_data * 255 / np.max(slice_data)).astype(np.uint8))
        img_tk = ImageTk.PhotoImage(image=img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

def load_mri():
    global mri_data, current_slice, slice_slider, new_mri_loaded
    file_path = filedialog.askopenfilename(filetypes=[("NIFTI files", "*.nii"), ("All Files", "*.*")])
    if file_path:
        mri_volume = nib.load(file_path)
        mri_data = mri_volume.get_fdata()
        new_mri_loaded = 1  # Set to 1 since a new MRI is loaded

        slice_slider.configure(from_=0, to=mri_data.shape[2] - 1)
        current_slice = mri_data.shape[2] // 2
        slice_slider.set(current_slice)
        update_image(current_slice)

# Slider for frame 1
slice_slider = ctk.CTkSlider(frame1, from_=0, to=0, command=lambda val: update_image(int(val)), width=400)
slice_slider.pack(padx=10, pady=50)

# Button to load MRI files
add_button = ctk.CTkButton(frame1, text="+", command=load_mri, font=("Arial", 24),
                           width=50, height=50, corner_radius=25, hover_color="gray")
add_button.place(relx=1.0, rely=1.0, anchor="se", x=-20, y=-20)

def show_frame1():
    frame1.pack(fill="both", expand=True)
    frame2.pack_forget()

# Create an image label for the axial view in frame2 (outside the function)
axial_view_label = ctk.CTkLabel(frame2, text="")
axial_view_label.pack(expand=True)

# Slider for selecting slices in frame2 (initialized outside show_frame2)
frame2_slider = ctk.CTkSlider(frame2, from_=0, to=0,
                               command=lambda val: update_analysis_slices(int(val)), width=600)
frame2_slider.pack(side="bottom", fill="x", padx=50, pady=20)  # Pack the slider initially but it will not show until frame2 is packed

def show_frame2():
    global frame2_slider
    frame1.pack_forget()
    frame2.pack(fill="both", expand=True)

    # Update the slider's range based on the loaded MRI data
    if frame2_mri_data is not None:
        frame2_slider.configure(from_=0, to=frame2_mri_data.shape[2] - 1)
        frame2_slider.set(frame2_mri_data.shape[2] // 2)  # Set the slider to the middle slice
        update_analysis_slices(frame2_slider.get())  # Show the current slice

def update_analysis_slices(slice_index):
    global frame2_mri_data, axial_view_label

    if frame2_mri_data is not None:
        axial_slice = frame2_mri_data[slice_index,:, :]  # Get the slice from frame2_mri_data

        # Normalize the data for colormap application
        normed_slice = (axial_slice - np.min(axial_slice)) / (np.max(axial_slice) - np.min(axial_slice))

        # Apply a heatmap colormap (e.g., "hot" or "jet")
        colormap = cm.hot(normed_slice)  # Returns an RGBA array

        # Convert the RGBA array to an RGB format understood by PIL
        heatmap_img = Image.fromarray((colormap[:, :, :3] * 255).astype(np.uint8))

        # Resize the image as before
        new_width, new_height = 400, 400
        heatmap_img = heatmap_img.resize((new_width, new_height), Image.LANCZOS)
        axial_tk_img = ImageTk.PhotoImage(image=heatmap_img)

        # Update the axial_view_label with the heatmap image
        axial_view_label.configure(image=axial_tk_img)
        axial_view_label.image = axial_tk_img

def analyze():
    global mri_data, axial_view_label, frame2_mri_data, new_mri_loaded

    if mri_data is not None and new_mri_loaded == 1:  # Only analyze if a new MRI is loaded
        # Create a copy of mri_data to use in frame 2
        frame2_mri_data = process_mri_volume(mri_data, threshold=0.5)
        new_mri_loaded = 0  # Reset to 0 after analysis is done

        # Show frame2 where the analysis result will be displayed
        show_frame2()

        # Get the middle slice for the axial view
        axial_slice = frame2_mri_data[frame2_mri_data.shape[2] // 2,:, :]

        # Normalize the data for colormap application
        normed_slice = (axial_slice - np.min(axial_slice)) / (np.max(axial_slice) - np.min(axial_slice))
        # Apply a heatmap colormap (e.g., "hot" or "jet")
        colormap = cm.hot(normed_slice)  # Returns an RGBA array

        # Convert the RGBA array to an RGB format understood by PIL
        heatmap_img = Image.fromarray((colormap[:, :, :3] * 255).astype(np.uint8))

        # Resize the image as before
        new_width, new_height = 400, 400
        heatmap_img = heatmap_img.resize((new_width, new_height), Image.LANCZOS)
        axial_tk_img = ImageTk.PhotoImage(image=heatmap_img)

        # Update the axial_view_label with the heatmap image
        axial_view_label.configure(image=axial_tk_img)
        axial_view_label.image = axial_tk_img
    elif mri_data is None:
        print("No MRI data available. Please load an MRI file first.")
    else:
        show_frame2()
        print("Analysis already performed on this MRI. Please load a new MRI to analyze.")

# Sidebar layout
sidebar.pack(side="left", fill="y", padx=10)

btn_page1 = ctk.CTkButton(sidebar, text="MRI Viewer", command=show_frame1, fg_color="grey", text_color="black")
btn_page1.pack(pady=10)

btn_page2 = ctk.CTkButton(sidebar, text="Analysis Result", command=analyze, fg_color="grey", text_color="black")
btn_page2.pack(pady=10)

is_dark_mode = ctk.get_appearance_mode() == "dark"

def toggle_mode():
    global is_dark_mode
    is_dark_mode = not is_dark_mode
    new_mode = "dark" if is_dark_mode else "light"
    ctk.set_appearance_mode(new_mode)
    toggle_switch.select() if is_dark_mode else toggle_switch.deselect()

toggle_switch = ctk.CTkSwitch(sidebar, text="Light/Dark Mode", command=toggle_mode)
toggle_switch.pack(side="bottom", fill="x", padx=10, pady=10)

if is_dark_mode:
    toggle_switch.select()
else:
    toggle_switch.deselect()

# Initialize the first frame
show_frame1()

root.mainloop()
