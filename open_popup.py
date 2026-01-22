import customtkinter as ctk
import tkinter as tk

# Configure appearance and theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Main app window
app = ctk.CTk()
# Global variable to store result
result_var = tk.StringVar()


# Callback for YES
def on_yes():
    result_var.set('Yes')  # Set result to 'y'
    popup.destroy()


# Callback for NO
def on_no():
    result_var.set('No')  # Set result to 'n'
    popup.destroy()


# Function to open popup and wait for response
def open_popup(text, title="Confirm Action"):
    global popup
    popup = ctk.CTkToplevel(app)

    # Define popup size
    popup_width = 300
    popup_height = 150

    # Get screen width and height
    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()

    # Calculate position x, y to center the popup
    x = int((screen_width / 2) - (popup_width / 2))
    y = int((screen_height / 2) - (popup_height / 2))

    popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
    popup.title(title)

    label = ctk.CTkLabel(popup, text=text, font=("Arial", 16))
    label.pack(pady=20)

    # Frame to hold Yes/No buttons
    button_frame = ctk.CTkFrame(popup, fg_color="transparent")
    button_frame.pack(pady=10)

    yes_button = ctk.CTkButton(button_frame, text="Yes", command=on_yes, width=100)
    yes_button.pack(side="left", padx=10)

    no_button = ctk.CTkButton(button_frame, text="No", command=on_no, width=100)
    no_button.pack(side="left", padx=10)

    # Wait for the popup to close before continuing
    popup.grab_set()
    app.wait_window(popup)

    # After popup closes, get result
    response = result_var.get()
    # print(f"User selected: {response}")  # 'y' or 'n'
    return response


