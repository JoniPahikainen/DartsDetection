import customtkinter as ctk

class MainUI(ctk.CTkFrame):
    def __init__(self, parent, ui_manager, functions):
        super().__init__(parent, corner_radius=10)
        self.ui_manager = ui_manager
        self.functions = functions

        # Title
        title_label = ctk.CTkLabel(self, text="Dart Score Detection", font=("Arial", 24))
        title_label.pack(pady=20)

        # Navigation buttons
        main_button = ctk.CTkButton(self, text="Main Menu", state="disabled", fg_color="blue")
        main_button.pack(pady=10)

        calibrate_button = ctk.CTkButton(self, text="Calibrate", command=lambda: self.ui_manager.show_ui("calibrate"), fg_color="blue")
        calibrate_button.pack(pady=10)

        detect_button = ctk.CTkButton(self, text="Detect", command=lambda: self.ui_manager.show_ui("detect"), fg_color="blue")
        detect_button.pack(pady=10)

        # Placeholder for additional content
        content_label = ctk.CTkLabel(self, text="Select an option to proceed", font=("Arial", 16))
        content_label.pack(pady=20)