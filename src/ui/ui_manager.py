import customtkinter as ctk
from .main_ui import MainUI
from .calibrate_ui import CalibrateUI
from .detect_ui import DetectUI

class UIManager:
    def __init__(self, root, functions):
        self.root = root
        self.functions = functions
        self.current_ui = None
        self.ui_instances = {}

    def show_ui(self, ui_name):
        # Forget or destroy the current UI
        if self.current_ui:
            self.current_ui.pack_forget()
            self.current_ui.destroy()
            self.current_ui = None

        # Create or retrieve the UI instance
        if ui_name not in self.ui_instances:
            if ui_name == "main":
                self.ui_instances[ui_name] = MainUI(self.root, self, self.functions)
            elif ui_name == "calibrate":
                self.ui_instances[ui_name] = CalibrateUI(self.root, self, self.functions)
            elif ui_name == "detect":
                self.ui_instances[ui_name] = DetectUI(self.root, self, self.functions)
            else:
                raise ValueError(f"Unknown UI: {ui_name}")

        self.current_ui = self.ui_instances[ui_name]
        self.current_ui.pack(fill="both", expand=True)