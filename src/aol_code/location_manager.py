"""
Keeps track of the output folder path
and provides functionality to write to progress and settings file.
Use (e.g.):
> from framework.location_manager import location_manager as loc
> loc.initialize(name="AlmostOrthogonalLipschitz", run_nr=1)
"""

import os

from datetime import datetime


class LocationManager:

    def __init__(self):
        self.run_id = None
        self._outfolder_name = None
        self.progress_file_name = None
        self.settings_file_name = None

    def __str__(self):
        return f"Location Manager with Run ID {self.run_id}."

    @property
    def outfolder_name(self):
        return self._outfolder_name

    def initialize(self, name=None, run_nr=None, outfolder_name=None):

        datetime_str = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
        self.run_id = f"{datetime_str}_{name}{run_nr}"

        if outfolder_name is None:
            outfolder_name = os.path.join("outputs", self.run_id)
        os.makedirs(outfolder_name)

        self._outfolder_name = outfolder_name
        self.progress_file_name = os.path.join(outfolder_name, "progress.txt")
        self.settings_file_name = os.path.join(outfolder_name, "settings.txt")

        self.add_settings("LocSettings", run_id=self.run_id)
        self.print_pro(f"Initialized {str(self)}", with_time=True)

    def add_settings(self, name="Settings", **settings):
        if self.settings_file_name is None:
            print(f"Settings {name}: {settings}")
        else:
            with open(self.settings_file_name, "a") as f:
                f.write(f"\n*** {name}:\n")
                for key, val in sorted(settings.items()):
                    f.write(key + ": " + str(val) + "\n")

    def print_pro(self, text, with_time=False, end="\n"):
        text = str(text)
        if with_time:
            text = "\n" + datetime.now().strftime("%H:%M:%S ") + text
        if self.progress_file_name is not None:
            with open(self.progress_file_name, "a") as prog_file:
                prog_file.write(text + end)
        print("# " + text.replace("\n", "\n# "), end=end)


# Define the one (pseudo singleton) location manager to be used everywhere:
location_manager = LocationManager()
