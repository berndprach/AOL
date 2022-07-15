"""
Keeps track of directories such as the checkpoint folder,
as well as files such as the progress or settings file,
and also provides functionality to log progress or settings.
Use (e.g.):
> from framework.location_manager import location_manager as loc
> loc.initialize(name="Experiment", run_nr=1)
"""

import os
import tensorflow as tf

from datetime import datetime

from typing import Optional


class LocationManager:

    def __init__(self):
        self.run_nr: Optional[int] = None
        self.name: Optional[str] = None
        self.run_id: Optional[str] = None

        self._outfolder_name = None
        self.progress_file_name: Optional[str] = None
        self.settings_file_name: Optional[str] = None
        self.plots_directory = None
        self.checkpoints_directory = None

    def __str__(self):
        return f"Location manager with run id {self.run_id}."

    @property
    def outfolder_name(self):
        return self._outfolder_name

    def initialize(self, name=None, run_nr=None, outfolder_name=None, force_gpu=False):
        self.reset()

        self.run_nr = run_nr
        self.name = name + str(run_nr)
        self.run_id = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S") + "_" + self.name

        self.prepare_files(outfolder_name=outfolder_name)

        self.print_name()
        self.add_settings("LocSettings", run_id=self.run_id)

        self.print_device_information(force_gpu)
        self.print_pro("Now to the real stuff:", with_time=True)

    def reset(self):
        self.run_nr = None
        self.name = None
        self.run_id = None

        self._outfolder_name = None
        self.progress_file_name = None
        self.settings_file_name = None
        self.plots_directory = None
        self.checkpoints_directory = None

    def prepare_files(self, outfolder_name=None):

        if outfolder_name is None: outfolder_name = os.path.join("outputs", self.run_id)
        self._outfolder_name = outfolder_name

        self.progress_file_name = os.path.join(outfolder_name, "progress.txt")
        self.settings_file_name = os.path.join(outfolder_name, "settings.txt")
        self.plots_directory = os.path.join(outfolder_name, "plots")
        self.checkpoints_directory = os.path.join(outfolder_name, "checkpoints")

        for folder in [self.plots_directory, self.checkpoints_directory]:
            os.makedirs(folder, exist_ok=True)

        with open(os.path.join("outputs", "latest_run_id.txt"), "w") as f:
            f.write(self.run_id + "\n")

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
        print("# " + text.replace("\n", "\n# "), end=end)  # end is a bit of a visual problem here.

    def print_name(self):
        self.print_pro(f"Run ID: {self.run_id}")
        if self.name is None: return None

        line_len = len(self.name) + 8
        text = ""
        text += ("\n" + "   " + "#" * line_len + "\n")
        text += ("   " + "### " + self.name + " ###" + "\n")
        text += ("   " + "#" * line_len + "\n")
        text += ("\n" + datetime.now().strftime("%H:%M:%S ") + "Started\n")

        self.print_pro(text, end="\n")

    def print_device_information(self, force_gpu=False):
        # A bit out of place here but what can you do...

        # Print some general stuff:
        self.print_pro("Tensorflow version: {}".format(tf.__version__))
        nrof_gpus = len([x for x in tf.config.experimental.list_physical_devices() if "GPU" in x.name])
        self.print_pro(f"Number of GPUs available: {nrof_gpus}")
        self.print_pro("Devices: " + ", ".join([x.name.replace("/physical_device:", "")
                                                for x in tf.config.experimental.list_physical_devices()]))
        self.print_pro("Eager mode: " + str(tf.executing_eagerly()) + ".")

        if force_gpu and nrof_gpus == 0:
            self.print_pro("\n\n!!! GPU is required but not found !!!")
            self.print_pro("!!! Terminating now !!!\n\n")
            raise RuntimeError("GPU required but not found!")


# Define the one (pseudo singleton) location manager to be used everywhere:
location_manager = LocationManager()

location_manager.print_pro("Just imported and created the location manager!!")
