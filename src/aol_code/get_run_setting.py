from aol_code.location_manager import location_manager as loc
from dataclasses import dataclass, asdict, field
from typing import List


@dataclass
class RunSetting:
    # Data:
    dataset: str = "cifar10"
    train_dataset_size: int = 50_000
    nrof_classes: int = 10
    batch_size: int = 250
    # Model:
    model_name: str = "patch"
    size_parameter: int = 16
    # Loss:
    loss_offset: float = 2 ** (1 / 2)
    loss_temperature: float = 1 / 4
    # Optimization:
    lr: float = 1e-3
    weight_decay_coefficient: float = 5 * 10 ** -4
    nrof_epochs: int = 1000
    lr_drops: List[int] = field(default_factory=lambda: [900, 990, 999])


def get_run_setting(run_id: int):
    rs = RunSetting()
    # run_id == 0 .. all defaults.

    # Different sizes:
    if run_id == 1: rs.size_parameter = 32
    if run_id == 2: rs.size_parameter = 48
    if run_id == 3: rs.size_parameter = 64

    # Different models:
    if run_id == 10: rs.model_name = "ffc"
    if run_id == 11: rs.model_name = "conv"
    if run_id == 12: rs.model_name = "std_conv"

    # Different loss hyperparameters:
    if 20 <= run_id < 30:
        base_offset = 1
        if run_id == 20: base_offset = 1 / 16
        if run_id == 21: base_offset = 1 / 4
        if run_id == 22: base_offset = 1
        if run_id == 23: base_offset = 4
        if run_id == 24: base_offset = 16

        rs.loss_offset = 2 ** (1 / 2) * base_offset
        rs.loss_temperature = base_offset / 4

    # Experiments on Cifar100, with different sizes:
    if 30 <= run_id <= 40:
        rs.dataset = "cifar100"
        rs.nrof_classes = 100
        if run_id == 31: rs.size_parameter = 32
        if run_id == 32: rs.size_parameter = 48
        if run_id == 33: rs.size_parameter = 64

    # Log run settings:
    loc.add_settings(name="Run Settings",
                     **asdict(rs))

    return rs
