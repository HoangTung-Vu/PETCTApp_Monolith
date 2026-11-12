"""Result file names use a short model tag, not the long nnU-Net folder name."""
from src.gui.workers.core import ENGINE_NNUNET_MODEL, TOTALSEG_MODEL, model_file_tag


def test_nnunet_preset_is_short():
    assert model_file_tag(ENGINE_NNUNET_MODEL) == "nnUNet"
    assert model_file_tag(f" {ENGINE_NNUNET_MODEL}/ ") == "nnUNet"


def test_other_nnunet_name_keeps_trainer():
    assert model_file_tag("nnUNetTrainer_150epochs__nnUNetPlans__3d_fullres") == "nnUNetTrainer_150epochs"


def test_other_names_unchanged():
    assert model_file_tag("my_model") == "my_model"
    assert model_file_tag(TOTALSEG_MODEL) == TOTALSEG_MODEL
