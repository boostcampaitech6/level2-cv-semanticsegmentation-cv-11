from torch.nn.modules.loss import(
    BCEWithLogitsLoss
)
from monai.losses import (
    PatchAdversarialLoss,
    SoftclDiceLoss, 
    SoftDiceclDiceLoss,
    ContrastiveLoss,
    BendingEnergyLoss, 
    Dice,
    DiceCELoss,
    DiceFocalLoss,
    DiceLoss,
    GeneralizedDiceFocalLoss,
    GeneralizedDiceLoss,
    GeneralizedWassersteinDiceLoss,
    MaskedDiceLoss,
    dice_ce,
    dice_focal,
    generalized_dice,
    generalized_dice_focal,
    generalized_wasserstein_dice,
    DeepSupervisionLoss,
    FocalLoss,
    AsymmetricUnifiedFocalLoss,
)