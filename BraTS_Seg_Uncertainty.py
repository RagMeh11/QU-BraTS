from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import auc
import nibabel as nib

from time import time

#############################################3
### Pretty plots
import matplotlib.pyplot as plt

from matplotlib import cycler

plt.style.use("default")
plt.rcParams.update(
  {"lines.linewidth": 1.5,
   "axes.grid": True,
   "grid.linestyle": ":",
   "axes.grid.axis": "both",
   "axes.prop_cycle": cycler('color',
                             ['0071bc', 'd85218', 'ecb01f',
                              '7d2e8d', '76ab2f', '4cbded', 'a1132e']),
   "xtick.top": True,
   "xtick.minor.size": 0,
   "xtick.direction": "in",
   "xtick.minor.visible": True,
   "ytick.right": True,
   "ytick.minor.size": 0,
   "ytick.direction": "in",
   "ytick.minor.visible": True,
   "legend.framealpha": 1.0,
   "legend.edgecolor": "black",
   "legend.fancybox": False,
   "figure.figsize": (2.5, 2.5),
   "figure.autolayout": False,
   "savefig.dpi": 300,
   "savefig.format": "png",
   "savefig.bbox": "tight",
   "savefig.pad_inches": 0.01,
   "savefig.transparent": False
  }
)

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

###############################################

EPS = np.finfo(np.float32).eps

def dice_metric(ground_truth, predictions):
    """

    Returns Dice coefficient for a single example.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target, 
                     with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [W, H, D].

    Returns:
      Dice coefficient overlap (`float` in [0.0, 1.0])
      between `ground_truth` and `predictions`.

    """

    # Cast to float32 type
    ground_truth = ground_truth.astype("float32")
    predictions = predictions.astype("float32")

    # Calculate intersection and union of y_true and y_predict
    intersection = np.sum(predictions * ground_truth)
    union = np.sum(predictions) + np.sum(ground_truth)

    # Calcualte dice metric
    if intersection == 0.0 and union == 0.0:
      dice = 1.0
    else:
      dice = (2. * intersection) / (union)

    return dice


def ftp_ratio_metric(ground_truth, 
                     predictions, 
                     unc_mask,
                     brain_mask):
    """

    Returns Filtered True Positive Ratio for a single example.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                    with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentatation predictions,
                    with shape [W, H, D].
      unc_mask:     `numpy.ndarray`, uncertainty binary mask, where uncertain voxels has value 0 
                    and certain voxels has value 1, with shape [W, H, D].
      brain_mask:   `numpy.ndarray`, brain binary mask, where background voxels has value 0 
                    and forground voxels has value 1, with shape [W, H, D].

    Returns:
      Filtered true positive ratio (`float` in [0.0, 1.0]).

    """

    # Cast to float32 type
    ground_truth = ground_truth.astype("float32")
    predictions = predictions.astype("float32")
    unc_mask = unc_mask.astype("float32")
    brain_mask = brain_mask.astype("float32")

    # Get filtered Filtered TP ratio (We use EPS for numeric stability)
    TP = (predictions * ground_truth) * brain_mask
    tp_before_filtering = TP.sum()  # TP before filtering
    tp_after_filtering = (TP * unc_mask).sum()  # TP after filtering

    ftp_ratio = (tp_before_filtering - tp_after_filtering) / (tp_before_filtering + EPS)

    return ftp_ratio


def ftn_ratio_metric(ground_truth, 
                     predictions, 
                     unc_mask,
                     brain_mask):
    """

    Returns Filtered True Negative Ratio for a single example.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                    with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentatation predictions,
                    with shape [W, H, D].
      unc_mask:     `numpy.ndarray`, uncertainty binary mask, where uncertain voxels has value 0 
                    and certain voxels has value 1, with shape [W, H, D].
      brain_mask:   `numpy.ndarray`, brain binary mask, where background voxels has value 0 
                    and forground voxels has value 1, with shape [W, H, D].

    Returns:
      Filtered true negative ratio (`float` in [0.0, 1.0]).

    """

    # Cast to float32 type
    ground_truth = ground_truth.astype("float32")
    predictions = predictions.astype("float32")
    unc_mask = unc_mask.astype("float32")
    brain_mask = brain_mask.astype("float32")

    # Get filtered Filtered TN ratio (We use EPS for numeric stability)
    TN = ((1-predictions) * (1-ground_truth)) * brain_mask
    tn_before_filtering = TN.sum() # TN before filtering
    tn_after_filtering = (TN * unc_mask).sum() # TN after filtering

    ftn_ratio = (tn_before_filtering - tn_after_filtering) / (tn_before_filtering + EPS)

    return ftn_ratio


def make(ground_truth, 
         predictions, 
         uncertainties,
         brain_mask, 
         thresholds):

    """
    Performs evaluation for a binary segmentation task.

    Args:
      ground_truth:  `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [W, H, D].
      predictions:   `numpy.ndarray`, binary segmentatation predictions,
                     with shape [W, H, D].
      uncertainties: `numpy.ndarray`, uncertainties for `predictions`, 
                     with values in [0, 100] and shape [W, H, D].
      brain_mask:    `numpy.ndarray`, binary brain mask, 
                     with shape [W, H, D].
      thresholds:    `numpy.ndarray`, the cut-off values for `uncertainties`,
                     with shape [K].

    Returns:
      dice: `numpy.ndarray`, the dice for the different uncertainty
        `thresholds`, with shape [K].
      ftp_ratio: `numpy.ndarray`, the FTP ratio for the different uncertainty
        `thresholds`, with shape [K].
      ftn_ratio: `numpy.ndarray`, the FTN ratio for the different uncertainty
        `thresholds`, with shape [K].
    """

    dice = list()
    ftp_ratio = list()
    ftn_ratio = list()

    # Iterate through different uncertainty thresholds
    for th in thresholds:

        # Convert uncertainty to binary mask according to uncertainty threshold
        # voxels with uncertainty greater than threshold are considered uncertain
        # and voxels with uncertainty less than threshold are considered certain
        unc_mask = np.ones_like(uncertainties, dtype='float32')
        unc_mask[uncertainties > th] = 0.0

        # Multiply ground truth and predictions with unc_mask his helps in filtering out uncertain voxels
        # we calculate metric of interest (here, dice) only on unfiltered certain voxels
        ground_truth_filtered = ground_truth * unc_mask
        predictions_filtered = predictions * unc_mask

        # Calculate dice
        dsc_i = dice_metric(ground_truth_filtered, predictions_filtered)
        dice.append(dsc_i)

        # Calculate filtered true positive ratio
        ftp_ratio_i = ftp_ratio_metric(ground_truth, predictions, unc_mask, brain_mask)
        ftp_ratio.append(ftp_ratio_i)

        # Calculate filtered true negative ratio
        ftn_ratio_i = ftn_ratio_metric(ground_truth, predictions, unc_mask, brain_mask)
        ftn_ratio.append(ftn_ratio_i)

    return dice, ftp_ratio, ftn_ratio



def evaluate(ground_truth,
             segmentation,
             whole,
             core,
             enhance,
             brain_mask,
             output_file,
             num_points,
             return_auc=True,
             return_plot=True):
  
    """
    Evaluates a single sample from BraTS.

    Args:
        ground_truth: `str`, path to ground truth segmentation .
        segmentation: `str`, path to segmentation map.
        whole: `str`, path to uncertainty map for whole tumor.
        core: `str`, path to uncertainty map for core tumor.
        enhance: `str`, path to uncertainty map for enhance tumor.
        brain_mask: `str`, path to brain mask.
        output_file: `str`, path to output file to store statistics.
        num_points: `int`, number of uncertainty threshold points.
        return_auc: `bool`, if it is True it returns AUCs.
        return_plot: `bool`, if it is True it returns plots (Dice vs 1 - Unc_thresholds, FTP vs 1 - Unc_thresholds, FTN vs 1 - Unc_thresholds).

    Returns:
        The table (`pandas.DataFrame`) that summarizes the metrics.
    """

    # Define Uncertainty Threshold points
    _UNC_POINTs = np.arange(0.0, 100.0 + EPS, 100.0 / num_points).tolist()
    _UNC_POINTs.reverse()

    # Parse NIFTI files
    GT = nib.load(ground_truth).get_fdata()
    PRED = nib.load(segmentation).get_fdata()
    WT = nib.load(whole).get_fdata()
    TC = nib.load(core).get_fdata()
    ET = nib.load(enhance).get_fdata()
    BM = nib.load(brain_mask).get_fdata()


    # convert mask into binary.
    # useful when you don't have access to the mask, but generating it from T1 image
    # 0 intensity is considered background, anything else is forground
    # works well with BraTS
    BM[BM>0] = 1.0
    
    # Output container
    METRICS = dict()

    ########
    # Whole Tumour: take 1,2, and 4 label as foreground, 0 as background.

    # convert multi-Label GT and Pred to binary class
    GT_bin = np.zeros_like(GT)
    Pred_bin = np.zeros_like(PRED)

    GT_bin[GT > 0] = 1.0
    Pred_bin[PRED > 0] = 1.0

    METRICS["WT_DICE"], METRICS["WT_FTP_RATIO"], METRICS["WT_FTN_RATIO"] = make(GT_bin, Pred_bin, WT, BM, _UNC_POINTs)

    #######
    # Tumour Core: take 1 and 4 label as foreground, 0 and 2 as background.

    # convert multi-Label GT and Pred to binary class
    GT_bin = np.zeros_like(GT)
    Pred_bin = np.zeros_like(PRED)

    GT_bin[GT == 1] = 1.0
    GT_bin[GT == 4] = 1.0
    Pred_bin[PRED == 1] = 1.0
    Pred_bin[PRED == 4] = 1.0

    METRICS["TC_DICE"], METRICS["TC_FTP_RATIO"], METRICS["TC_FTN_RATIO"] = make(GT_bin, Pred_bin, TC, BM, _UNC_POINTs)

    ##########
    # Enhancing Tumour: take 4 label as foreground, 0, 1, and 2 as bacground.

    # convert multi-Label GT and Pred to binary class
    GT_bin = np.zeros_like(GT)
    Pred_bin = np.zeros_like(PRED)

    GT_bin[GT == 4] = 1.0
    Pred_bin[PRED == 4] = 1.0

    METRICS["ET_DICE"], METRICS["ET_FTP_RATIO"], METRICS["ET_FTN_RATIO"] = make(GT_bin, Pred_bin, ET, BM, _UNC_POINTs)


    ##########
    # save plot

    if return_plot:

        # create a plot for Dice vs 100 - Unc_Thres, FTP vs 100 - Unc_Thres, FTN vs 100 - Unc_Thres for all three tumour types: WT, TC, ET 
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12.0, 8.0), sharex=True)
        
        # loop through Different Metrics
        for j, met in enumerate(["DICE", "FTP_RATIO", "FTN_RATIO"]):

             # loop throug different Tumour Type
            for i, typ in enumerate(["WT", "TC", "ET"]): 

            	# plot 100 - _Unc_threshold on X-axis and Metric on Y-axis. Calculate AUC also
                axes[j,i].plot(100 - np.array(_UNC_POINTs), np.array(METRICS[typ+"_"+met]), 
               	               color=COLORS[i], 
               	               # marker='o', 
               	               label='AUC: {:.4f}'.format(auc(100 - np.array(_UNC_POINTs), np.array(METRICS[typ+"_"+met]))/100.0))

                # set ylabel for first column
                if i == 0:
                    axes[j, i].set(ylabel=met)

                # set title for first row
                if j == 0:
                    axes[j, i].set(title=typ) 
        
                # set xlabel for last row          
                if j == 2:
                    axes[j, i].set(xlabel="1 - Uncertainty Threshold")

                axes[j,i].set(ylim = (0.00,1.0001))
                axes[j,i].set(xlim = (0.00,100.0001))


        [ax.legend() for ax in axes.flatten()]

        fig.savefig(output_file+'.png', dpi=300, format="png", trasparent=True)


    ################
    # Print to CSV

    if not return_auc:
    
        # Returns <thresholds: [DICE_{type}, FTP_RATIO_{type}, FTN_RATIO_{type}]>
        METRICS["THRESHOLDS"] = _UNC_POINTs
        df = pd.DataFrame(METRICS).set_index("THRESHOLDS")
        df.to_csv(output_file+'.csv')
    
        return df
    
    else:
    
        # Returns <{type}: [DICE_AUC, FTP_RATIO_AUC, FTN_RATIO_AUC]>
        df = pd.DataFrame(index=["WT", "TC", "ET"],
                          columns=["DICE_AUC", "FTP_RATIO_AUC", "FTN_RATIO_AUC"],
                          dtype=float)
        
        for ttype in df.index:
            df.loc[ttype, "DICE_AUC"]      = auc(_UNC_POINTs, METRICS["{}_DICE".format(ttype)]) / 100.0
            df.loc[ttype, "FTP_RATIO_AUC"] = auc(_UNC_POINTs, METRICS["{}_FTP_RATIO".format(ttype)]) / 100.0
            df.loc[ttype, "FTN_RATIO_AUC"] = auc(_UNC_POINTs, METRICS["{}_FTN_RATIO".format(ttype)]) / 100.0
    
        df.index.names = ["TUMOR_TYPE"]
        df.to_csv(output_file+'.csv')
    
        return df








########################################################################################################
##
########################################################################################################

parser = argparse.ArgumentParser(description="Uncertainty Analysis on BraTS dataset")

parser.add_argument("-s",
                    "--segmentation",
                    type=str,
                    required=True,
                    help="Path to predicted segmentation map.")

parser.add_argument("-w",
                    "--whole",
                    type=str,
                    required=True,
                    help="Path to uncertainty map for whole tumor.")

parser.add_argument("-c",
                    "--core",
                    type=str,
                    required=True,
                    help="Path to uncertainty map for core tumor.")

parser.add_argument("-e",
                    "--enhance",
                    type=str,
                    required=True,
                    help="Path to uncertainty map for enhance tumor.")

parser.add_argument("-r",
                    "--ground_truth",
                    type=str,
                    required=True,
                    help="Path to ground truth segmentation.")

parser.add_argument("-m",
                    "--brain_mask",
                    type=str,
                    required=True,
                    help="Path to brain mask. You can also provide T1 MR images path.")

parser.add_argument("-o",
                    "--output_file",
                    type=str,
                    required=True,
                    help="Path to output file to store statistics.")

parser.add_argument("-n",
                    "--num_points",
                    type=int,
                    default=40,
                    help="Number of threshold points.")

parser.add_argument("-a",
                    "--return_auc",	
                    action='store_true', 
                    help='If this is True, then returns AUCs. Default: False')

parser.add_argument("-p",
                    "--return_plot",	
                    action='store_true', 
                    help='If this is True, then returns Plots. Default: False')

args = parser.parse_args()


# to calculate time for each subject

start = time()

evaluate(ground_truth=args.ground_truth,
         segmentation=args.segmentation,
         whole=args.whole,
         core=args.core,
         enhance=args.enhance,
         brain_mask=args.brain_mask,
         output_file=args.output_file,
         num_points=args.num_points,
         return_auc=args.return_auc,
         return_plot=args.return_plot)

print("Total Analysis Time: {:.02f} \n".format(time() - start))

