"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/train_net.py
"""

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import os
import logging
from collections import OrderedDict
import numpy as np
import cv2

from typing import Any, Dict, List, Set

import torch
from torchtnt.utils.distributed import revert_sync_batchnorm
import torch.nn as nn

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
)


from kmax_deeplab import (
    PanoptickMaXDeepLabDatasetMapper,
    add_kmax_deeplab_config,
    InstanceSegEvaluator,
)
from kmax_deeplab.data.datasets.cityscapes_panoptic_separated import register_all_cityscapes_panoptic

from detectron2.data import MetadataCatalog

import train_net_utils


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"] and cfg.MODEL.KMAX_DEEPLAB.TEST.SEMANTIC_ON:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.KMAX_DEEPLAB.TEST.PANOPTIC_ON:
                evaluator_list.append(train_net_utils.COCOPanopticEvaluatorwithVis(dataset_name, output_folder, save_vis_num=cfg.MODEL.KMAX_DEEPLAB.SAVE_VIS_NUM))
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.KMAX_DEEPLAB.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.KMAX_DEEPLAB.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.KMAX_DEEPLAB.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.KMAX_DEEPLAB.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.KMAX_DEEPLAB.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME in [
            "coco_panoptic_lsj",
            "ade20k_panoptic_lsj",
            "cityscapes_panoptic_lsj"
            ]:
            mapper = PanoptickMaXDeepLabDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)
        

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        name = cfg.SOLVER.LR_SCHEDULER_NAME
        if name == "TF2WarmupPolyLR":
            return train_net_utils.TF2WarmupPolyLR(
                optimizer,
                cfg.SOLVER.MAX_ITER,
                warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                warmup_method=cfg.SOLVER.WARMUP_METHOD,
                power=cfg.SOLVER.POLY_LR_POWER,
                constant_ending=cfg.SOLVER.POLY_LR_CONSTANT_ENDING,
            )
        else:
            return build_lr_scheduler(cfg, optimizer)
    
    @staticmethod
    def produce_overly_visualization(raw_img: np.ndarray, inst_mask: np.ndarray, ratio: float=0.7) -> np.ndarray:
        
        inst_mask = inst_mask * 1.0
        raw_image = cv2.resize(raw_img, fx=ratio, fy=ratio, dsize=None)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        inst_ = cv2.resize(inst_mask, fx=ratio, fy=ratio, dsize=None)
        overlay = np.zeros_like(raw_image)
        overlay[inst_ > 0.0] = (255, 0, 0)
        # print(overlay, np.unique(overlay)); exit()
        viz_ = raw_image + np.uint8(0.5*overlay)

        return viz_

    @classmethod
    def visualize_prediction(cls, input_: dict, output_: dict, path_to_save: str) -> None:

        name_img = input_["file_name"].split("/")[-1].split(".")[0]
        tmp = os.path.join(path_to_save, name_img)
        panoptic_info = output_["panoptic_seg"][1]
        panoptic_info_dict = {}

        for infor_ in panoptic_info:
            panoptic_info_dict[infor_["id"]] = infor_

        if not os.path.isdir(tmp):
            os.makedirs(tmp)

        panoptic_seg = output_["panoptic_seg"][0].cpu().numpy()
        original_image = output_["original_image"]

        original_image = original_image.permute(1, 2, 0).cpu().numpy()

        inst_ids = np.unique(panoptic_seg)

        for id_ in inst_ids:
            if id_ != 0:
                instance_mask = (panoptic_seg == id_)
                viz_ = cls.produce_overly_visualization(original_image, instance_mask)
                info_tmp = panoptic_info_dict[id_]
                content = "{} - {}".format(info_tmp["isthing"], info_tmp["category_id"])
                viz_ = cv2.putText(viz_, content, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, \
                        1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(tmp, "{}.png".format(id_)), viz_)

    @classmethod
    def run_predict(cls, model: nn.Module, input_: dict, path_to_save: str) -> None:

        output_ = model(input_)
        cls.visualize_prediction(input_[0], output_[0], path_to_save)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """

        model.eval()

        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            data_loader_copy = copy.deepcopy(data_loader)
            model_copy = copy.deepcopy(model)

            for idx, input_ in enumerate(data_loader_copy):
                cls.run_predict(model_copy, input_, "{}/visualization".format(cfg.OUTPUT_DIR))

            del data_loader_copy
            del model_copy

            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]

        model.train()
        return results

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        from kmax_deeplab.modeling.backbone.convnext import LayerNorm

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
            LayerNorm
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                hyperparams["name"] = (module_name, module_param_name)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                # Rule for kMaX.
                if "_rpe" in module_name:
                    # relative positional embedding in axial attention.
                    hyperparams["weight_decay"] = 0.0
                if "_cluster_centers" in module_name:
                    # cluster center embeddings.
                    hyperparams["weight_decay"] = 0.0
                if "bias" in module_param_name:
                    # any bias terms.
                    hyperparams["weight_decay"] = 0.0
                if "gamma" in module_param_name:
                    # gamma term in convnext
                    hyperparams["weight_decay"] = 0.0

                params.append({"params": [value], **hyperparams})
        for param_ in params:
            print(param_["name"], param_["lr"], param_["weight_decay"])

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        elif optimizer_type == "ADAM":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.Adam)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_kmax_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="kmax_deeplab")
    return cfg


def main(args):
    cfg = setup(args)
    register_all_cityscapes_panoptic(cfg=cfg, root=cfg.DATA_DIR)

    torch.backends.cudnn.enabled = True
    if args.eval_only:
        model = Trainer.build_model(cfg)
        model.eval()
        model = revert_sync_batchnorm(model)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.model = revert_sync_batchnorm(trainer.model)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
