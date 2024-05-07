# %%
"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from jmp.lightning import Runner, Trainer
from jmp.models.gemnet.config import BackboneConfig
from jmp.modules.ema import EMA
from jmp.modules.transforms.normalize import NormalizationConfig as NC
from jmp.tasks.config import AdamWConfig
from jmp.tasks.finetune import dataset_config as DC
from jmp.tasks.finetune import matbench, md22, qm9, qmof, rmd17, spice
from jmp.tasks.finetune.base import (
    CheckpointBestConfig,
    EarlyStoppingConfig,
    FinetuneConfigBase,
    FinetuneModelBase,
    MulticlassClassificationTargetConfig,
    PrimaryMetricConfig,
    RLPConfig,
    WarmupCosRLPConfig,
)
from jmp.utils.param_specific_util import make_parameter_specific_optimizer_config

FinetuneConfigBase.set_seed(42)

BASE_DATASET_PATH = Path("/mnt/shared/datasets")
SCALE_FILE_PATH = Path("/path/to/gemnet/scale_files/large.pt")
PRETRAINED_CKPT_PATH = Path("/path/to/pretrained/checkpoint.ckpt")


def config_(
    config: FinetuneConfigBase,
    *,
    batch: int,
    scalar: bool = False,
):
    # Large model
    config.backbone = BackboneConfig.large()
    config.embedding.embedding_size = config.backbone.emb_size_atom
    # config.ln = False
    # config.backbone.ln = False
    # config.backbone.replace_scale_factors_with_ln = False
    config.backbone.scale_basis = False

    # Misc
    config.meta["ckpt_path"] = str(PRETRAINED_CKPT_PATH.absolute())
    config.backbone.scale_file = str(SCALE_FILE_PATH.absolute())
    config.meta["ema_backbone"] = True
    config.trainer.precision = "16-mixed"

    # Stopping criteria
    if isinstance(config, rmd17.RMD17Config):
        config.trainer.max_epochs = 100000
        config.trainer.max_time = "07:00:00:00"
        config.early_stopping = EarlyStoppingConfig(
            patience=1000,
            min_delta=1.0e-8,
            min_lr=1.0e-10,
        )
    else:
        config.trainer.max_epochs = 500
        config.trainer.max_time = "07:00:00:00"
        config.early_stopping = EarlyStoppingConfig(
            patience=50,
            min_delta=1.0e-8,
            min_lr=1.0e-8,
        )
    # Checkpointing
    config.ckpt_best = CheckpointBestConfig()

    # Training speedup by disabling some features
    config.trainer.optimizer.log_grad_norm = True
    config.trainer.optimizer.log_grad_norm_per_param = False
    config.trainer.optimizer.log_param_norm = False
    config.trainer.optimizer.log_param_norm_per_param = False
    config.trainer.supports_parameter_hooks = False
    config.trainer.supports_skip_batch_exception = False
    config.trainer.logging.wandb.log_model = False

    # Optimizer
    config.optimizer = AdamWConfig(
        lr=8.0e-5,
        amsgrad=False,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )
    config.trainer.gradient_clip_val = 1.0
    config.trainer.gradient_clip_algorithm = "value"

    # LR Scheduler
    if isinstance(config, rmd17.RMD17Config):
        config.lr_scheduler = WarmupCosRLPConfig(
            warmup_epochs=5,
            warmup_start_lr_factor=1.0e-1,
            should_restart=False,
            max_epochs=32,
            min_lr_factor=0.1,
            rlp=RLPConfig(mode="min", patience=10, factor=0.8),
        )
    else:
        config.lr_scheduler = WarmupCosRLPConfig(
            warmup_epochs=5,
            warmup_start_lr_factor=1.0e-1,
            should_restart=False,
            max_epochs=32,
            min_lr_factor=0.1,
            rlp=RLPConfig(mode="min", patience=3, factor=0.8),
        )

    config.parameter_specific_optimizers = make_parameter_specific_optimizer_config(
        config,
        config.backbone.num_blocks,
        {
            "embedding": 0.3,
            "blocks_0": 0.55,
            "blocks_1": 0.40,
            "blocks_2": 0.30,
            "blocks_3": 0.40,
            "blocks_4": 0.55,
            "blocks_5": 0.625,
        },
    )

    # Passed args
    config.project = "4_11_ft_lg_jmp_testing"
    config.batch_size = batch
    if scalar:
        config.backbone.regress_forces = False
        config.backbone.direct_forces = False


MD17_STATS: dict[rmd17.RMD17Molecule, dict[str, NC]] = {
    "aspirin": {
        "y": NC(mean=17617.379355234374, std=0.2673998440577667),
        "force": NC(mean=0.0, std=1.2733363),
    },
    "azobenzene": {
        "y": NC(mean=15553.118351233397, std=0.2866098335926971),
        "force": NC(mean=0.0, std=1.2940075),
    },
    "benzene": {
        "y": NC(mean=6306.374855859375, std=0.10482645661015047),
        "force": NC(mean=0.0, std=0.90774584),
    },
    "ethanol": {
        "y": NC(mean=4209.534573266602, std=0.18616576961275716),
        "force": NC(mean=0.0, std=1.1929188),
    },
    "malonaldehyde": {
        "y": NC(mean=7254.903633896484, std=0.1812291921138577),
        "force": NC(mean=0.0, std=1.302443),
    },
    "naphthalene": {
        "y": NC(mean=10478.192319667969, std=0.24922674853668708),
        "force": NC(mean=0.0, std=1.3102233),
    },
    "paracetamol": {
        "y": NC(mean=13998.780924130859, std=0.26963984094801224),
        "force": NC(mean=0.0, std=1.2707518),
    },
    "salicylic": {
        "y": NC(mean=13472.110348867187, std=0.2437920552529055),
        "force": NC(mean=0.0, std=1.3030343),
    },
    "toluene": {
        "y": NC(mean=7373.347077485351, std=0.22534282741069667),
        "force": NC(mean=0.0, std=1.246547),
    },
    "uracil": {
        "y": NC(mean=11266.351949697266, std=0.2227113171300836),
        "force": NC(mean=0.0, std=1.3692871),
    },
}


def md17_configs(is_grad: bool = True):
    for molecule in MD17_STATS:
        config = rmd17.RMD17Config.draft()
        config.molecule = molecule

        config.name = f"md17_{molecule}"
        config.model_type = "forces"
        if is_grad:
            config.gradient_forces = True
            config.trainer.inference_mode = False
            config.name += "_grad"

        config_(config, batch=1, scalar=is_grad)
        config.trainer.precision = "32-true"

        config.normalization = MD17_STATS[molecule]
        config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")

        # Dataset
        base_path = BASE_DATASET_PATH / "rmd17"
        config.train_dataset = DC.rmd17_config(molecule, base_path, "train")
        config.val_dataset = DC.rmd17_config(molecule, base_path, "val")
        config.test_dataset = DC.rmd17_config(molecule, base_path, "test")

        yield config, rmd17.RMD17Model


MD22_NORM: dict[md22.MD22Molecule, dict[str, NC]] = {
    "Ac-Ala3-NHMe": {
        "y": NC(mean=26913.953, std=0.35547638),
        "force": NC(mean=1.4777572e-11, std=1.1291506),
    },
    "DHA": {
        "y": NC(mean=27383.035, std=0.41342595),
        "force": NC(mean=5.5828797e-10, std=1.1258113),
    },
    "stachyose": {
        "y": NC(mean=68463.59, std=0.5940788),
        "force": NC(mean=4.9331733e-10, std=1.1104717),
    },
    "AT-AT": {
        "y": NC(mean=50080.08, std=0.47309175),
        "force": NC(mean=1.3477714e-09, std=1.2109985),
    },
    "AT-AT-CG-CG": {
        "y": NC(mean=101034.23, std=0.680055),
        "force": NC(mean=3.476294e-10, std=1.2021886),
    },
    "buckyball-catcher": {
        "y": NC(mean=124776.7, std=0.64662045),
        "force": NC(mean=6.8671324e-10, std=1.0899031),
    },
    "double-walled_nanotube": {
        "y": NC(mean=338224.16, std=3.3810701),
        "force": NC(mean=7.239396e-11, std=1.0137014),
    },
}

MD22_MOLECULES: list[tuple[md22.MD22Molecule, int, bool, bool]] = [
    ("Ac-Ala3-NHMe", 2, False, True),
    ("DHA", 1, True, False),
    ("stachyose", 1, True, True),
    ("stachyose", 1, True, True),
    ("AT-AT", 1, True, True),
    ("AT-AT-CG-CG", 1, True, True),
    ("buckyball-catcher", 1, True, True),
    ("double-walled_nanotube", 1, False, True),
]


def md22_configs():
    for molecule, bsz, is_grad, amp in MD22_MOLECULES:
        config = md22.MD22Config.draft()
        config.molecule = molecule

        config.name = f"md22_{molecule}"
        config.model_type = "forces"
        if is_grad:
            config.gradient_forces = True
            config.trainer.inference_mode = False
            config.name += "_grad"
        else:
            config.name += "_direct"

        config_(config, batch=bsz, scalar=is_grad)

        config.normalization = MD22_NORM[molecule]
        config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")

        base_path = BASE_DATASET_PATH / "md22"
        config.train_dataset = DC.md22_config(molecule, base_path, "train")
        config.val_dataset = DC.md22_config(molecule, base_path, "val")
        config.test_dataset = DC.md22_config(molecule, base_path, "test")

        if amp:
            config.trainer.precision = "16-mixed"
        else:
            config.trainer.precision = "32-true"

        yield config, md22.MD22Model


QM9_NORMALIZATION: dict[str, NC] = {
    "mu": NC(mean=2.674587, std=1.5054824),
    "alpha": NC(mean=75.31013, std=8.164021),
    "eps_HMO": NC(mean=-6.5347567, std=0.59702325),
    "eps_LUMO": NC(mean=0.323833, std=1.273586),
    "delta_eps": NC(mean=6.8585854, std=1.283122),
    "R_2_Abs": NC(mean=1189.6819, std=280.0421),
    "ZPVE": NC(mean=-0.00052343315, std=0.04904531),
    "U_0": NC(mean=0.0028667436, std=1.0965848),
    "U": NC(mean=0.0028711546, std=1.0941933),
    "H": NC(mean=0.0029801112, std=1.0942822),
    "G": NC(mean=0.000976671, std=1.101572),
    "c_v": NC(mean=-0.005799451, std=2.2179737),
}
QM9_TARGETS = [
    "mu",
    "alpha",
    "eps_HMO",
    "eps_LUMO",
    "delta_eps",
    "R_2_Abs",
    "ZPVE",
    "U_0",
    "U",
    "H",
    "G",
    "c_v",
]
QM9_REDUCTION: dict[str, Literal["sum", "mean", "max"]] = {
    "mu": "sum",
    "alpha": "sum",
    "eps_HMO": "sum",
    "eps_LUMO": "sum",
    "delta_eps": "sum",
    "R_2_Abs": "sum",
    "ZPVE": "sum",
    "U_0": "sum",
    "U": "sum",
    "H": "sum",
    "G": "sum",
    "c_v": "sum",
}


def qm9_configs():
    for target in QM9_TARGETS:
        config = qm9.QM9Config.draft()
        config.graph_scalar_targets = [target]
        config.name = f"qm9_{target}"
        config.normalization = QM9_NORMALIZATION
        # config.use_scalar_head_for_all_targets = True
        config.graph_scalar_reduction = QM9_REDUCTION
        config_(config, batch=48, scalar=True)
        config.primary_metric = PrimaryMetricConfig(name=f"{target}_mae", mode="min")

        if target == "R_2_Abs":
            config.output_head = qm9.SpatialExtentConfig()

            # Also, we don't use any normalization for this target
            config.normalization = {}
        else:
            config.output_head = qm9.DefaultOutputHeadConfig()

        base_path = BASE_DATASET_PATH / "qm9"
        config.train_dataset = DC.qm9_config(base_path, "train")
        config.val_dataset = DC.qm9_config(base_path, "val")
        config.test_dataset = DC.qm9_config(base_path, "test")

        yield config, qm9.QM9Model


QMOF_NORMALIZATION: dict[str, NC] = {"y": NC(mean=2.1866251527, std=1.175752521125648)}


def qmof_configs():
    config = qmof.QMOFConfig.draft()

    config.name = "qmof"
    config.graph_scalar_reduction_default = "mean"
    config.normalization = QMOF_NORMALIZATION
    config_(config, batch=4, scalar=True)

    config.primary_metric = PrimaryMetricConfig(name="y_mae", mode="min")

    base_path = BASE_DATASET_PATH / "qmof"
    config.train_dataset = DC.qmof_config(base_path, "train")
    config.val_dataset = DC.qmof_config(base_path, "val")
    config.test_dataset = DC.qmof_config(base_path, "test")

    yield config, qmof.QMOFModel


SPICE_DATASETS: list[spice.SPICEDataset] = [
    "solvated_amino_acids",
    "dipeptides",
]
SPICE_NORMALIZATION: dict[str, dict[str, NC]] = {
    "dipeptides": {
        "y": NC(mean=-31213.615, std=4636.815),
        "force": NC(mean=3.3810358e-07, std=0.5386545),
    },
    "solvated_amino_acids": {
        "y": NC(mean=-60673.68, std=3310.6692),
        "force": NC(mean=2.7950014e-07, std=0.81945145),
    },
}


def spice_configs(is_grad: bool = True):
    for dataset in SPICE_DATASETS:
        config = spice.SPICEConfig.draft()

        config.dataset = dataset
        config.name = f"spice_{dataset}"
        config.model_type = "forces"
        if is_grad:
            config.gradient_forces = True
            config.trainer.inference_mode = False
            config.name += "_grad"

        config_(config, batch=2, scalar=is_grad)
        config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")

        config.normalization = SPICE_NORMALIZATION[dataset]

        base_path = BASE_DATASET_PATH / "spice"
        config.train_dataset = DC.spice_config(dataset, base_path, "train")
        config.val_dataset = DC.spice_config(dataset, base_path, "val")
        config.test_dataset = DC.spice_config(dataset, base_path, "test")

        yield config, spice.SPICEModel


MATBENCH_DATASETS: list[tuple[matbench.MatbenchDataset, int, bool]] = [
    ("mp_is_metal", 3, False),
    ("jdft2d", 3, False),
    ("phonons", 8, True),
    ("dielectric", 8, True),
    ("log_gvrh", 8, True),
    ("log_kvrh", 8, True),
    ("perovskites", 8, True),
    ("mp_gap", 2, False),
    ("mp_e_form", 6, True),
]
MATBENCH_NORMALIZATION: dict[str, dict[str, NC]] = {
    "jdft2d_fold0": {"y": NC(mean=110.63706001904778, std=132.02502987887982)},
    "jdft2d_fold1": {"y": NC(mean=100.05996525195053, std=114.26362221432791)},
    "jdft2d_fold2": {"y": NC(mean=101.59535193788061, std=112.45760038504558)},
    "jdft2d_fold3": {"y": NC(mean=99.43551549230911, std=109.9220303290942)},
    "jdft2d_fold4": {"y": NC(mean=95.50851385805468, std=76.27587565670332)},
    "phonons_fold0": {"y": NC(mean=602.9007780432183, std=471.03858838413055)},
    "phonons_fold1": {"y": NC(mean=613.7996473907606, std=486.75099875453213)},
    "phonons_fold2": {"y": NC(mean=619.3868976573087, std=495.2975486965762)},
    "phonons_fold3": {"y": NC(mean=609.7402387661577, std=462.3438660855412)},
    "phonons_fold4": {"y": NC(mean=595.4547502676089, std=476.8567310885976)},
    "dielectric_fold0": {"y": NC(mean=2.417849270334958, std=2.208662738016193)},
    "dielectric_fold1": {"y": NC(mean=2.3716402963883074, std=2.1271523121706912)},
    "dielectric_fold2": {"y": NC(mean=2.354418196731436, std=1.5712251872961516)},
    "dielectric_fold3": {"y": NC(mean=2.392308273978868, std=2.0724149898647544)},
    "dielectric_fold4": {"y": NC(mean=2.3891527750974495, std=2.011348533899877)},
    "log_gvrh_fold0": {"y": NC(mean=1.5557434474198688, std=0.37307197984408746)},
    "log_gvrh_fold1": {"y": NC(mean=1.5584101768747889, std=0.36743473539736493)},
    "log_gvrh_fold2": {"y": NC(mean=1.55746252819908, std=0.36800038945046654)},
    "log_gvrh_fold3": {"y": NC(mean=1.5543022349873286, std=0.3684552493569905)},
    "log_gvrh_fold4": {"y": NC(mean=1.5595705795473838, std=0.37039750391284176)},
    "log_kvrh_fold0": {"y": NC(mean=1.880001033036957, std=0.36820395518377785)},
    "log_kvrh_fold1": {"y": NC(mean=1.883820392919235, std=0.3679308395031994)},
    "log_kvrh_fold2": {"y": NC(mean=1.883778380784775, std=0.3724392829717956)},
    "log_kvrh_fold3": {"y": NC(mean=1.8828457515367547, std=0.3731179944882516)},
    "log_kvrh_fold4": {"y": NC(mean=1.8862681006404232, std=0.3671596024523317)},
    "perovskites_fold0": {"y": NC(mean=1.4726657310327749, std=0.7384309800882398)},
    "perovskites_fold1": {"y": NC(mean=1.4690728968876414, std=0.736635027626099)},
    "perovskites_fold2": {"y": NC(mean=1.4702980269132337, std=0.7456716470700677)},
    "perovskites_fold3": {"y": NC(mean=1.46773815420175, std=0.7431740904365189)},
    "perovskites_fold4": {"y": NC(mean=1.478002311375268, std=0.7435117654840315)},
    "mp_gap_fold0": {"y": NC(mean=1.236432240252091, std=1.6096424425437108)},
    "mp_gap_fold1": {"y": NC(mean=1.2345678083402052, std=1.6044708412420103)},
    "mp_gap_fold2": {"y": NC(mean=1.2352391374131229, std=1.6058092380465256)},
    "mp_gap_fold3": {"y": NC(mean=1.230066812934386, std=1.6003749533498033)},
    "mp_gap_fold4": {"y": NC(mean=1.2350543114618917, std=1.6035590734723943)},
    "mp_e_form_fold0": {"y": NC(mean=-1.4371594843879998, std=1.1577096884761835)},
    "mp_e_form_fold1": {"y": NC(mean=-1.4372781184639032, std=1.1576872656463288)},
    "mp_e_form_fold2": {"y": NC(mean=-1.4353308741245294, std=1.1568986659292604)},
    "mp_e_form_fold3": {"y": NC(mean=-1.4337824626302396, std=1.1570679204976484)},
    "mp_e_form_fold4": {"y": NC(mean=-1.437067044514929, std=1.1567267481888575)},
}


def matbench_configs(folds: Iterable[matbench.MatbenchFold]):
    for dataset, bsz, amp in MATBENCH_DATASETS:
        for fold in folds:
            config = matbench.MatbenchConfig.draft()
            config.dataset = dataset
            match dataset:
                case "phonons":
                    config.graph_scalar_reduction_default = "max"
                case "mp_is_metal":
                    config.graph_scalar_targets = []
                    config.graph_classification_targets = []
                    config.graph_classification_targets.append(
                        MulticlassClassificationTargetConfig(
                            name="y",
                            num_classes=2,
                            class_weights=[1.0, 1.34219693],
                            dropout=0.5,
                        )
                    )
                    config.node_vector_targets = []
                case _:
                    config.graph_scalar_reduction_default = "mean"

            config.fold = fold
            config.name = f"matbench_{dataset}_fold{fold}"
            config.mp_e_form_dev = False

            config.normalization = MATBENCH_NORMALIZATION.get(
                f"{dataset}_fold{fold}", {}
            )
            config.conditional_max_neighbors = True
            config_(config, batch=bsz, scalar=True)

            if dataset == "mp_is_metal":
                config.primary_metric = PrimaryMetricConfig(
                    name="y_balanced_accuracy", mode="max"
                )
            else:
                config.primary_metric = PrimaryMetricConfig(name="y_mae", mode="min")

            if amp:
                config.trainer.precision = "16-mixed"
            else:
                config.trainer.precision = "32-true"

            base_path = BASE_DATASET_PATH / "matbench"
            config.train_dataset = DC.matbench_config(dataset, base_path, "train", fold)
            config.val_dataset = DC.matbench_config(dataset, base_path, "val", fold)
            config.test_dataset = DC.matbench_config(dataset, base_path, "test", fold)

            yield config, matbench.MatbenchModel


def all_runs():
    yield from matbench_configs([0, 1, 2, 3, 4])
    yield from md17_configs()
    yield from md22_configs()
    yield from qm9_configs()
    yield from qmof_configs()
    yield from spice_configs()


configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
for base_config, model_cls in all_runs():
    config = copy.deepcopy(base_config)
    config.id = FinetuneConfigBase.generate_id()

    config.trainer.logging.wandb.log_model = False
    config = config.finalize()
    configs.append((config, model_cls))

for config, _ in configs:
    assert config.backbone.scale_file, f"Scale file not set for {config.name}"

print("\n".join([c.name for c, _ in configs]))
print(len(configs))


# %%
from jmp.lightning import Runner, Trainer
from jmp.modules.ema import EMA
from jmp.utils.finetune_state_dict import (
    filter_state_dict,
    retreive_state_dict_for_finetuning,
)


def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]):
    model = model_cls(config)

    if (ckpt_path := config.meta.get("ckpt_path")) is None:
        raise ValueError("ckpt_path must be provided")

    state_dict = retreive_state_dict_for_finetuning(
        ckpt_path,
        load_emas=config.meta.get("ema_backbone", False),
    )
    embedding = filter_state_dict(state_dict, "embedding.atom_embedding.")
    backbone = filter_state_dict(state_dict, "backbone.")

    model.load_backbone_state_dict(backbone=backbone, embedding=embedding, strict=False)

    callbacks = []
    if (ema := config.meta.get("ema")) is not None:
        ema = EMA(decay=ema)
        callbacks.append(ema)

    trainer = Trainer(config, callbacks=callbacks)
    trainer.fit(model)


# %%
# runner = Runner(run)
# runner.fast_dev_run(configs)

# %%
runner = Runner(run)
jobs = runner.submit(
    configs,
    gpus=1,
    nodes=1,
    cpus_per_task=configs[0][0].num_workers + 1,
    partition="ocp",
    constraint="volta32gb",
    snapshot=True,
)
