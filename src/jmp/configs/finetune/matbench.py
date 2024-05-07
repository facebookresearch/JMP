"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import MatbenchConfig
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import PrimaryMetricConfig

STATS: dict[str, dict[str, NC]] = {
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


def jmp_l_matbench_config_(
    config: MatbenchConfig,
    dataset: DC.MatbenchDataset,
    fold: DC.MatbenchFold,
    base_path: Path,
):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Set up dataset
    config.train_dataset = DC.matbench_config(dataset, base_path, "train", fold)
    config.val_dataset = DC.matbench_config(dataset, base_path, "val", fold)
    config.test_dataset = DC.matbench_config(dataset, base_path, "test", fold)

    # Set up normalization
    if (normalization_config := STATS.get(f"{dataset}_{fold}")) is None:
        raise ValueError(f"Normalization for {dataset}_{fold} not found")
    config.normalization = normalization_config

    # MatBench specific settings
    config.dataset = dataset
    config.primary_metric = PrimaryMetricConfig(name="y_mae", mode="min")
