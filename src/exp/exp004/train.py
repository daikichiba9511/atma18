import argparse
import multiprocessing as mp
import pathlib
from collections.abc import Callable, Mapping
from typing import Any

import albumentations as albu
import cv2
import numpy as np
import numpy.typing as npt
import polars as pl
import timm.utils as timm_utils
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch.amp import autocast_mode, grad_scaler
from tqdm.auto import tqdm
from typing_extensions import TypeAlias

import wandb
from src import constants, log, metrics, optim, train_tools, utils

from . import config, models

logger = log.get_root_logger()
EXP_NO = __file__.split("/")[-2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


LossFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def get_loss_fn(loss_name: str, loss_params: dict[str, Any]) -> LossFn:
    if loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(**loss_params)
    if loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(**loss_params)
    if loss_name == "MSELoss":
        return nn.MSELoss(**loss_params)
    if loss_name == "L1Loss":
        return nn.L1Loss(**loss_params)
    raise ValueError(f"Unknown loss name: {loss_name}")


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    ema_model: timm_utils.ModelEmaV3,
    optimizer: torch.optim.Optimizer,
    scheduler: optim.Schedulers,
    criterion: LossFn,
    loader: torch_data.DataLoader,
    device: torch.device,
    use_amp: bool,
    scaler: grad_scaler.GradScaler | None = None,
    max_norm: float = 1000.0,
) -> tuple[float, float]:
    """
    Args:
        epoch: number of epoch
        model: model to train
        ema_model: timm_utils.ModelEmaV3
        optimizer: torch.optim.Optimizer. I almost use AdamW.
        scheduler: optim.Schedulers. I almost use transformers.get_cosine_schedule_with_warmup
        criterion: LossFn. see get_loss_fn.
        loader: torch_data.DataLoader for training set
        device: torch.device
        use_amp: If True, use auto mixed precision. I use f16 as dtype.
        scaler: grad_scaler.GradScaler | None


    Returns:
        loss_meter.avg: float
        lr : float
    """
    lr = scheduler.get_last_lr()[0]
    model = model.train()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train", dynamic_ncols=True)
    loss_meter = train_tools.AverageMeter("train/loss")
    for batch_idx, batch in pbar:
        _key, x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_mode.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            output = model(x)
            y_pred = output.float()
            loss = criterion(y_pred, y)

        optimizer.zero_grad()
        if scaler is not None:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
        scheduler.step()
        ema_model.update(model)

        loss_meter.update(loss.detach().cpu().item())
        if batch_idx % 20 == 0:
            pbar.set_postfix_str(f"Loss:{loss_meter.avg:.4f},Epoch:{epoch},Norm:{grad_norm:.4f},LR:{lr:.4e}")

    return loss_meter.avg, lr


def valid_one_epoch(
    model: nn.Module,
    loader: torch_data.DataLoader,
    criterion: LossFn,
    device: torch.device,
) -> tuple[float, float, pl.DataFrame]:
    """
    Args:
        model: nn.Module
        loader: torch_data.DataLoader for validation set
        criterion: LossFn
        device: torch.device

    Returns: tuple
        loss: float
        score: float
        oof_df: pl.DataFrame
    """
    model = model.eval()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Valid", dynamic_ncols=True)
    loss_meter = train_tools.AverageMeter("valid/loss")
    oofs: list[pl.DataFrame] = []
    for batch_idx, batch in pbar:
        sample_id, x, y = batch
        x = x.to(device, non_blocking=True)
        with torch.inference_mode():
            output = model(x)

            y_pred = output.float()
            loss = criterion(y_pred.detach().cpu(), y)
            loss_meter.update(loss.item())

        y = y.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        oofs.append(
            pl.DataFrame({
                "sample_id": sample_id,
                **{f"{c}": y[:, i] for i, c in enumerate(constants.TARGET_COLS)},
                **{f"pred-{c}": y_pred[:, i] for i, c in enumerate(constants.TARGET_COLS)},
            })
        )
        if batch_idx % 20 == 0:
            pbar.set_postfix_str(f"Loss:{loss_meter.avg:.4f}")

    oof = pl.concat(oofs, how="vertical")
    valid_score = metrics.score(
        y_true=oof[constants.TARGET_COLS].to_numpy(),
        y_pred=oof[[f"pred-{c}" for c in constants.TARGET_COLS]].to_numpy(),
    )
    return loss_meter.avg, valid_score, oof


# =============================================================================
# Dataset
# =============================================================================
TrainBatch: TypeAlias = tuple[str, torch.Tensor, torch.Tensor]
ValidBatch: TypeAlias = tuple[str, torch.Tensor, torch.Tensor]

AlbuTransforms: TypeAlias = list[albu.BasicTransform | albu.OneOf | albu.BaseCompose]


def as_compose(composed_fns: AlbuTransforms) -> albu.ReplayCompose:
    return albu.ReplayCompose(composed_fns)


def read_multiframe_image(
    base_path: pathlib.Path, image_caches: Mapping[str, npt.NDArray] | None
) -> list[npt.NDArray]:
    fps = [str(base_path / "image_t-1.0.png"), str(base_path / "image_t-0.5.png"), str(base_path / "image_t.png")]
    images = []
    for fp in fps:
        if image_caches is not None and fp in image_caches:
            image = image_caches[fp]
        else:
            image = cv2.imread(fp)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images


class MyTrainDataset(torch_data.Dataset[TrainBatch]):
    def __init__(
        self,
        df: pl.DataFrame,
        label: npt.NDArray,
        transforms: AlbuTransforms | None = None,
        video_caches: Mapping[str, npt.NDArray] | None = None,
    ) -> None:
        super().__init__()
        self.df = df.to_pandas(use_pyarrow_extension_array=True)
        self.transform_fn = as_compose(transforms) if transforms is not None else None
        self.label = label
        self.video_caches = video_caches

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> TrainBatch:
        base_path = pathlib.Path(self.df["base_path"][idx])
        # list: (3, H, W, C)
        image = read_multiframe_image(base_path, self.video_caches)
        if self.transform_fn is not None:
            replay: dict | None = None
            images = []
            for img in image:
                if replay is None:
                    sample = self.transform_fn(image=img)
                    replay = sample["replay"]
                    images.append(sample["image"])
                else:
                    sample = albu.ReplayCompose.replay(replay, image=img)
                    images.append(sample["image"])
            imgs = torch.concat(images, dim=0)
        else:
            # shape: (3, H, W, C) -> (3, C, H, W)
            imgs = torch.concat([torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) for img in image], dim=0)

        label = torch.tensor(self.label[idx], dtype=torch.float32)
        return str(base_path), imgs, label


class MyValidDataset(torch_data.Dataset[ValidBatch]):
    def __init__(
        self,
        df: pl.DataFrame,
        label: npt.NDArray,
        transforms: AlbuTransforms | None = None,
        video_caches: Mapping[str, npt.NDArray] | None = None,
    ) -> None:
        super().__init__()
        self.df = df
        self.transform_fn = as_compose(transforms) if transforms is not None else None
        self.label = label
        self.video_caches = video_caches

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> TrainBatch:
        base_path = pathlib.Path(self.df["base_path"][idx])
        image = read_multiframe_image(base_path, self.video_caches)
        if self.transform_fn is not None:
            replay: dict | None = None
            images = []
            for img in image:
                if replay is None:
                    sample = self.transform_fn(image=img)
                    replay = sample["replay"]
                    images.append(sample["image"])
                else:
                    sample = albu.ReplayCompose.replay(replay, image=img)
                    images.append(sample["image"])
            imgs = torch.concat(images, dim=0)
        else:
            imgs = torch.concat([torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) for img in image], dim=0)

        label = torch.tensor(self.label[idx], dtype=torch.float32)
        return str(base_path), imgs, label


def read_image(fp: str) -> tuple[str, npt.NDArray]:
    if not pathlib.Path(fp).exists():
        raise FileNotFoundError(f"{fp} is not found.")
    img = cv2.imread(fp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return fp, img


def init_dataloader(
    df_fp: pathlib.Path,
    train_batch_size: int,
    valid_batch_size: int,
    num_workers: int = 16,
    fold: int = 0,
    train_transforms: AlbuTransforms | None = None,
    valid_transforms: AlbuTransforms | None = None,
    debug: bool = False,
) -> tuple[torch_data.DataLoader, torch_data.DataLoader]:
    if mp.cpu_count() < num_workers:
        num_workers = mp.cpu_count()

    print(f"{df_fp=}, {fold=}")
    if df_fp.suffix == ".csv":
        df = pl.read_csv(df_fp)
    else:
        df = pl.read_parquet(df_fp)

    df = df.with_columns(base_path=str(constants.DATA_DIR) + "/images/" + df["ID"].cast(pl.Utf8) + "/")

    df_train = df.filter(pl.col("fold") != fold)
    df_valid = df.filter(pl.col("fold") == fold)
    assert len(df_train) > 0, f"df_train is empty: {df_fp=}, {fold=}"
    assert len(df_valid) > 0, f"df_valid is empty: {df_fp=}, {fold=}"
    if debug:
        df_train = df_train.head(200)
        df_valid = df_valid.head(200)

    label_train = df_train[constants.TARGET_COLS].to_numpy()
    label_valid = df_valid[constants.TARGET_COLS].to_numpy()

    cache_paths = []
    for base_path in df["base_path"].to_list():
        cache_paths.extend([
            f"{base_path}/image_t-1.0.png",
            f"{base_path}/image_t-0.5.png",
            f"{base_path}/image_t.png",
        ])
    video_cached = dict(utils.call_mp_unordered(read_image, cache_paths, with_progress=True))

    train_ds: torch_data.Dataset[TrainBatch] = MyTrainDataset(
        df_train, label=label_train, video_caches=video_cached, transforms=train_transforms
    )
    valid_ds: torch_data.Dataset[ValidBatch] = MyValidDataset(
        df_valid, label=label_valid, video_caches=video_cached, transforms=valid_transforms
    )

    train_dl = torch_data.DataLoader(
        dataset=train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=lambda _: utils.seed_everything(42),
        persistent_workers=True if num_workers > 0 else False,
    )

    valid_loader = torch_data.DataLoader(
        dataset=valid_ds,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=lambda _: utils.seed_everything(42),
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_dl, valid_loader


def _test_init_dataloader() -> None:
    train_dl, valid_dl = init_dataloader(
        constants.INPUT_DIR / "train_folds.parquet",
        train_batch_size=8,
        valid_batch_size=8,
        num_workers=0,
        fold=0,
        train_transforms=None,
        valid_transforms=None,
    )
    for batch in train_dl:
        key, x, y = batch
        print(key, x.shape, y.shape)
        break
    for batch in valid_dl:
        key, x, y = batch
        print(key, x.shape, y.shape)
        break


def main() -> None:
    args = parse_args()
    if args.debug:
        cfg = config.Config(is_debug=True)
    else:
        cfg = config.Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    utils.pinfo(cfg.model_dump())
    log.attach_file_handler(logger, str(cfg.output_dir / "train.log"))
    logger.info(f"Exp: {cfg.name}, DESC: {cfg.description}, COMMIT_HASH: {utils.get_commit_hash_head()}")
    # =============================================================================
    # TrainLoop
    # =============================================================================
    for fold in range(cfg.n_folds):
        logger.info(f"Start fold: {fold}")
        utils.seed_everything(cfg.seed + fold)
        if cfg.is_debug:
            run = None
        else:
            run = wandb.init(
                project=constants.COMPE_NAME,
                name=f"{cfg.name}_{fold}",
                config=cfg.model_dump(),
                reinit=True,
                group=f"{fold}",
                dir="./src",
            )
        model, ema_model = models.get_model(cfg.model_name, cfg.model_params)
        if args.compile:
            model, ema_model = models.compile_models(model, ema_model)
        model, ema_model = model.to(cfg.device), ema_model.to(cfg.device)
        train_loader, valid_loader = init_dataloader(
            cfg.train_data_fp,
            cfg.train_batch_size,
            cfg.valid_batch_size,
            cfg.num_workers,
            fold,
            debug=cfg.is_debug,
        )
        optimizer = optim.get_optimizer(cfg.train_optimizer_name, cfg.train_optimizer_params, model=model)
        if cfg.train_scheduler_params.get("num_training_steps") == -1:
            scheduler_params = optim.setup_scheduler_params(
                cfg.train_scheduler_params, num_step_per_epoch=len(train_loader), n_epoch=cfg.train_n_epochs
            )
        else:
            scheduler_params = cfg.train_scheduler_params
        scheduler = optim.get_scheduler(cfg.train_scheduler_name, scheduler_params, optimizer=optimizer)
        criterion = get_loss_fn(cfg.train_loss_name, cfg.train_loss_params)
        metric_monitor = train_tools.MetricsMonitor(metrics=["epoch", "train/loss", "lr", "valid/loss", "valid/score"])
        best_score, best_oof = np.inf, pl.DataFrame()
        for epoch in range(cfg.train_n_epochs):
            train_loss_avg, lr = train_one_epoch(
                epoch=epoch,
                model=model,
                ema_model=ema_model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                device=cfg.device,
                use_amp=cfg.train_use_amp,
            )
            valid_loss_avg, valid_score, valid_oof = valid_one_epoch(
                model=ema_model.module, loader=valid_loader, criterion=criterion, device=cfg.device
            )
            if valid_score < best_score:
                best_oof = valid_oof
                best_score = valid_score
            metric_map = {
                "epoch": epoch,
                "train/loss": train_loss_avg,
                "lr": lr,
                "valid/loss": valid_loss_avg,
                "valid/score": valid_score,
            }
            metric_monitor.update(metric_map)
            if epoch % cfg.train_log_interval == 0:
                metric_monitor.show()
            if run:
                wandb.log(metric_map)

        # -- Save Results
        scores = {}
        for col in constants.TARGET_COLS:
            scores[col] = metrics.score(y_true=best_oof[col].to_numpy(), y_pred=best_oof[f"pred-{col}"].to_numpy())
        utils.pinfo(scores)
        logger.info(f"Best Score: {best_score}")
        best_oof.write_parquet(cfg.output_dir / f"oof_{fold}.parquet")
        metric_monitor.save(cfg.output_dir / f"metrics_{fold}.csv", fold=fold)
        model_state = train_tools.get_model_state_dict(ema_model.module)
        save_fp_model = cfg.output_dir / f"last_model_{fold}.pth"
        torch.save(model_state, save_fp_model)
        logger.info(f"Saved model to {save_fp_model}")

        if run is not None:
            run.finish()

        if cfg.is_debug:
            break
    logger.info("End of Training")


if __name__ == "__main__":
    # _test_init_dataloader()
    main()
