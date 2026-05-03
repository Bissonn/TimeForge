import math
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.train_loop import _train_one_epoch
from utils.pytorch_dataset import TimeSeriesForecastDataset, collate_forecast_batch


class TinyModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, dec=None):
        # x: (B, L, D) -> produce (B, L, out_dim)
        return self.lin(x)


class FakeGradScaler:
    """
    Minimal GradScaler-like object for deterministic unit tests.
    We simulate a skipped step by decreasing scale on update().
    """
    def __init__(self, start_scale: float = 1024.0, force_skip: bool = True):
        self._scale = float(start_scale)
        self.force_skip = force_skip

    def scale(self, loss):
        # return a scaled tensor-like: simplest is identity for our unit tests
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        # do nothing
        return None

    def update(self):
        if self.force_skip:
            # mimic overflow detection: scale decreases
            self._scale = max(self._scale / 2.0, 1.0)

    def get_scale(self):
        return self._scale


@pytest.mark.parametrize("device_str", ["cpu"])
def test_train_one_epoch_skips_on_nonfinite_loss(device_str):
    device = torch.device(device_str)
    torch.manual_seed(0)

    B, L, Din, Dout = 8, 4, 3, 2
    model = TinyModel(Din, Dout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction="mean")

    # Data on CPU (DataLoader will handle device transfer)
    x = torch.randn(B, L, Din)
    y = torch.randn(B, L, Dout)

    # Inject NaN into target to make loss non-finite deterministically
    y[0, 0, 0] = float("nan")

    # Create DataLoader (num_workers=0 default, no multiprocessing)
    dataset = TimeSeriesForecastDataset(x, y, None)
    loader = DataLoader(dataset, batch_size=B, shuffle=False, collate_fn=collate_forecast_batch)

    avg_loss, avg_grad, global_step, consecutive, skip_stats = _train_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=opt,
        loss_fn=loss_fn,
        device=device,
        epoch_idx=0,
        scaler=None,
        use_amp=False,
        autocast_dtype=torch.float16,
        max_grad_norm=1.0,
        gradient_monitor=None,
        scheduler=None,
        scheduler_step_frequency=None,
        global_step=0,
        consecutive_nonfinite=0,
    )

    assert global_step == 0
    assert consecutive == 1
    assert np.isfinite(avg_loss)  # avg_loss should be computed over 0 successful samples -> 0.0
    assert avg_loss == 0.0
    assert int(skip_stats["total_batches"]) == 1
    assert int(skip_stats["skipped_loss"]) == 1
    assert int(skip_stats["skipped_grad_norm"]) == 0
    assert int(skip_stats["skipped_scaler"]) == 0

@pytest.mark.parametrize("device_str", ["cpu"])
def test_train_one_epoch_skips_on_infinite_grad_norm(device_str):
    device = torch.device(device_str)
    torch.manual_seed(0)

    B, L, Din, Dout = 8, 4, 3, 2
    model = TinyModel(Din, Dout).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss(reduction="mean")

    # Data on CPU
    x = torch.randn(B, L, Din)
    y = torch.randn(B, L, Dout)

    # Create DataLoader (num_workers=0 default, no multiprocessing)
    dataset = TimeSeriesForecastDataset(x, y, None)
    loader = DataLoader(dataset, batch_size=B, shuffle=False, collate_fn=collate_forecast_batch)

    # Force infinite gradients via hook on weight
    def inf_grad_hook(grad):
        return torch.full_like(grad, float("inf"))

    hook_handle = model.lin.weight.register_hook(inf_grad_hook)

    try:
        avg_loss, avg_grad, global_step, consecutive, skip_stats = _train_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=opt,
            loss_fn=loss_fn,
            device=device,
            epoch_idx=0,
            scaler=None,
            use_amp=False,
            autocast_dtype=torch.float16,
            max_grad_norm=1.0,  # triggers clip + grad_norm check
            gradient_monitor=None,
            scheduler=None,
            scheduler_step_frequency=None,
            global_step=0,
            consecutive_nonfinite=0,
        )
    finally:
        hook_handle.remove()

    assert global_step == 0
    assert consecutive == 1
    assert avg_loss == 0.0
    assert avg_grad == 0.0
    assert int(skip_stats["total_batches"]) == 1
    assert int(skip_stats["skipped_loss"]) == 0
    assert int(skip_stats["skipped_grad_norm"]) == 1
    assert int(skip_stats["skipped_scaler"]) == 0

@pytest.mark.parametrize("device_str", ["cpu"])
def test_train_one_epoch_detects_gradscaler_skip(device_str):
    device = torch.device(device_str)
    torch.manual_seed(0)

    B, L, Din, Dout = 8, 4, 3, 2
    model = TinyModel(Din, Dout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction="mean")

    # Data on CPU
    x = torch.randn(B, L, Din)
    y = torch.randn(B, L, Dout)

    # Create DataLoader (num_workers=0 default, no multiprocessing)
    dataset = TimeSeriesForecastDataset(x, y, None)
    loader = DataLoader(dataset, batch_size=B, shuffle=False, collate_fn=collate_forecast_batch)

    scaler = FakeGradScaler(start_scale=1024.0, force_skip=True)

    avg_loss, avg_grad, global_step, consecutive, skip_stats = _train_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=opt,
        loss_fn=loss_fn,
        device=device,
        epoch_idx=0,
        scaler=scaler,          # fake scaler path
        use_amp=True,           # enabled, but CPU autocast doesn't matter here
        autocast_dtype=torch.bfloat16,
        max_grad_norm=1.0,
        gradient_monitor=None,
        scheduler=None,
        scheduler_step_frequency=None,
        global_step=0,
        consecutive_nonfinite=0,
    )

    # Since scaler scale decreases, step_applied=False and update should not be committed
    assert global_step == 0
    assert consecutive == 1
    assert avg_loss == 0.0
    assert int(skip_stats["total_batches"]) == 1
    assert int(skip_stats["skipped_loss"]) == 0
    assert int(skip_stats["skipped_grad_norm"]) == 0
    assert int(skip_stats["skipped_scaler"]) == 1

@pytest.mark.parametrize("device_str", ["cpu"])
def test_train_one_epoch_happy_path_updates_state(device_str):
    device = torch.device(device_str)
    torch.manual_seed(0)

    B, L, Din, Dout = 8, 4, 3, 2
    model = TinyModel(Din, Dout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction="mean")

    # Data on CPU
    x = torch.randn(B, L, Din)
    # Make targets correlated with x so loss is reasonable and gradients finite
    with torch.no_grad():
        y = model(x.to(device)).detach().cpu() + 0.01 * torch.randn(B, L, Dout)

    # Create DataLoader (num_workers=0 default, no multiprocessing)
    dataset = TimeSeriesForecastDataset(x, y, None)
    loader = DataLoader(dataset, batch_size=B, shuffle=False, collate_fn=collate_forecast_batch)

    avg_loss, avg_grad, global_step, consecutive, skip_stats = _train_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=opt,
        loss_fn=loss_fn,
        device=device,
        epoch_idx=0,
        scaler=None,
        use_amp=False,
        autocast_dtype=torch.float16,
        max_grad_norm=1.0,
        gradient_monitor=None,
        scheduler=None,
        scheduler_step_frequency=None,
        global_step=0,
        consecutive_nonfinite=0,
    )

    assert global_step == 1
    assert consecutive == 0
    assert np.isfinite(avg_loss) and avg_loss > 0.0
    assert np.isfinite(avg_grad)
    assert int(skip_stats["total_batches"]) == 1
    assert int(skip_stats["skipped_loss"]) == 0
    assert int(skip_stats["skipped_grad_norm"]) == 0
    assert int(skip_stats["skipped_scaler"]) == 0