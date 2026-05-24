
import copy
import logging
import os
import random
import shutil
import tempfile
import warnings
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import math
import calendar

import gymnasium as gym
import numpy as np
import sinergym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl
import tqdm as tqdm_pkg
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing as mp
from torch.nn.utils import clip_grad_norm_
from torchrl.collectors import SyncDataCollector
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    InitTracker,
    ParallelEnv,
    SerialEnv,
    StepCounter,
    TransformedEnv,
    default_info_dict_reader,
)
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.envs.utils import check_env_specs
from torchrl.modules import LSTMModule, TanhNormal, set_recurrent_mode

warnings.filterwarnings("ignore")

_TQDM_PATCHED = False


def tensor_corr_line(x: torch.Tensor, y: torch.Tensor, name: str) -> str:
    corr = tensor_corrcoef(x.reshape(-1), y.reshape(-1))
    return f"{name}: corr={float(corr.detach().cpu()):.6f}"


def tensor_stats_line(x: torch.Tensor, name: str) -> str:
    x = x.float().reshape(-1)
    pos_frac = (x > 0).float().mean().item()
    neg_frac = (x < 0).float().mean().item()
    zero_frac = (x == 0).float().mean().item()
    return (
        f"{name}: μ={x.mean().item():.6f} | σ={x.std(unbiased=False).item():.6f} | "
        f"min={x.min().item():.6f} | max={x.max().item():.6f} | "
        f"+={pos_frac:.2%} | -={neg_frac:.2%} | 0={zero_frac:.2%}"
    )


def _patch_tqdm_disable():
    global _TQDM_PATCHED
    if _TQDM_PATCHED:
        return
    try:
        import tqdm as _tqdm_pkg

        _orig_tqdm = _tqdm_pkg.tqdm

        def _quiet_tqdm(*args, **kwargs):
            kwargs.setdefault("disable", True)
            return _orig_tqdm(*args, **kwargs)

        _tqdm_pkg.tqdm = _quiet_tqdm
        try:
            import tqdm.auto as _tqdm_auto

            _tqdm_auto.tqdm = _quiet_tqdm
        except Exception:
            pass
        _TQDM_PATCHED = True
    except Exception:
        pass


@contextmanager
def suppress_stdout_stderr(enabled: bool = True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def _safe_remove_path(path: Path):
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)
    except Exception:
        pass


def silence_all_library_logging():
    logging.disable(logging.CRITICAL)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.CRITICAL)
    for logger_name in list(logging.root.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)
        logger.addHandler(logging.NullHandler())


class NoOpPbar:
    def set_description(self, *args, **kwargs):
        pass

    def write(self, msg):
        print(msg, flush=True)

    def close(self):
        pass


class Config:
    def __init__(self):
        # Persistent uCloud project folder
        self.thesis_dir = "/work/AbaAntal#6776/Thesis"

        # Save policies/checkpoints here, NOT in /tmp
        self.save_dir = os.path.join(self.thesis_dir, "checkpoints")

        self.run_name = "sinergym_lstm_transformer_switching_dynamics_openloop_clean"

        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sinergym_env_id = "Eplus-5zone-cool-continuous-v1"
        self.season = "winter"
        self.season_year = 2000
        self.season_runperiod_days = 7
        self.timesteps_per_hour = 4
        self.sinergym_building_config = {}
        self.sinergym_reward_kwargs = {
            "temperature_variables": ["air_temperature"],
            "energy_variables": ["HVAC_electricity_demand_rate"],
            "range_comfort_winter": (20.0, 23.5),
            "range_comfort_summer": (23.0, 26.0),
            "summer_start": (6, 1),
            "summer_final": (9, 30),
            "energy_weight": 0.5,
            "lambda_energy": 1.0e-4,
            "lambda_temperature": 1.0,
        }
        self.sinergym_observation_keys = [
            "month",
            "day_of_month",
            "hour",
            "outdoor_temperature",
            "outdoor_humidity",
            "wind_speed",
            "wind_direction",
            "diffuse_solar_radiation",
            "direct_solar_radiation",
            "air_temperature",
            "air_humidity",
            "people_occupant",
            "heating_setpoint",
            "cooling_setpoint",
            "co2_emission",
            "HVAC_electricity_demand_rate",
            "total_electricity_HVAC",
        ]
        self.model_observation_keys = [
            "month",
            "day_of_month",
            "hour",
            "outdoor_temperature",
            "outdoor_humidity",
            "diffuse_solar_radiation",
            "direct_solar_radiation",
            "air_temperature",
            "air_humidity",
            "people_occupant",
            "heating_setpoint",
            "cooling_setpoint",
            "HVAC_electricity_demand_rate",
        ]
        self.sinergym_weather_file = None
        self.sinergym_weather_variability = None

        self.sinergym_randomize_runperiod = True
        self.sinergym_randomize_runperiod_on_reset = True
        self.sinergym_window_candidate_months = [10, 11, 12, 1, 2, 3]
        self.sinergym_window_keep_within_month = True

        # Keep randomized EnergyPlus/Sinergym episodes, but prevent output-folder spam
        # in the project directory. Each worker writes into its own sandbox under /tmp
        # and old EnergyPlus output is removed after each environment rebuild.
        self.sinergym_isolate_output_dirs = True
        self.sinergym_output_root = os.path.join(tempfile.gettempdir(), "sinergym_energyplus_outputs")
        self.sinergym_cleanup_output_on_rebuild = True
        self.sinergym_cleanup_output_on_close = True
        self.sinergym_max_ep_store = 1

        self.sinergym_temperature_variable = "air_temperature"
        self.sinergym_power_variable = "HVAC_electricity_demand_rate"
        self.sinergym_heating_setpoint_variable = "heating_setpoint"
        self.sinergym_cooling_setpoint_variable = "cooling_setpoint"

        self.num_envs = 80
        self.use_parallel_env = True
        self.mp_start_method = "spawn"
        self.check_env_specs_on_single_env = False
        self.collector_storing_on_gpu = True

        self.sinergym_quiet = True
        self.sinergym_log_level = "CRITICAL"
        self.sinergym_disable_progress_bar = True
        self.suppress_torchrl_logs = True
        self.suppress_env_stdout_stderr = True
        self.show_startup_prints = False

        self.observation_norm_clip = 10.0

        self.actor_num_cells = 256
        self.actor_lstm_hidden = 256
        self.actor_window_size = 8
        self.actor_xfm_nhead = 4
        self.actor_xfm_layers = 2
        self.actor_xfm_ff_dim = 512
        self.actor_xfm_dropout = 0.0

        self.dyn_state_dim = 32
        self.dyn_num_regimes = 6
        self.dyn_hidden = 128

        self.dyn_loss_coef = 0.0005
        self.latent_dyn1_coef = 0.10
        self.latent_roll_coef = 0.05
        self.dyn_roll_horizon = 10
        self.dyn_rollout_ramp_batches = 60
        self.dyn_metric_min_scale = 0.10
        self.dyn_metric_huber_delta = 1.0

        self.temp_residual_dyn1_coef = 6.0
        self.temp_residual_roll_coef = 14.0
        # Match the refrigeration training logic: keep the re-anchored rollout and
        # rollout-bias auxiliary branches active instead of leaving them as dead code.
        self.temp_reanchor_roll_coef = 6.0
        self.temp_roll_bias_coef = 3.0

        self.viol_dyn1_coef = 10.0
        self.viol_roll_coef = 12.0
        self.viol_tail_coef = 3.0

        self.temp_persist_dyn1_coef = 12.0
        self.temp_persist_roll_coef = 24.0
        self.viol_persist_dyn1_coef = 8.0
        self.viol_persist_roll_coef = 16.0

        self.temp_gain_tail_coef = 6.0
        self.temp_gain_consistency_coef = 3.0
        self.temp_persist_margin = 0.0
        self.temp_residual_scale = 0.50
        self.violation_loss_scale = 0.25
        self.train_reanchor_prob = 0.5
        self.train_reanchor_gap = 3
        # Match the refrigeration script: use separate gates for temperature supervision
        # and latent/projector dynamics. Both are enabled from the first batch by default.
        # Set either value above zero if you want a delayed warm-up.
        self.temp_supervision_enable_after_batch = 0
        self.backbone_dynamics_enable_after_batch = 0
        # Legacy compatibility only. Leave this as None so the two explicit gates above
        # control the actual training logic.
        self.dynamics_aux_enable_after_batch = None

        self.shield_enable = False
        self.shield_apply_after_batch = 15
        self.shield_horizon = 5
        self.shield_opt_steps = 4
        self.shield_opt_lr = 0.25
        self.shield_opt_tol = 1e-6
        self.shield_opt_grad_clip = 1.0
        self.shield_detach_action_for_logprob = True
        self.shield_violation_coef = 10.0
        self.shield_violation_max_coef = 1.0
        self.shield_deviation_coef = 0.1
        self.shield_trigger_violation = 1.0
        self.shield_only_if_triggered = True
        self.shield_commit_max_steps = 1
        self.shield_frontload_decay = 0.65
        self.shield_frontload_min_weight = 0.25
        self.shield_early_horizon = 4
        self.shield_early_violation_coef = 2.0
        self.shield_early_violation_max_coef = 2.0

        self.total_frames = 6_000_000
        self.chunk_T = self.season_runperiod_days * 24 * self.timesteps_per_hour
        self.frames_per_batch = 0
        self.ppo_epochs = 10

        self.actor_lr = 5e-5
        self.reward_critic_lr = 5e-5
        self.cost_critic_lr = 5e-5
        self.backbone_dyn_lr = 5e-5
        self.projector_lr = 5e-5
        self.dynamics_lr = 5e-5

        self.cuda_empty_cache_every_minibatches = 16
        self.empty_cache_after_checkpoint = True

        self.reward_gamma = 0.99
        self.reward_lmbda = 0.98
        self.cost_gamma = 0.99
        self.cost_lmbda = 0.98

        self.clip_epsilon = 0.1
        self.entropy_coef = 1e-4
        self.reward_value_coef = 1.0
        self.cost_value_coef = 2.0
        self.max_grad_norm = 1

        self.smoothness_coef = 0.1
        self.smoothness_mode = "l2"
        self.smoothness_normalize_by_action_range = True

        self.lambda_signal_type = "binary"
        self.lambda_init = 5.0
        self.lambda_lr = 1.2
        self.lambda_max = 30.0
        self.lambda_min = 0.3
        self.cost_budget = 0.015
        self.lambda_warmup_batches = 0

        self.log_every_batches = 1
        self.save_every_batches = 1 
        self.eps = 1e-8

        self.reward_value_target_scale = 100.0
        self.reward_value_target_shift = 0.0
        self.cost_value_target_scale = 5.0
        self.cost_value_target_shift = 0.0

        self.temperature_scale = 30.0
        self.latent_slot_residual_scale = 0.01

        self.dynamics_exog_keys = [
            "month",
            "day_of_month",
            "hour",
            "outdoor_temperature",
            "outdoor_humidity",
            "diffuse_solar_radiation",
            "direct_solar_radiation",
            "people_occupant",
            "heating_setpoint",
            "cooling_setpoint",
        ]
        self.dynamics_exog_normalize = True
        self.forecast_key_map = {}
        self.model_observation_indices = []
        self.dynamics_exog_indices = []
        self.dynamics_exog_dim = 0
        self.forecast_index_map = {}

        if int(self.frames_per_batch) <= 0:
            self.frames_per_batch = int(self.chunk_T * self.num_envs)

    def to_dict(self):
        return copy.deepcopy(self.__dict__)


def violation_mag_from_temp(temp: torch.Tensor, comfort_low: float, comfort_high: float):
    low_t = torch.as_tensor(float(comfort_low), device=temp.device, dtype=temp.dtype)
    high_t = torch.as_tensor(float(comfort_high), device=temp.device, dtype=temp.dtype)
    return torch.maximum(
        (low_t - temp).clamp_min(0.0),
        (temp - high_t).clamp_min(0.0),
    )


def normalize_value_target(x: torch.Tensor, scale: float, shift: float = 0.0) -> torch.Tensor:
    scale_t = torch.as_tensor(float(scale), device=x.device, dtype=x.dtype).clamp_min(1e-8)
    shift_t = torch.as_tensor(float(shift), device=x.device, dtype=x.dtype)
    y = (x - shift_t) / scale_t
    require_finite(y, "normalized value target")
    return y


def unnormalize_value_prediction(x: torch.Tensor, scale: float, shift: float = 0.0) -> torch.Tensor:
    scale_t = torch.as_tensor(float(scale), device=x.device, dtype=x.dtype)
    shift_t = torch.as_tensor(float(shift), device=x.device, dtype=x.dtype)
    y = x * scale_t + shift_t
    require_finite(y, "unnormalized value prediction")
    return y


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_finite(x: torch.Tensor, name: str):
    if x is None:
        raise RuntimeError(f"[NON-FINITE CHECK] {name} is None")
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        raise RuntimeError(f"[NON-FINITE] {name}: found {bad} non-finite entries; shape={tuple(x.shape)}")


def _has_bad(x: torch.Tensor) -> bool:
    return bool(torch.isnan(x).any().item() or torch.isinf(x).any().item())


def safe_key(td, key):
    try:
        return td.get(key, None)
    except Exception:
        return None


def get_time_dim(td) -> int:
    names = getattr(td, "names", None)
    if names is not None and "time" in names:
        return names.index("time")
    return td.ndim - 1


def to_TB(x, time_dim: int):
    if x is None:
        return None
    if time_dim != x.ndim - 1:
        x = x.movedim(time_dim, -1)
    return x.movedim(-1, 0)


def _move_time_last(x: torch.Tensor, time_dim: int) -> torch.Tensor:
    if time_dim == x.ndim - 1:
        return x
    return x.movedim(time_dim, -1)


def move_time_to_dim1(x: torch.Tensor, time_dim: int) -> torch.Tensor:
    if time_dim == 1:
        return x
    return x.movedim(time_dim, 1)


def check_batch_for_nans(td, keys, label=""):
    bad = []
    for k in keys:
        if k in td.keys(True, True):
            v = td.get(k)
            if isinstance(v, torch.Tensor) and _has_bad(v):
                bad.append(k)
    if bad:
        print(f"\n[NaN/Inf DETECTED] {label} bad keys: {bad}")
    return bad


def mean_of(td, key_name: str, time_dim: int):
    x = safe_key(td, ("next", key_name))
    if x is None:
        return None
    x = to_TB(x, time_dim=time_dim).float()
    if x.ndim > 2:
        x = x.squeeze(-1)
    return x.mean().item()


def mean_std_max(td, key_name: str, time_dim: int):
    x = safe_key(td, ("next", key_name))
    if x is None:
        return None
    x = to_TB(x, time_dim=time_dim).float()
    if x.ndim > 2:
        x = x.squeeze(-1)
    return x.mean().item(), x.std(unbiased=False).item(), x.max().item(), x.min().item()


def rollout_summary_line(x_tb: torch.Tensor, name: str):
    x = x_tb.float()
    if x.ndim > 2:
        x = x.squeeze(-1)
    rollout_mean = x.mean(dim=0)
    rollout_std = x.std(dim=0, unbiased=False)
    rollout_max = x.max(dim=0).values
    return (
        f"{name}: rollout_mean μ={rollout_mean.mean().item():.6f} σ={rollout_mean.std(unbiased=False).item():.6f} | "
        f"rollout_std μ={rollout_std.mean().item():.6f} | rollout_max μ={rollout_max.mean().item():.6f}"
    )


def action_stats_by_dim(x: torch.Tensor, name: str = "action"):
    if x is None:
        return f"{name}: NA"
    x = x.float()
    act_dim = x.shape[-1]
    flat = x.reshape(-1, act_dim)
    parts = [f"{name}:"]
    for i in range(act_dim):
        xi = flat[:, i]
        parts.append(
            f"a{i}[μ={xi.mean().item():.4f} σ={xi.std(unbiased=False).item():.4f} min={xi.min().item():.4f} max={xi.max().item():.4f}]"
        )
    return " | ".join(parts)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    var_y = torch.var(y_true)
    if var_y.item() < 1e-8:
        return float("nan")
    return float((1.0 - torch.var(y_true - y_pred) / var_y).item())


def normalize_advantage_(td, adv_key="advantage", eps=1e-8):
    adv = td[adv_key]
    require_finite(adv, adv_key)
    adv_flat = adv.reshape(-1)
    mean = adv_flat.mean()
    std = adv_flat.std(unbiased=False)
    td[adv_key] = (adv - mean) / (std + eps) if torch.isfinite(std) and std.item() > 0.0 else adv - mean
    require_finite(td[adv_key], f"{adv_key} (normalized)")


def contiguous_minibatches(td, chunk_T: int, time_dim: int):
    T = int(td.shape[time_dim])
    for start in range(0, T, chunk_T):
        sl = slice(start, min(start + chunk_T, T))
        index = [slice(None)] * td.ndim
        index[time_dim] = sl
        yield td[tuple(index)], start


def get_done_TB(td, time_dim: int):
    if ("next", "done") in td.keys(True, True):
        d = td["next", "done"]
    else:
        term = td.get(("next", "terminated"), None)
        trunc = td.get(("next", "truncated"), None)
        if term is None and trunc is None:
            d = torch.zeros_like(td["next", "reward"], dtype=torch.bool)
        else:
            if term is None:
                term = torch.zeros_like(trunc, dtype=torch.bool)
            if trunc is None:
                trunc = torch.zeros_like(term, dtype=torch.bool)
            d = term | trunc
    d = to_TB(d, time_dim=time_dim)
    if d.ndim > 2:
        d = d.squeeze(-1)
    return d.bool()


def mean_done_rate(td, time_dim: int):
    return get_done_TB(td, time_dim=time_dim).float().mean().item()


def update_episode_length_stats(td, *, time_dim: int, running_steps_np: np.ndarray, term_counter: Counter):
    done_tb = get_done_TB(td, time_dim=time_dim)
    done_np = done_tb.detach().cpu().numpy().astype(np.bool_)
    term_code = safe_key(td, ("next", "term_code"))
    term_code_np = None
    if term_code is not None:
        tc_tb = to_TB(term_code, time_dim=time_dim).detach().cpu().numpy()
        if tc_tb.ndim > 2:
            tc_tb = tc_tb.squeeze(-1)
        term_code_np = tc_tb
    T, B = done_np.shape
    ended_lengths = []
    for t in range(T):
        for b in range(B):
            running_steps_np[b] += 1
            if done_np[t, b]:
                ended_lengths.append(int(running_steps_np[b]))
                running_steps_np[b] = 0
                if term_code_np is not None:
                    try:
                        term_counter[int(term_code_np[t, b])] += 1
                    except Exception:
                        pass
    return ended_lengths, float(np.mean(running_steps_np))


def _find_time_axis(x: torch.Tensor, T: int):
    axes = [i for i, d in enumerate(x.shape) if int(d) == T]
    return axes[0] if axes else None


def strip_recurrent_to_t0(td, keys, time_dim: int):
    T = int(td.shape[time_dim])
    for k in keys:
        if k not in td.keys(True, True):
            continue
        x = td.get(k)
        if not isinstance(x, torch.Tensor):
            continue
        t_ax = _find_time_axis(x, T)
        if t_ax is None:
            continue
        x_new = torch.zeros_like(x)
        sl = [slice(None)] * x.ndim
        sl[t_ax] = 0
        x0 = x[tuple(sl)].contiguous()
        sl_new = [slice(None)] * x_new.ndim
        sl_new[t_ax] = 0
        x_new[tuple(sl_new)] = x0
        td.set(k, x_new)
    return td


def default_runperiod_for_season(cfg: Config) -> tuple[int, int, int, int, int, int]:
    season = str(cfg.season).lower()
    year = int(cfg.season_year)
    days = max(1, int(cfg.season_runperiod_days))
    if season == "summer":
        start_month, start_day = 7, 1
    elif season == "winter":
        start_month, start_day = 1, 1
    else:
        raise ValueError(f"Unsupported season={cfg.season!r}; expected 'winter' or 'summer'.")
    end_day = start_day + days - 1
    return (start_day, start_month, year, end_day, start_month, year)


def sample_moving_window_runperiod(cfg: Config, rng: random.Random | None = None) -> tuple[int, int, int, int, int, int]:
    if rng is None:
        rng = random

    year = int(cfg.season_year)
    days = max(1, int(cfg.season_runperiod_days))
    months = [int(m) for m in getattr(cfg, "sinergym_window_candidate_months", [12, 1, 2])]
    keep_within_month = bool(getattr(cfg, "sinergym_window_keep_within_month", True))

    if len(months) == 0:
        raise ValueError("sinergym_window_candidate_months must contain at least one month")

    candidates: list[tuple[int, int, int, int, int, int]] = []
    for month in months:
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month in sinergym_window_candidate_months: {month}")
        month_days = int(calendar.monthrange(year, month)[1])

        if keep_within_month:
            max_start_day = month_days - days + 1
            if max_start_day < 1:
                continue
            for start_day in range(1, max_start_day + 1):
                end_day = start_day + days - 1
                candidates.append((start_day, month, year, end_day, month, year))
        else:
            raise NotImplementedError(
                "sinergym_window_keep_within_month=False is not implemented in this training script."
            )

    if len(candidates) == 0:
        raise ValueError(
            f"No valid moving-window runperiod candidates for year={year}, days={days}, months={months}"
        )

    return candidates[rng.randrange(len(candidates))]


def describe_runperiod_sampling(cfg: Config) -> str:
    if bool(getattr(cfg, "sinergym_randomize_runperiod", False)):
        months = [int(m) for m in getattr(cfg, "sinergym_window_candidate_months", [12, 1, 2])]
        keep_within_month = bool(getattr(cfg, "sinergym_window_keep_within_month", True))
        return (
            f"random_{int(cfg.season_runperiod_days)}d_windows"
            f"_months={months}"
            f"_within_month={keep_within_month}"
        )
    return f"fixed_runperiod={cfg.sinergym_building_config.get('runperiod', default_runperiod_for_season(cfg))}"


def configure_sinergym_verbosity(cfg: Config):
    if bool(cfg.sinergym_disable_progress_bar):
        os.environ["TQDM_DISABLE"] = "1"
        _patch_tqdm_disable()

    if bool(cfg.suppress_torchrl_logs):
        silence_all_library_logging()

    if not bool(cfg.sinergym_quiet):
        return

    level_name = str(cfg.sinergym_log_level).upper()
    level = getattr(logging, level_name, logging.CRITICAL)
    for logger_name in ["ENVIRONMENT", "MODEL", "SIMULATOR", "REWARD", "sinergym"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = False
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())


def active_comfort_band_from_cfg(cfg: Config, month: int | None = None, day_of_month: int | None = None) -> tuple[float, float]:
    rk = dict(cfg.sinergym_reward_kwargs)
    winter = tuple(rk.get("range_comfort_winter", (20.0, 23.5)))
    summer = tuple(rk.get("range_comfort_summer", (23.0, 26.0)))
    summer_start = tuple(rk.get("summer_start", (6, 1)))
    summer_final = tuple(rk.get("summer_final", (9, 30)))
    if month is None or day_of_month is None:
        return summer if str(cfg.season).lower() == "summer" else winter
    cur = (int(month), int(day_of_month))
    return summer if summer_start <= cur <= summer_final else winter


def obs_index_map(keys: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(keys)}


def obs_index_map_from_cfg(cfg: Config) -> dict[str, int]:
    return obs_index_map(cfg.sinergym_observation_keys)


def require_obs_index_in_keys(keys: list[str], key: str, *, label: str) -> int:
    mapping = obs_index_map(keys)
    if key not in mapping:
        raise KeyError(f"Observation key {key!r} not in {label}")
    return int(mapping[key])


def resolve_model_observation_indices(cfg: Config) -> list[int]:
    full_map = obs_index_map_from_cfg(cfg)
    missing = [key for key in cfg.model_observation_keys if key not in full_map]
    if missing:
        raise KeyError(f"model_observation_keys contains keys not present in sinergym_observation_keys: {missing}")
    return [int(full_map[key]) for key in cfg.model_observation_keys]


def default_obs_affine_for_key(key: str) -> tuple[float, float]:
    name = str(key).lower()
    if name == "month":
        return 6.5, 6.0
    if name in {"day_of_month", "day"}:
        return 15.5, 15.5
    if name == "hour":
        return 12.0, 12.0
    if name in {"outdoor_temperature", "drybulb", "dry_bulb_temperature"}:
        return 15.0, 20.0
    if name in {"air_temperature", "heating_setpoint", "cooling_setpoint", "zone_temperature"}:
        return 22.0, 10.0
    if "humidity" in name:
        return 50.0, 50.0
    if name == "wind_speed":
        return 0.0, 10.0
    if name == "wind_direction":
        return 180.0, 180.0
    if "solar_radiation" in name or "solar" in name:
        return 500.0, 500.0
    if "occupant" in name or "occupancy" in name or "people" in name:
        return 0.0, 10.0
    if "co2" in name or "emission" in name:
        return 0.0, 1.0e7
    if "electricity" in name or "power" in name:
        return 0.0, 1.0e4
    return 0.0, 1.0


def observation_affine_tensors_for_keys(keys: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    shifts = []
    scales = []
    for key in keys:
        shift, scale = default_obs_affine_for_key(key)
        shifts.append(float(shift))
        scales.append(max(float(scale), 1.0e-6))
    return torch.tensor(shifts, dtype=torch.float32), torch.tensor(scales, dtype=torch.float32)


def observation_affine_tensors_from_cfg(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    return observation_affine_tensors_for_keys(cfg.model_observation_keys)


def select_model_observation_tensor(obs: torch.Tensor, cfg: Config) -> torch.Tensor:
    if len(cfg.model_observation_indices) == 0:
        raise RuntimeError("cfg.model_observation_indices is empty; resolve it before building the model")
    idx = torch.as_tensor(cfg.model_observation_indices, device=obs.device, dtype=torch.long)
    return obs.index_select(-1, idx)


def resolve_dynamics_exog_indices(cfg: Config) -> list[int]:
    mapping = obs_index_map(cfg.model_observation_keys)
    out = []
    for key in getattr(cfg, "dynamics_exog_keys", []):
        if key in mapping:
            out.append(int(mapping[key]))
    return out


def dynamics_exog_affine_tensors_from_cfg(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    keys = [cfg.model_observation_keys[i] for i in getattr(cfg, "dynamics_exog_indices", [])]
    if not keys:
        return torch.zeros(0, dtype=torch.float32), torch.ones(0, dtype=torch.float32)
    return observation_affine_tensors_for_keys(keys)


def extract_dynamics_exog_tensor(model_obs: torch.Tensor, cfg: Config, normalize: bool | None = None) -> torch.Tensor:
    if normalize is None:
        normalize = bool(getattr(cfg, "dynamics_exog_normalize", True))
    idxs = list(getattr(cfg, "dynamics_exog_indices", []))
    shape0 = tuple(model_obs.shape[:-1])
    if len(idxs) == 0:
        return model_obs.new_zeros(shape0 + (0,))
    idx = torch.as_tensor(idxs, device=model_obs.device, dtype=torch.long)
    exog = model_obs.index_select(-1, idx).float()
    if not normalize:
        return exog
    shift, scale = dynamics_exog_affine_tensors_from_cfg(cfg)
    if shift.numel() == 0:
        return exog
    shift = shift.to(device=exog.device, dtype=exog.dtype)
    scale = scale.to(device=exog.device, dtype=exog.dtype).clamp_min(1.0e-6)
    while shift.ndim < exog.ndim:
        shift = shift.unsqueeze(0)
        scale = scale.unsqueeze(0)
    return torch.clamp((exog - shift) / scale, -10.0, 10.0)


def infer_forecast_index_map(cfg: Config, horizon: int) -> dict[int, list[int | None]]:
    keys = list(getattr(cfg, "model_observation_keys", []))
    mapping = {name: i for i, name in enumerate(keys)}
    explicit = dict(getattr(cfg, "forecast_key_map", {}))
    out: dict[int, list[int | None]] = {}
    for base_idx in getattr(cfg, "dynamics_exog_indices", []):
        base_key = keys[int(base_idx)]
        per_step: list[int | None] = []
        explicit_names = explicit.get(base_key, None)
        for step in range(1, int(max(horizon, 0)) + 1):
            found = None
            if isinstance(explicit_names, (list, tuple)) and step - 1 < len(explicit_names):
                name = explicit_names[step - 1]
                found = mapping.get(name, None)
            if found is None:
                candidates = [
                    f"{base_key}_forecast_t+{step}",
                    f"{base_key}_forecast_{step}",
                    f"forecast_{base_key}_t+{step}",
                    f"forecast_{base_key}_{step}",
                    f"{base_key}_t+{step}",
                    f"{base_key}_plus{step}",
                    f"{base_key}_step{step}",
                    f"{base_key}_h{step}",
                    f"future_{base_key}_{step}",
                    f"future_{base_key}_t+{step}",
                    f"pred_{base_key}_{step}",
                    f"pred_{base_key}_t+{step}",
                    f"{base_key}_next" if step == 1 else None,
                ]
                for cand in candidates:
                    if cand is None:
                        continue
                    if cand in mapping:
                        found = mapping[cand]
                        break
            per_step.append(None if found is None else int(found))
        out[int(base_idx)] = per_step
    return out


def build_exog_rollout_sequence_from_current_model_obs(model_obs: torch.Tensor, cfg: Config, horizon: int) -> torch.Tensor:
    base_exog = extract_dynamics_exog_tensor(model_obs, cfg, normalize=True)
    H = int(max(horizon, 0))
    if H <= 0:
        return base_exog.unsqueeze(-2)[..., :0, :]
    if base_exog.shape[-1] == 0:
        return base_exog.unsqueeze(-2).expand(*base_exog.shape[:-1], H, 0)
    forecast_map = getattr(cfg, "forecast_index_map", {}) or {}
    seq = []
    for step in range(H):
        parts = []
        for exog_slot, base_idx in enumerate(getattr(cfg, "dynamics_exog_indices", [])):
            source_idx = None
            per_step = forecast_map.get(int(base_idx), None)
            if isinstance(per_step, list) and step < len(per_step):
                source_idx = per_step[step]
            if source_idx is None:
                parts.append(base_exog[..., exog_slot:exog_slot + 1])
            else:
                raw = model_obs[..., int(source_idx): int(source_idx) + 1].float()
                base_key = cfg.model_observation_keys[int(base_idx)]
                shift, scale = default_obs_affine_for_key(base_key)
                raw = torch.clamp((raw - float(shift)) / max(float(scale), 1.0e-6), -10.0, 10.0)
                parts.append(raw)
        seq.append(torch.cat(parts, dim=-1))
    return torch.stack(seq, dim=-2)


def get_current_temperature_from_model_obs(model_obs: torch.Tensor, cfg: Config) -> torch.Tensor:
    idx = int(getattr(cfg, "air_temperature_obs_idx", 0))
    return model_obs[..., idx].float()


def get_future_temperature_from_next_obs(next_model_obs: torch.Tensor, cfg: Config) -> torch.Tensor:
    idx = int(getattr(cfg, "air_temperature_obs_idx", 0))
    return next_model_obs[..., idx].float()


def get_constraint_cost_tensor(td) -> torch.Tensor:
    x = safe_key(td, ("next", "constraint_cost"))
    if x is None:
        x = torch.zeros_like(td["next", "reward"])
    x = x.float()
    require_finite(x, "constraint_cost")
    return x


def get_lambda_signal_tensor(td, cfg: Config) -> torch.Tensor:
    signal_type = str(getattr(cfg, "lambda_signal_type", "magnitude")).lower()

    if signal_type == "magnitude":
        x = safe_key(td, ("next", "constraint_cost"))
        if x is None:
            x = torch.zeros_like(td["next", "reward"])
        x = x.float()
        require_finite(x, "constraint_cost")
        return x

    if signal_type == "binary":
        x = safe_key(td, ("next", "env_constraint_violation"))
        if x is None:
            x = torch.zeros_like(td["next", "reward"])
        x = x.float()
        require_finite(x, "env_constraint_violation")
        return x

    raise ValueError(f"Unsupported lambda_signal_type={signal_type!r}")


class AffineObservationNormalizer(nn.Module):
    def __init__(self, shift: torch.Tensor, scale: torch.Tensor, clip_value: float = 10.0):
        super().__init__()
        self.register_buffer("shift", shift.float().reshape(1, -1))
        self.register_buffer("scale", scale.float().reshape(1, -1).clamp_min(1.0e-6))
        self.clip_value = float(clip_value)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs.float() - self.shift) / self.scale
        if self.clip_value > 0.0:
            x = torch.clamp(x, -self.clip_value, self.clip_value)
        return x


class ObservationFeatureExtractor(nn.Module):
    def __init__(self, selected_indices: list[int], shift: torch.Tensor, scale: torch.Tensor, clip_value: float, hidden: int):
        super().__init__()
        self.register_buffer("selected_indices", torch.as_tensor(selected_indices, dtype=torch.long))
        self.normalizer = AffineObservationNormalizer(shift, scale, clip_value=clip_value)
        self.net = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def select(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.index_select(-1, self.selected_indices.to(device=obs.device))

    def forward(self, obs: torch.Tensor):
        model_obs = self.select(obs.float())
        obs_n = self.normalizer(model_obs)
        return model_obs, obs_n, self.net(obs_n)


class NormalizedCritic(nn.Module):
    def __init__(self, selected_indices: list[int], shift: torch.Tensor, scale: torch.Tensor, clip_value: float, obs_dim: int, hidden_dims=(256, 256, 256)):
        super().__init__()
        self.register_buffer("selected_indices", torch.as_tensor(selected_indices, dtype=torch.long))
        self.normalizer = AffineObservationNormalizer(shift, scale, clip_value=clip_value)
        self.net = mlp(obs_dim, list(hidden_dims), 1, activation=nn.Tanh)

    def select(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.index_select(-1, self.selected_indices.to(device=obs.device))

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return self.normalizer(self.select(obs.float()))

    def forward(self, obs: torch.Tensor):
        obs_n = self.normalize(obs)
        return self.net(obs_n).squeeze(-1)


def attach_normalized_observation_(td, critic: NormalizedCritic):
    if td is None:
        return td
    if "observation" in td.keys(True, True):
        td["model_observation"] = critic.select(td["observation"].float())
        td["observation_normalized"] = critic.normalize(td["observation"].float())
    if ("next", "observation") in td.keys(True, True):
        td["next", "model_observation"] = critic.select(td["next", "observation"].float())
        td["next", "observation_normalized"] = critic.normalize(td["next", "observation"].float())
    return td


def compute_gae_inplace(td, *, reward_key=("next", "reward"), value_key="state_value", next_value_key=("next", "state_value"), out_adv_key="advantage", out_tgt_key="value_target", gamma=0.99, lmbda=0.95):
    time_dim = get_time_dim(td)
    r = td[reward_key]
    v = td[value_key]
    vn = td[next_value_key]

    if ("next", "done") in td.keys(True, True):
        d = td["next", "done"]
    else:
        term = td.get(("next", "terminated"), None)
        trunc = td.get(("next", "truncated"), None)
        if term is None and trunc is None:
            d = torch.zeros_like(r, dtype=torch.bool)
        else:
            if term is None:
                term = torch.zeros_like(trunc, dtype=torch.bool)
            if trunc is None:
                trunc = torch.zeros_like(term, dtype=torch.bool)
            d = term | trunc

    if r.ndim == 2:
        r = r.unsqueeze(-1)
    if v.ndim == 2:
        v = v.unsqueeze(-1)
    if vn.ndim == 2:
        vn = vn.unsqueeze(-1)
    if d.ndim == 2:
        d = d.unsqueeze(-1)

    require_finite(r, f"{reward_key}")
    require_finite(v, value_key)
    require_finite(vn, f"{next_value_key}")
    require_finite(d.float(), "done/terminated/truncated")

    r_bt = r.movedim(time_dim, 1)
    v_bt = v.movedim(time_dim, 1)
    vn_bt = vn.movedim(time_dim, 1)
    d_bt = d.movedim(time_dim, 1)

    not_done = (~d_bt.bool()).float()
    delta = r_bt + gamma * not_done * vn_bt - v_bt
    require_finite(delta, "delta")

    B, T, _ = delta.shape
    gae_bt = torch.zeros_like(delta)
    acc = torch.zeros((B, 1), device=delta.device, dtype=delta.dtype)

    for t in reversed(range(T)):
        acc = delta[:, t, :] + gamma * lmbda * not_done[:, t, :] * acc
        gae_bt[:, t, :] = acc

    tgt_bt = gae_bt + v_bt
    td[out_adv_key] = gae_bt.movedim(1, time_dim)
    td[out_tgt_key] = tgt_bt.movedim(1, time_dim)

    require_finite(td[out_adv_key], out_adv_key)
    require_finite(td[out_tgt_key], out_tgt_key)


def mlp(in_dim: int, hidden_dims: list[int], out_dim: int, activation=nn.Tanh) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), activation()])
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ClampLocScale(nn.Module):
    def __init__(self, min_scale=1e-3, max_scale=10.0, loc_max=5.0):
        super().__init__()
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.loc_max = float(loc_max)

    def forward(self, loc, scale):
        loc = self.loc_max * torch.tanh(loc / self.loc_max)
        scale = torch.clamp(scale, self.min_scale, self.max_scale)
        return loc, scale


class RollingWindowBuffer(nn.Module):
    def __init__(self, window_size: int, hidden_size: int):
        super().__init__()
        self.W = int(window_size)
        self.H = int(hidden_size)

    def _normalize_is_init(self, is_init, B, device):
        if is_init is None or not isinstance(is_init, torch.Tensor):
            return None
        x = is_init.to(device=device)
        while x.ndim > 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim == 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 1:
            raise RuntimeError(f"is_init must become shape [B], got {tuple(x.shape)}")
        if x.shape[0] != B:
            raise RuntimeError(f"is_init batch mismatch: got {tuple(x.shape)}, expected [B={B}]")
        return x.to(torch.bool)

    def _step_update(self, x_bh, window_bwh, is_init_b=None):
        B, W, H = window_bwh.shape
        if W != self.W or H != self.H:
            raise RuntimeError(f"window shape mismatch: got {tuple(window_bwh.shape)}, expected [B,{self.W},{self.H}]")

        is_init_b = self._normalize_is_init(is_init_b, B, x_bh.device)
        if is_init_b is not None:
            reset_win = is_init_b.unsqueeze(-1).unsqueeze(-1)
            window_bwh = torch.where(reset_win, torch.zeros_like(window_bwh), window_bwh)

        return torch.cat([window_bwh[:, 1:, :], x_bh.unsqueeze(1)], dim=1)

    def forward(self, lstm_out, recurrent_window, is_init=None):
        if lstm_out.ndim == 2:
            next_window = self._step_update(lstm_out, recurrent_window, is_init)
            return next_window, next_window

        if lstm_out.ndim != 3:
            raise RuntimeError(f"Unsupported lstm_out rank: {lstm_out.ndim}")

        B, T, _ = lstm_out.shape
        if recurrent_window.ndim == 4:
            window = recurrent_window[:, 0, :, :]
            expanded_in = True
        elif recurrent_window.ndim == 3:
            window = recurrent_window
            expanded_in = False
        else:
            raise RuntimeError(f"Seq mode expected recurrent_window [B,W,H] or [B,T,W,H], got {tuple(recurrent_window.shape)}")

        is_init_bt = None
        if isinstance(is_init, torch.Tensor):
            x = is_init[..., 0] if is_init.ndim == 3 and is_init.shape[-1] == 1 else is_init
            is_init_bt = x.to(torch.bool)

        seq_windows = []
        cur_window = window
        for t in range(T):
            init_t = is_init_bt[:, t] if is_init_bt is not None else None
            cur_window = self._step_update(lstm_out[:, t, :], cur_window, init_t)
            seq_windows.append(cur_window)

        window_seq = torch.stack(seq_windows, dim=1)
        next_window = window_seq if expanded_in else cur_window.unsqueeze(1).expand(B, T, self.W, self.H)
        return window_seq, next_window


class WindowTransformer(nn.Module):
    def __init__(self, d_model, nhead, n_layers, ff_dim, dropout=0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, window_seq):
        if window_seq.ndim == 3:
            return self.enc(window_seq)[:, -1, :]
        if window_seq.ndim == 4:
            B, T, W, H = window_seq.shape
            x = window_seq.reshape(B * T, W, H)
            return self.enc(x)[:, -1, :].reshape(B, T, H)
        raise RuntimeError(f"WindowTransformer expected ndim 3 or 4, got {window_seq.ndim}")


class LatentStateProjector(nn.Module):
    def __init__(self, in_dim: int, state_dim: int, hidden: int = 128, obs_clip: float = 10.0, **_ignored):
        super().__init__()
        self.state_dim = int(state_dim)
        self.obs_clip = float(obs_clip)
        self.net = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.state_dim),
        )

    def forward(self, x: torch.Tensor, obs: torch.Tensor | None = None):
        x = x.float()
        if obs is not None:
            obs = obs.float()
            obs = torch.clamp(obs, -self.obs_clip, self.obs_clip)
            feat = torch.cat([x, obs], dim=-1)
        else:
            feat = x
        return self.net(feat)


TemperatureAnchoredLatentProjector = LatentStateProjector


class LatentTemperatureDecoder(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128, exog_dim: int = 0, **_ignored):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q: torch.Tensor, exog: torch.Tensor | None = None) -> torch.Tensor:
        q = q.float()
        if exog is not None and exog.shape[-1] > 0:
            feat = torch.cat([q, exog.float()], dim=-1)
        else:
            feat = q
        return self.net(feat).squeeze(-1)


class SwitchingLatentDynamics(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, num_regimes: int, hidden: int = 128, exog_dim: int = 0):
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.num_regimes = int(num_regimes)
        self.exog_dim = int(max(exog_dim, 0))

        regime_in = self.state_dim + self.exog_dim
        self.regime_net = nn.Sequential(
            nn.Linear(regime_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.num_regimes),
        )

        A0 = torch.eye(self.state_dim).unsqueeze(0).repeat(self.num_regimes, 1, 1)
        self.A = nn.Parameter(A0 + 0.03 * torch.randn_like(A0))
        self.B = nn.Parameter(0.03 * torch.randn(self.num_regimes, self.state_dim, self.act_dim))
        self.E = nn.Parameter(0.03 * torch.randn(self.num_regimes, self.state_dim, self.exog_dim)) if self.exog_dim > 0 else None
        self.b = nn.Parameter(torch.zeros(self.num_regimes, self.state_dim))

    def _zero_exog(self, q: torch.Tensor) -> torch.Tensor:
        return q.new_zeros(q.shape[:-1] + (self.exog_dim,))

    def regime_probs(self, q: torch.Tensor, exog: torch.Tensor | None = None):
        if self.exog_dim > 0:
            if exog is None:
                exog = self._zero_exog(q)
            feat = torch.cat([q, exog], dim=-1)
        else:
            feat = q
        logits = self.regime_net(feat)
        alpha = torch.softmax(logits, dim=-1)
        return logits, alpha

    def state_drift(self, q: torch.Tensor, alpha: torch.Tensor, exog: torch.Tensor | None = None):
        I = torch.eye(self.state_dim, device=q.device, dtype=q.dtype)
        drift_per_regime = torch.einsum("kij,...j->...ki", self.A - I.unsqueeze(0), q) + self.b
        if self.exog_dim > 0:
            if exog is None:
                exog = self._zero_exog(q)
            drift_per_regime = drift_per_regime + torch.einsum("kij,...j->...ki", self.E, exog)
        return torch.einsum("...k,...ki->...i", alpha, drift_per_regime)

    def control_effect(self, a: torch.Tensor, alpha: torch.Tensor):
        ctrl_per_regime = torch.einsum("kij,...j->...ki", self.B, a)
        return torch.einsum("...k,...ki->...i", alpha, ctrl_per_regime)

    def exog_effect(self, exog: torch.Tensor, alpha: torch.Tensor):
        if self.exog_dim <= 0:
            return exog.new_zeros(exog.shape[:-1] + (self.state_dim,))
        exog_per_regime = torch.einsum("kij,...j->...ki", self.E, exog)
        return torch.einsum("...k,...ki->...i", alpha, exog_per_regime)

    def predict_next(self, q: torch.Tensor, a: torch.Tensor, alpha: torch.Tensor, exog: torch.Tensor | None = None):
        next_per_regime = (
            torch.einsum("kij,...j->...ki", self.A, q)
            + torch.einsum("kij,...j->...ki", self.B, a)
            + self.b
        )
        if self.exog_dim > 0:
            if exog is None:
                exog = self._zero_exog(q)
            next_per_regime = next_per_regime + torch.einsum("kij,...j->...ki", self.E, exog)
        return torch.einsum("...k,...ki->...i", alpha, next_per_regime)


class DynamicsSummaryModule(nn.Module):
    def __init__(self, dyn_model: SwitchingLatentDynamics, cfg: Config | None = None):
        super().__init__()
        self.dyn_model = dyn_model
        self.cfg = cfg

    def forward(self, q: torch.Tensor, model_obs: torch.Tensor | None = None):
        exog = None
        if model_obs is not None and self.cfg is not None:
            exog = extract_dynamics_exog_tensor(model_obs, self.cfg, normalize=True)
        logits, alpha = self.dyn_model.regime_probs(q, exog)
        drift = self.dyn_model.state_drift(q, alpha, exog)
        return logits, alpha, drift


class PolicyConditioner(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, q: torch.Tensor, alpha: torch.Tensor, drift: torch.Tensor):
        return torch.cat([q, alpha, drift], dim=-1)


class ActorLocScaleHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * act_dim),
            NormalParamExtractor(),
        )

    def forward(self, feat: torch.Tensor):
        return self.net(feat)


def decode_temperature_from_q(q: torch.Tensor, temp_decoder: LatentTemperatureDecoder, cfg: Config, temp_base: torch.Tensor | None = None, exog: torch.Tensor | None = None) -> torch.Tensor:
    residual_raw = temp_decoder(q, exog)
    require_finite(residual_raw, "decoded temperature residual (raw)")
    _temp_shift, temp_scale = default_obs_affine_for_key(cfg.sinergym_temperature_variable)
    residual = unnormalize_value_prediction(
        residual_raw,
        scale=float(temp_scale),
        shift=0.0,
    )
    if temp_base is None:
        temperature_hat = residual
    else:
        temperature_hat = temp_base.to(device=residual.device, dtype=residual.dtype) + residual
    require_finite(temperature_hat, "decoded temperature")
    return temperature_hat


def comfort_terms_from_q(q: torch.Tensor, cfg: Config, temp_decoder: LatentTemperatureDecoder, temp_base: torch.Tensor | None = None, exog: torch.Tensor | None = None):
    temperature_hat = decode_temperature_from_q(q, temp_decoder, cfg, temp_base=temp_base, exog=exog)
    comfort_low, comfort_high = active_comfort_band_from_cfg(cfg)
    comfort_low_t = torch.as_tensor(float(comfort_low), device=temperature_hat.device, dtype=temperature_hat.dtype)
    comfort_high_t = torch.as_tensor(float(comfort_high), device=temperature_hat.device, dtype=temperature_hat.dtype)
    lower_gap = comfort_low_t - temperature_hat
    upper_gap = temperature_hat - comfort_high_t
    signed_margin = torch.maximum(lower_gap, upper_gap)
    violation_mag = torch.clamp(signed_margin, min=0.0)
    hard_violation = (violation_mag > 0.0).float()
    return violation_mag, hard_violation, signed_margin, temperature_hat, comfort_low_t, comfort_high_t


def comfort_violation_from_q(q: torch.Tensor, cfg: Config, temp_decoder: LatentTemperatureDecoder, temp_base: torch.Tensor | None = None, exog: torch.Tensor | None = None) -> torch.Tensor:
    violation_mag, *_ = comfort_terms_from_q(q, cfg, temp_decoder, temp_base=temp_base, exog=exog)
    return torch.clamp(violation_mag, min=0.0)


def dreamed_comfort_violation_from_q_sequence(q_seq: torch.Tensor, cfg: Config, temp_decoder: LatentTemperatureDecoder, temp_base: torch.Tensor | None = None, exog_seq: torch.Tensor | None = None):
    if q_seq.ndim != 3:
        raise RuntimeError(f"dreamed_comfort_violation_from_q_sequence expected [N,H+1,D], got {tuple(q_seq.shape)}")
    if q_seq.shape[1] <= 1:
        z = q_seq.new_zeros((q_seq.shape[0], 0))
        return z, z, z, z, z, z
    q_future = q_seq[:, 1:, :]
    if temp_base is None:
        temp_base_seq = None
    else:
        base = temp_base.float()
        if base.ndim == q_future.ndim - 1:
            temp_base_seq = base.unsqueeze(1).expand(q_future.shape[:-1])
        else:
            temp_base_seq = base
    return comfort_terms_from_q(q_future, cfg, temp_decoder, temp_base=temp_base_seq, exog=exog_seq)


def frontloaded_horizon_weights(horizon: int, *, device: torch.device, dtype: torch.dtype, decay: float = 0.65, min_weight: float = 0.25) -> torch.Tensor:
    H = int(max(horizon, 1))
    decay = float(min(max(decay, 1.0e-4), 1.0))
    min_weight = float(max(min_weight, 0.0))
    idx = torch.arange(H, device=device, dtype=dtype)
    weights = torch.pow(torch.full((H,), decay, device=device, dtype=dtype), idx)
    if min_weight > 0.0:
        weights = torch.clamp(weights, min=min_weight)
    return weights / weights.mean().clamp_min(1.0e-8)


def backloaded_horizon_weights(horizon: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    H = int(max(horizon, 1))
    weights = torch.arange(1, H + 1, device=device, dtype=dtype)
    return weights / weights.mean().clamp_min(1.0e-8)


def squash_to_action_bounds(loc: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    low = low.to(device=loc.device, dtype=loc.dtype)
    high = high.to(device=loc.device, dtype=loc.dtype)
    mid = 0.5 * (high + low)
    half = 0.5 * (high - low)
    while mid.ndim < loc.ndim:
        mid = mid.unsqueeze(0)
        half = half.unsqueeze(0)
    return mid + half * torch.tanh(loc)


def action_to_pre_tanh(action: torch.Tensor, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    low = low.to(device=action.device, dtype=action.dtype)
    high = high.to(device=action.device, dtype=action.dtype)
    mid = 0.5 * (high + low)
    half = 0.5 * (high - low)
    while mid.ndim < action.ndim:
        mid = mid.unsqueeze(0)
        half = half.unsqueeze(0)
    y = (action - mid) / torch.clamp(half, min=eps)
    y = torch.clamp(y, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(y) - torch.log1p(-y))


class ShieldedLatentPolicyActor(nn.Module):
    def __init__(self, actor_trunk, actor_policy_core, dyn_model, temp_decoder, low: torch.Tensor, high: torch.Tensor, cfg: Config):
        super().__init__()
        self.actor_trunk = actor_trunk
        self.actor_policy_core = actor_policy_core
        self.dyn_model = dyn_model
        self.temp_decoder = temp_decoder
        self.cfg = cfg
        self.register_buffer("low", low.detach().clone())
        self.register_buffer("high", high.detach().clone())
        active_now = bool(cfg.shield_enable) and int(cfg.shield_apply_after_batch) <= 0
        self.shield_eval_enabled = active_now
        self.shield_apply_enabled = active_now

    def set_shield_apply_enabled(self, enabled: bool):
        enabled = bool(enabled) and bool(self.cfg.shield_enable)
        self.shield_eval_enabled = enabled
        self.shield_apply_enabled = enabled

    def _dist(self, loc: torch.Tensor, scale: torch.Tensor):
        return TanhNormal(loc, torch.clamp(scale, 1e-3, 10.0), low=self.low, high=self.high)

    def _policy_mean_from_latent(self, q: torch.Tensor, model_obs: torch.Tensor | None = None, exog_override: torch.Tensor | None = None):
        exog = exog_override
        if exog is None and model_obs is not None:
            exog = extract_dynamics_exog_tensor(model_obs, self.cfg, normalize=True)
        _, alpha = self.dyn_model.regime_probs(q, exog)
        drift = self.dyn_model.state_drift(q, alpha, exog)
        feat = torch.cat([q, alpha, drift], dim=-1)
        loc, _scale = self.actor_policy_core(feat)
        act = squash_to_action_bounds(loc, self.low, self.high)
        return act, alpha, drift

    def _build_nominal_action_sequence(self, q0: torch.Tensor, first_action: torch.Tensor, exog_seq: torch.Tensor | None = None, model_obs: torch.Tensor | None = None):
        q = q0
        a_cur = first_action
        seq = []
        H = int(self.cfg.shield_horizon)
        for h in range(H):
            seq.append(a_cur)
            exog_h = None if exog_seq is None else exog_seq[:, h, :]
            _, alpha = self.dyn_model.regime_probs(q, exog_h)
            q = self.dyn_model.predict_next(q, a_cur, alpha, exog_h)
            if h < H - 1:
                next_exog = None if exog_seq is None else exog_seq[:, min(h + 1, H - 1), :]
                a_cur, _, _ = self._policy_mean_from_latent(q, model_obs if exog_seq is None else None, exog_override=next_exog)
        return torch.stack(seq, dim=1)

    def _frontload_weights(self, horizon: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return frontloaded_horizon_weights(
            horizon,
            device=device,
            dtype=dtype,
            decay=float(getattr(self.cfg, "shield_frontload_decay", 0.65)),
            min_weight=float(getattr(self.cfg, "shield_frontload_min_weight", 0.25)),
        )

    def _summarize_rollout_sequences(self, pred_cost_seq: torch.Tensor):
        if pred_cost_seq.ndim != 2:
            raise RuntimeError(f"pred_cost_seq must be [B,H], got {tuple(pred_cost_seq.shape)}")
        if pred_cost_seq.shape[1] == 0:
            z = pred_cost_seq.new_zeros(pred_cost_seq.shape[0])
            return {
                "weighted_violation_mean": z,
                "max_violation": z,
                "early_violation_mean": z,
                "early_violation_max": z,
            }

        H = int(pred_cost_seq.shape[1])
        weights = self._frontload_weights(H, pred_cost_seq.device, pred_cost_seq.dtype).view(1, H)
        weighted_violation_mean = (pred_cost_seq * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0e-8)

        early_h = max(1, min(int(getattr(self.cfg, "shield_early_horizon", 4)), H))
        early_cost_seq = pred_cost_seq[:, :early_h]
        early_weights = weights[:, :early_h]
        early_violation_mean = (early_cost_seq * early_weights).sum(dim=1) / early_weights.sum(dim=1).clamp_min(1.0e-8)
        early_violation_max = early_cost_seq.max(dim=1).values
        max_violation = pred_cost_seq.max(dim=1).values
        return {
            "weighted_violation_mean": weighted_violation_mean,
            "max_violation": max_violation,
            "early_violation_mean": early_violation_mean,
            "early_violation_max": early_violation_max,
        }

    def _rollout_objectives_from_q_sequence(self, q_seq: torch.Tensor, temp_base: torch.Tensor, exog_seq: torch.Tensor | None = None):
        pred_cost_seq, *_ = dreamed_comfort_violation_from_q_sequence(q_seq, self.cfg, self.temp_decoder, temp_base=temp_base, exog_seq=exog_seq)
        if pred_cost_seq.shape[1] == 0:
            z = q_seq.new_zeros(q_seq.shape[0])
            return {
                "weighted_violation_mean": z,
                "max_violation": z,
                "early_violation_mean": z,
                "early_violation_max": z,
            }
        return self._summarize_rollout_sequences(pred_cost_seq)

    def _rollout_objectives_from_first_action(self, q0: torch.Tensor, first_action: torch.Tensor, temp_base: torch.Tensor, exog_seq: torch.Tensor | None = None, model_obs: torch.Tensor | None = None):
        q = q0
        a_cur = first_action
        q_seq = [q]
        H = int(self.cfg.shield_horizon)
        for h in range(H):
            exog_h = None if exog_seq is None else exog_seq[:, h, :]
            _, alpha = self.dyn_model.regime_probs(q, exog_h)
            q = self.dyn_model.predict_next(q, a_cur, alpha, exog_h)
            q_seq.append(q)
            if h < H - 1:
                next_exog = None if exog_seq is None else exog_seq[:, min(h + 1, H - 1), :]
                a_cur, _, _ = self._policy_mean_from_latent(q, model_obs if exog_seq is None else None, exog_override=next_exog)
        q_seq = torch.stack(q_seq, dim=1)
        return self._rollout_objectives_from_q_sequence(q_seq, temp_base=temp_base, exog_seq=exog_seq)

    def _latent_rollout_from_action_sequence(self, q0: torch.Tensor, action_seq: torch.Tensor, exog_seq: torch.Tensor | None = None) -> torch.Tensor:
        return rollout_latent_sequence_from_start(q0, action_seq, exog_seq, self.dyn_model)

    def _rollout_objectives_from_action_sequence(self, q0: torch.Tensor, action_seq: torch.Tensor, temp_base: torch.Tensor, exog_seq: torch.Tensor | None = None):
        q_seq = self._latent_rollout_from_action_sequence(q0, action_seq, exog_seq=exog_seq)
        return self._rollout_objectives_from_q_sequence(q_seq, temp_base=temp_base, exog_seq=exog_seq)

    def _shield_score(self, stats: dict[str, torch.Tensor], deviation: torch.Tensor | None = None) -> torch.Tensor:
        score = (
            float(self.cfg.shield_violation_coef) * stats["weighted_violation_mean"]
            + float(self.cfg.shield_violation_max_coef) * stats["max_violation"]
            + float(getattr(self.cfg, "shield_early_violation_coef", 0.0)) * stats["early_violation_mean"]
            + float(getattr(self.cfg, "shield_early_violation_max_coef", 0.0)) * stats["early_violation_max"]
        )
        if deviation is not None:
            score = score + float(self.cfg.shield_deviation_coef) * deviation
        return score

    def _optimize_action_sequence(self, q0: torch.Tensor, base_action: torch.Tensor, temp_base: torch.Tensor, exog_seq: torch.Tensor | None = None, model_obs: torch.Tensor | None = None):
        steps = int(self.cfg.shield_opt_steps)
        q0_det = q0.detach()
        base_det = base_action.detach()

        with torch.no_grad():
            nominal_seq = self._build_nominal_action_sequence(q0_det, base_det, exog_seq=exog_seq, model_obs=model_obs)
            B, H, A = nominal_seq.shape
            low = self.low.to(device=nominal_seq.device, dtype=nominal_seq.dtype).view(1, 1, A)
            high = self.high.to(device=nominal_seq.device, dtype=nominal_seq.dtype).view(1, 1, A)
            best_seq = nominal_seq.clone()
            best_q_seq = self._latent_rollout_from_action_sequence(q0_det, best_seq, exog_seq=exog_seq)
            best_stats = self._rollout_objectives_from_q_sequence(best_q_seq, temp_base=temp_base, exog_seq=exog_seq)
            best_dev = ((best_seq - nominal_seq) ** 2).mean(dim=(1, 2))
            best_score = self._shield_score(best_stats, deviation=best_dev)

        if steps <= 0:
            return {
                "nominal_seq": nominal_seq.detach(),
                "best_seq": best_seq.detach(),
                "best_q_seq": best_q_seq.detach(),
                "best_score": best_score.detach(),
                "best_stats": {k: v.detach() for k, v in best_stats.items()},
            }

        lr = float(self.cfg.shield_opt_lr)
        grad_clip = float(self.cfg.shield_opt_grad_clip)
        pre_tanh = action_to_pre_tanh(nominal_seq, low, high).detach()

        with torch.enable_grad():
            for _ in range(steps):
                pre_tanh = pre_tanh.detach().requires_grad_(True)
                cand_seq = squash_to_action_bounds(pre_tanh, low, high)
                cand_q_seq = self._latent_rollout_from_action_sequence(q0_det, cand_seq, exog_seq=exog_seq)
                cand_stats = self._rollout_objectives_from_q_sequence(cand_q_seq, temp_base=temp_base, exog_seq=exog_seq)
                dev = ((cand_seq - nominal_seq) ** 2).mean(dim=(1, 2))
                score = self._shield_score(cand_stats, deviation=dev)

                loss = score.sum()
                grad_pre_tanh, = torch.autograd.grad(loss, pre_tanh, retain_graph=False, create_graph=False, allow_unused=False)
                if not torch.isfinite(grad_pre_tanh).all():
                    break

                if grad_clip > 0.0:
                    flat = grad_pre_tanh.reshape(B, -1)
                    grad_norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    clip_scale = torch.clamp(grad_clip / grad_norm, max=1.0)
                    grad_pre_tanh = grad_pre_tanh * clip_scale.view(B, 1, 1)

                with torch.no_grad():
                    improved = score < best_score
                    best_seq = torch.where(improved.view(B, 1, 1), cand_seq.detach(), best_seq)
                    best_q_seq = torch.where(improved.view(B, 1, 1), cand_q_seq.detach(), best_q_seq)
                    best_score = torch.where(improved, score.detach(), best_score)
                    for key in list(best_stats.keys()):
                        best_stats[key] = torch.where(improved, cand_stats[key].detach(), best_stats[key])
                    pre_tanh = pre_tanh - lr * grad_pre_tanh

        return {
            "nominal_seq": nominal_seq.detach(),
            "best_seq": best_seq.detach(),
            "best_q_seq": best_q_seq.detach(),
            "best_score": best_score.detach(),
            "best_stats": {k: v.detach() for k, v in best_stats.items()},
        }

    def plan_shield_sequence(self, q0: torch.Tensor, base_action: torch.Tensor, model_obs: torch.Tensor | None = None):
        qd = q0.detach()
        bd = base_action.detach()
        horizon = max(1, int(getattr(self.cfg, "shield_horizon", 1)))

        if (not self.shield_eval_enabled) or horizon <= 0:
            z = torch.zeros(qd.shape[:-1], device=qd.device, dtype=qd.dtype)
            base_seq = bd.unsqueeze(1)
            q_seq = torch.stack([qd, qd], dim=1)
            return {
                "planned_action": bd,
                "planned_sequence": base_seq,
                "planned_q_sequence": q_seq,
                "active": z.bool(),
                "trigger": z.bool(),
                "commit_length": torch.ones_like(z, dtype=torch.long),
                "shield_logs": {
                    "shield_eval_enabled": z.unsqueeze(-1),
                    "shield_apply_enabled": z.unsqueeze(-1),
                    "shield_active": z.unsqueeze(-1),
                    "shield_changed": z.unsqueeze(-1),
                    "shield_would_change": z.unsqueeze(-1),
                    "shield_base_violation": z.unsqueeze(-1),
                    "shield_best_violation": z.unsqueeze(-1),
                    "shield_chosen_violation": z.unsqueeze(-1),
                    "shield_delta_norm": z.unsqueeze(-1),
                    "shield_would_delta_norm": z.unsqueeze(-1),
                },
            }

        with torch.no_grad():
            exog_seq = None if model_obs is None else build_exog_rollout_sequence_from_current_model_obs(model_obs, self.cfg, horizon)
            temp_base = qd.new_zeros(qd.shape[:-1]) if model_obs is None else get_current_temperature_from_model_obs(model_obs, self.cfg)
            base_stats = self._rollout_objectives_from_first_action(qd, bd, temp_base=temp_base, exog_seq=exog_seq, model_obs=model_obs)
            base_viol = base_stats["max_violation"]
            base_score = self._shield_score(base_stats, deviation=None)
            nominal_seq = self._build_nominal_action_sequence(qd, bd, exog_seq=exog_seq, model_obs=model_obs)
            nominal_q_seq = self._latent_rollout_from_action_sequence(qd, nominal_seq, exog_seq=exog_seq)

        trigger = base_viol > float(self.cfg.shield_trigger_violation)
        tol = float(self.cfg.shield_opt_tol)

        opt_seq_full = nominal_seq.clone()
        opt_q_seq_full = nominal_q_seq.clone()
        opt_score_full = base_score.clone()
        opt_viol_full = base_viol.clone()

        if trigger.any().item():
            trig_exog = None if exog_seq is None else exog_seq[trigger]
            trig_model_obs = None if model_obs is None else model_obs[trigger]
            result = self._optimize_action_sequence(qd[trigger], bd[trigger], temp_base=temp_base[trigger], exog_seq=trig_exog, model_obs=trig_model_obs)
            opt_seq_full[trigger] = result["best_seq"]
            opt_q_seq_full[trigger] = result["best_q_seq"]
            opt_score_full[trigger] = result["best_score"]
            opt_viol_full[trigger] = result["best_stats"]["max_violation"]

        improved = opt_score_full < (base_score - tol)
        if bool(self.cfg.shield_only_if_triggered):
            use_opt = trigger & improved
        else:
            use_opt = improved

        candidate_seq = torch.where(use_opt.view(-1, 1, 1), opt_seq_full, nominal_seq)
        candidate_q_seq = torch.where(use_opt.view(-1, 1, 1), opt_q_seq_full, nominal_q_seq)
        candidate_action = candidate_seq[:, 0, :]
        candidate_viol = torch.where(use_opt, opt_viol_full, base_viol)

        would_change = ((candidate_action - bd).abs().max(dim=-1).values > 1e-6).float()
        would_delta_norm = (candidate_action - bd).norm(dim=-1)

        if self.shield_apply_enabled:
            final_action = candidate_action
            final_viol = candidate_viol
            active = (use_opt & trigger).float() if bool(self.cfg.shield_only_if_triggered) else use_opt.float()
        else:
            final_action = bd
            final_viol = base_viol
            active = torch.zeros_like(base_viol)
            candidate_seq = nominal_seq
            candidate_q_seq = nominal_q_seq

        changed = ((final_action - bd).abs().max(dim=-1).values > 1e-6).float()
        delta_norm = (final_action - bd).norm(dim=-1)
        apply_flag = torch.full_like(base_viol, 1.0 if self.shield_apply_enabled else 0.0)
        eval_flag = torch.full_like(base_viol, 1.0 if self.shield_eval_enabled else 0.0)
        best_viol = torch.minimum(base_viol, opt_viol_full)
        max_commit = max(1, int(getattr(self.cfg, "shield_commit_max_steps", 1)))
        commit_length = torch.full_like(base_viol, int(min(max_commit, candidate_seq.shape[1])), dtype=torch.long)

        return {
            "planned_action": final_action.detach(),
            "planned_sequence": candidate_seq.detach(),
            "planned_q_sequence": candidate_q_seq.detach(),
            "active": active > 0.5,
            "trigger": trigger.detach(),
            "commit_length": commit_length.detach(),
            "shield_logs": {
                "shield_eval_enabled": eval_flag.unsqueeze(-1).detach(),
                "shield_apply_enabled": apply_flag.unsqueeze(-1).detach(),
                "shield_active": active.unsqueeze(-1).detach(),
                "shield_changed": changed.unsqueeze(-1).detach(),
                "shield_would_change": would_change.unsqueeze(-1).detach(),
                "shield_base_violation": base_viol.unsqueeze(-1).detach(),
                "shield_best_violation": best_viol.unsqueeze(-1).detach(),
                "shield_chosen_violation": final_viol.unsqueeze(-1).detach(),
                "shield_delta_norm": delta_norm.unsqueeze(-1).detach(),
                "shield_would_delta_norm": would_delta_norm.unsqueeze(-1).detach(),
            },
        }

    def _shield(self, q0: torch.Tensor, base_action: torch.Tensor, model_obs: torch.Tensor | None = None):
        if (not self.shield_eval_enabled) or int(self.cfg.shield_horizon) <= 0:
            z = torch.zeros(q0.shape[:-1], device=q0.device, dtype=q0.dtype)
            return base_action.detach(), {
                "shield_eval_enabled": z.unsqueeze(-1),
                "shield_apply_enabled": z.unsqueeze(-1),
                "shield_active": z.unsqueeze(-1),
                "shield_changed": z.unsqueeze(-1),
                "shield_would_change": z.unsqueeze(-1),
                "shield_base_violation": z.unsqueeze(-1),
                "shield_best_violation": z.unsqueeze(-1),
                "shield_chosen_violation": z.unsqueeze(-1),
                "shield_delta_norm": z.unsqueeze(-1),
                "shield_would_delta_norm": z.unsqueeze(-1),
            }

        plan = self.plan_shield_sequence(q0, base_action, model_obs=model_obs)
        return plan["planned_action"], plan["shield_logs"]

    def forward(self, td):
        td = self.actor_trunk(td)
        dist = self._dist(td["loc"], td["scale"])
        base_action = dist.rsample()
        action, shield_logs = self._shield(td["actor_dyn_state"].float(), base_action, td.get("model_observation", None))
        action_for_logp = action.detach() if bool(self.cfg.shield_detach_action_for_logprob) else action
        logp = dist.log_prob(action_for_logp)
        if logp.ndim == action.ndim:
            logp = logp.sum(-1, keepdim=True)
        elif logp.ndim == action.ndim - 1:
            logp = logp.unsqueeze(-1)
        td["action"] = action.detach()
        td["action_log_prob"] = logp
        td["base_action"] = base_action.detach()
        for k, v in shield_logs.items():
            td[k] = v
        return td


def shield_apply_enabled_for_batch(batch_idx: int, cfg: Config) -> bool:
    return bool(cfg.shield_enable) and int(batch_idx) >= int(cfg.shield_apply_after_batch)


def _resolve_dynamics_gate(cfg: Config, name: str, *, legacy_fallback: bool = True) -> int:
    gate = getattr(cfg, name, 0)
    if gate is None and legacy_fallback:
        gate = getattr(cfg, "dynamics_aux_enable_after_batch", 0)
    if gate is None:
        gate = 0
    return max(0, int(gate))


def dynamics_aux_enable_after_batch(cfg: Config) -> int:
    # Legacy/reporting helper. The actual training logic uses the two explicit gates below.
    return max(
        _resolve_dynamics_gate(cfg, "temp_supervision_enable_after_batch"),
        _resolve_dynamics_gate(cfg, "backbone_dynamics_enable_after_batch"),
    )


def dynamics_aux_enabled_for_batch(batch_idx: int, cfg: Config) -> bool:
    return temperature_supervision_enabled_for_batch(batch_idx, cfg) or backbone_dynamics_enabled_for_batch(batch_idx, cfg)


def temperature_supervision_enabled_for_batch(batch_idx: int, cfg: Config) -> bool:
    return int(batch_idx) >= _resolve_dynamics_gate(cfg, "temp_supervision_enable_after_batch")


def backbone_dynamics_enabled_for_batch(batch_idx: int, cfg: Config) -> bool:
    return int(batch_idx) >= _resolve_dynamics_gate(cfg, "backbone_dynamics_enable_after_batch")


def unique_params(*modules: nn.Module):
    seen = set()
    out = []
    for module in modules:
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                out.append(p)
                seen.add(id(p))
    return out


def _broadcast_like_last_dim(scale: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor(scale, device=ref.device, dtype=ref.dtype)
    else:
        scale = scale.to(device=ref.device, dtype=ref.dtype)
    while scale.ndim < ref.ndim:
        scale = scale.unsqueeze(0)
    return scale


def channelwise_target_scale(target: torch.Tensor, min_scale: float = 1e-3) -> torch.Tensor:
    if target.ndim == 1:
        scale = target.std(unbiased=False).reshape(1)
    else:
        flat = target.reshape(-1, target.shape[-1])
        scale = flat.std(dim=0, unbiased=False)
    return scale.detach().clamp_min(float(min_scale))


def tensor_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((pred - target) ** 2).mean() + 1e-12)


def normalized_rmse(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    scale_b = _broadcast_like_last_dim(scale, pred).clamp_min(1e-8)
    err = (pred - target) / scale_b
    return torch.sqrt((err ** 2).mean() + 1e-12)


def tensor_corrcoef(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.reshape(-1).float()
    tgt_f = target.reshape(-1).float()
    pred_c = pred_f - pred_f.mean()
    tgt_c = tgt_f - tgt_f.mean()
    pred_std = pred_c.std(unbiased=False)
    tgt_std = tgt_c.std(unbiased=False)
    if float(pred_std.detach().cpu()) < 1e-8 or float(tgt_std.detach().cpu()) < 1e-8:
        return pred_f.new_zeros(())
    denom = (pred_std * tgt_std).clamp_min(1e-8)
    return (pred_c * tgt_c).mean() / denom


def normalized_huber_loss(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor | float, delta: float = 1.0) -> torch.Tensor:
    scale_b = _broadcast_like_last_dim(scale, pred).clamp_min(1e-8)
    err = (pred - target) / scale_b
    abs_err = err.abs()
    delta_t = torch.as_tensor(float(delta), device=err.device, dtype=err.dtype)
    loss = torch.where(abs_err <= delta_t, 0.5 * err.pow(2), delta_t * (abs_err - 0.5 * delta_t))
    return loss.mean()


def rollout_ramp_value(batch_idx: int, ramp_batches: int) -> float:
    ramp_batches = int(max(ramp_batches, 0))
    if ramp_batches <= 0:
        return 1.0
    return float(min(1.0, float(batch_idx + 1) / float(ramp_batches)))


def persistence_surplus_loss(model_metric: torch.Tensor, persist_metric: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    margin_t = torch.as_tensor(float(margin), device=model_metric.device, dtype=model_metric.dtype)
    return (model_metric - persist_metric + margin_t).clamp_min(0.0)


def rollout_latent_for_horizon(q: torch.Tensor, a: torch.Tensor, exog_next: torch.Tensor | None, dyn_model: SwitchingLatentDynamics, horizon: int):
    T = int(q.shape[1])
    if int(horizon) < 1 or int(horizon) >= T:
        raise ValueError(f"Invalid horizon={horizon} for sequence length T={T}")

    q_hat = q[:, : T - horizon, :]
    for j in range(int(horizon)):
        a_j = a[:, j : T - horizon + j, :]
        exog_j = None if exog_next is None else exog_next[:, j : T - horizon + j, :]
        _, alpha_j = dyn_model.regime_probs(q_hat, exog_j)
        q_hat = dyn_model.predict_next(q_hat, a_j, alpha_j, exog_j)
    return q_hat


def rollout_latent_for_horizon_reanchored(q: torch.Tensor, a: torch.Tensor, exog_next: torch.Tensor | None, dyn_model: SwitchingLatentDynamics, horizon: int, reanchor_gap: int):
    """Open-loop rollout with periodic teacher re-anchoring, matching the refrigeration training flow."""
    T = int(q.shape[1])
    if int(horizon) < 1 or int(horizon) >= T:
        raise ValueError(f"Invalid horizon={horizon} for sequence length T={T}")
    gap = max(1, int(reanchor_gap))

    q_hat = q[:, : T - horizon, :]
    steps_done = 0
    while steps_done < int(horizon):
        seg = min(gap, int(horizon) - steps_done)
        for j in range(seg):
            jj = steps_done + j
            a_j = a[:, jj : T - horizon + jj, :]
            exog_j = None if exog_next is None else exog_next[:, jj : T - horizon + jj, :]
            _, alpha_j = dyn_model.regime_probs(q_hat, exog_j)
            q_hat = dyn_model.predict_next(q_hat, a_j, alpha_j, exog_j)
        steps_done += seg
        if steps_done < int(horizon):
            q_hat = q[:, steps_done : T - horizon + steps_done, :]
    return q_hat


def rollout_latent_sequence_from_start(q0: torch.Tensor, action_seq: torch.Tensor, exog_seq: torch.Tensor | None, dyn_model: SwitchingLatentDynamics) -> torch.Tensor:
    q = q0
    q_seq = [q]
    H = int(action_seq.shape[1])
    for h in range(H):
        a_cur = action_seq[:, h, :]
        exog_cur = None if exog_seq is None else exog_seq[:, h, :]
        _, alpha = dyn_model.regime_probs(q, exog_cur)
        q = dyn_model.predict_next(q, a_cur, alpha, exog_cur)
        q_seq.append(q)
    return torch.stack(q_seq, dim=1)



def latent_dynamics_loss(sub_td, state_proj_model: LatentStateProjector, dyn_model: SwitchingLatentDynamics, temp_decoder: LatentTemperatureDecoder, cfg: Config, batch_idx: int):
    time_dim = get_time_dim(sub_td)

    x_lat = move_time_to_dim1(sub_td["actor_xfm_latent"].float(), time_dim)
    obs = select_model_observation_tensor(move_time_to_dim1(sub_td["observation"].float(), time_dim), cfg)
    next_obs = select_model_observation_tensor(move_time_to_dim1(sub_td["next", "observation"].float(), time_dim), cfg)
    exog_now = extract_dynamics_exog_tensor(obs, cfg, normalize=True)
    exog_next = extract_dynamics_exog_tensor(next_obs, cfg, normalize=True)

    temp_supervision_start_batch = _resolve_dynamics_gate(cfg, "temp_supervision_enable_after_batch")
    backbone_dynamics_start_batch = _resolve_dynamics_gate(cfg, "backbone_dynamics_enable_after_batch")
    dynamics_aux_start_batch = max(temp_supervision_start_batch, backbone_dynamics_start_batch)
    backbone_dyn_on = backbone_dynamics_enabled_for_batch(batch_idx, cfg)
    temp_supervision_on = temperature_supervision_enabled_for_batch(batch_idx, cfg)

    q = state_proj_model(x_lat if backbone_dyn_on else x_lat.detach(), obs)
    q_decoder = state_proj_model(x_lat.detach(), obs)
    a = move_time_to_dim1(sub_td["action"].float(), time_dim)

    if q.ndim != 3 or a.ndim != 3:
        raise RuntimeError(f"latent_dynamics_loss expected q/a rank 3, got q={tuple(q.shape)}, a={tuple(a.shape)}")

    _, T, _ = q.shape
    max_h_cfg = int(cfg.dyn_roll_horizon)
    rollout_progress = rollout_ramp_value(batch_idx, cfg.dyn_rollout_ramp_batches)

    def _zero_logs():
        logs = {
            "dyn_rollout_progress": float(rollout_progress),
            "dynamics_aux_enabled": float(temp_supervision_on or backbone_dyn_on),
            "dynamics_aux_enable_after_batch": float(dynamics_aux_start_batch),
            "temp_supervision_enabled": float(temp_supervision_on),
            "temp_supervision_enable_after_batch": float(temp_supervision_start_batch),
            "backbone_dynamics_enabled": float(backbone_dyn_on),
            "backbone_dynamics_enable_after_batch": float(backbone_dynamics_start_batch),
            "latent_dynamics_loss_enabled": float(backbone_dyn_on),
            "dyn_1step": 0.0,
            "dyn_roll": 0.0,
            "regime_maxprob": 0.0,
            "drift_norm": 0.0,
            "ctrl_norm": 0.0,
            "dyn_1step_rmse": 0.0,
            "dyn_1step_nrmse": 0.0,
            "dyn_roll_rmse": 0.0,
            "dyn_roll_nrmse": 0.0,
            "loss_temp_res_dyn1": 0.0,
            "loss_temp_res_roll": 0.0,
            "loss_viol_dyn1": 0.0,
            "loss_viol_roll": 0.0,
            "loss_temp_persist_dyn1": 0.0,
            "loss_temp_persist_roll": 0.0,
            "loss_viol_persist_dyn1": 0.0,
            "loss_viol_persist_roll": 0.0,
            "loss_viol_tail": 0.0,
            "loss_temp_gain_tail": 0.0,
            "loss_temp_gain_consistency": 0.0,
            "loss_temp_roll_bias": 0.0,
            "loss_temp_res_reanchor_roll": 0.0,
            "reanchor_active": 0.0,
            "decoded_temperature_mean": 0.0,
            "decoded_temperature_std": 0.0,
            "decoded_temperature_min": 0.0,
            "decoded_temperature_max": 0.0,
            "latent_state_norm": 0.0,
            "latent_temperature_direct_rmse": 0.0,
            "latent_temperature_direct_nrmse": 0.0,
            "latent_temperature_direct_bias": 0.0,
            "latent_temperature_direct_corr": 0.0,
            "latent_temperature_dyn1_rmse": 0.0,
            "latent_temperature_dyn1_nrmse": 0.0,
            "latent_temperature_dyn1_bias": 0.0,
            "latent_temperature_dyn1_corr": 0.0,
            "latent_temperature_roll_rmse": 0.0,
            "latent_temperature_roll_nrmse": 0.0,
            "latent_temperature_roll_bias": 0.0,
            "latent_temperature_reanchor_roll_rmse": 0.0,
            "latent_temperature_reanchor_roll_nrmse": 0.0,
            "temp_persistence_dyn1_rmse": 0.0,
            "temp_persistence_dyn1_nrmse": 0.0,
            "temp_persistence_roll_rmse": 0.0,
            "temp_persistence_roll_nrmse": 0.0,
            "temp_rmse_gain_dyn1": 0.0,
            "temp_rmse_gain_roll": 0.0,
            "temp_gain_nrmse_last_h": 0.0,
            "temp_gain_nrmse_min_h": 0.0,
            "viol_dyn1_rmse": 0.0,
            "viol_dyn1_nrmse": 0.0,
            "viol_roll_rmse": 0.0,
            "viol_roll_nrmse": 0.0,
            "viol_persistence_dyn1_rmse": 0.0,
            "viol_persistence_dyn1_nrmse": 0.0,
            "viol_persistence_roll_rmse": 0.0,
            "viol_persistence_roll_nrmse": 0.0,
            "viol_rmse_gain_dyn1": 0.0,
            "viol_rmse_gain_roll": 0.0,
            "viol_gain_nrmse_last_h": 0.0,
            "viol_gain_nrmse_min_h": 0.0,
            "latent_constraint_cost_mean": 0.0,
            "latent_constraint_cost_std": 0.0,
            "latent_constraint_cost_max": 0.0,
            "latent_constraint_violation_rate": 0.0,
            "temp_vs_persist_dyn1_surplus": 0.0,
            "temp_vs_persist_roll_surplus": 0.0,
            "temp_beats_persist_dyn1": 0.0,
            "temp_beats_persist_roll": 0.0,
            "viol_vs_persist_dyn1_surplus": 0.0,
            "viol_vs_persist_roll_surplus": 0.0,
            "viol_beats_persist_dyn1": 0.0,
            "viol_beats_persist_roll": 0.0,
        }
        for h in range(1, max_h_cfg + 1):
            logs[f"dyn_latent_nrmse_h{h:02d}"] = 0.0
            logs[f"latent_temperature_nrmse_h{h:02d}"] = 0.0
            logs[f"latent_temperature_rmse_h{h:02d}"] = 0.0
            logs[f"latent_temperature_persist_nrmse_h{h:02d}"] = 0.0
            logs[f"latent_temperature_gain_nrmse_h{h:02d}"] = 0.0
            logs[f"latent_violation_nrmse_h{h:02d}"] = 0.0
            logs[f"latent_violation_rmse_h{h:02d}"] = 0.0
            logs[f"latent_violation_persist_nrmse_h{h:02d}"] = 0.0
            logs[f"latent_violation_gain_nrmse_h{h:02d}"] = 0.0
        return logs

    if T <= 1:
        return q.new_zeros(()), _zero_logs()

    temp_now = get_current_temperature_from_model_obs(obs, cfg)
    temp_next_seq = get_future_temperature_from_next_obs(next_obs, cfg)

    temperature_decoded = decode_temperature_from_q(q_decoder, temp_decoder, cfg, temp_base=temp_now, exog=exog_now)
    temp_direct_err = temperature_decoded - temp_now
    rmse_temp_direct = torch.sqrt((temp_direct_err ** 2).mean() + 1e-12)
    nrmse_temp_direct = normalized_rmse(temperature_decoded, temp_now, 10.0)
    temp_direct_bias = temp_direct_err.mean()
    corr_temp_direct = tensor_corrcoef(temperature_decoded, temp_now)

    latent_cost_now, latent_hard_now, *_ = comfort_terms_from_q(q_decoder, cfg, temp_decoder, temp_base=temp_now, exog=exog_now)

    logits_all, alpha_all = dyn_model.regime_probs(q, exog_now)
    drift_all = dyn_model.state_drift(q, alpha_all, exog_now)
    ctrl_all = dyn_model.control_effect(a, alpha_all)

    q_now = q[:, :-1, :]
    q_next = q[:, 1:, :]
    a_now = a[:, :-1, :]
    exog_step1 = exog_next[:, :-1, :]
    alpha_now = alpha_all[:, :-1, :]
    q_pred = dyn_model.predict_next(q_now, a_now, alpha_now, exog_step1)

    q_decoder_now = q_decoder[:, :-1, :]
    _, alpha_now_decoder = dyn_model.regime_probs(q_decoder_now, exog_step1)
    q_pred_decoder = dyn_model.predict_next(q_decoder_now, a_now, alpha_now_decoder, exog_step1)

    huber_delta = float(cfg.dyn_metric_huber_delta)
    latent_scale = channelwise_target_scale(q_next, min_scale=float(cfg.dyn_metric_min_scale))

    loss_1 = normalized_huber_loss(q_pred, q_next, latent_scale, delta=huber_delta)
    dyn_1step_rmse = tensor_rmse(q_pred, q_next)
    dyn_1step_nrmse = normalized_rmse(q_pred, q_next, latent_scale)

    comfort_low, comfort_high = active_comfort_band_from_cfg(cfg)

    temp_tgt_dyn1 = temp_next_seq[:, :-1]
    temp_base_dyn1 = temp_now[:, :-1]
    temp_hat_dyn1 = decode_temperature_from_q(q_pred_decoder, temp_decoder, cfg, temp_base=temp_base_dyn1, exog=exog_step1)

    temp_res_hat_dyn1 = temp_hat_dyn1 - temp_base_dyn1
    temp_res_tgt_dyn1 = temp_tgt_dyn1 - temp_base_dyn1

    viol_hat_dyn1 = violation_mag_from_temp(temp_hat_dyn1, comfort_low, comfort_high)
    viol_tgt_dyn1 = violation_mag_from_temp(temp_tgt_dyn1, comfort_low, comfort_high)
    viol_base_dyn1 = violation_mag_from_temp(temp_base_dyn1, comfort_low, comfort_high)

    loss_temp_res_dyn1 = normalized_huber_loss(temp_res_hat_dyn1, temp_res_tgt_dyn1, float(cfg.temp_residual_scale), delta=huber_delta)
    loss_viol_dyn1 = normalized_huber_loss(viol_hat_dyn1, viol_tgt_dyn1, float(cfg.violation_loss_scale), delta=huber_delta)

    rmse_temp_dyn1 = tensor_rmse(temp_hat_dyn1, temp_tgt_dyn1)
    nrmse_temp_dyn1 = normalized_rmse(temp_hat_dyn1, temp_tgt_dyn1, 10.0)
    temp_dyn1_bias = (temp_hat_dyn1 - temp_tgt_dyn1).mean()
    corr_temp_dyn1 = tensor_corrcoef(temp_hat_dyn1, temp_tgt_dyn1)

    temp_persist_dyn1_rmse = tensor_rmse(temp_base_dyn1, temp_tgt_dyn1)
    temp_persist_dyn1_nrmse = normalized_rmse(temp_base_dyn1, temp_tgt_dyn1, 10.0)
    temp_rmse_gain_dyn1 = temp_persist_dyn1_rmse - rmse_temp_dyn1
    loss_temp_persist_dyn1 = persistence_surplus_loss(rmse_temp_dyn1, temp_persist_dyn1_rmse, margin=float(cfg.temp_persist_margin))

    viol_rmse_dyn1 = tensor_rmse(viol_hat_dyn1, viol_tgt_dyn1)
    viol_nrmse_dyn1 = normalized_rmse(viol_hat_dyn1, viol_tgt_dyn1, float(cfg.violation_loss_scale))
    viol_persist_dyn1_rmse = tensor_rmse(viol_base_dyn1, viol_tgt_dyn1)
    viol_persist_dyn1_nrmse = normalized_rmse(viol_base_dyn1, viol_tgt_dyn1, float(cfg.violation_loss_scale))
    viol_rmse_gain_dyn1 = viol_persist_dyn1_rmse - viol_rmse_dyn1
    loss_viol_persist_dyn1 = persistence_surplus_loss(viol_rmse_dyn1, viol_persist_dyn1_rmse, margin=0.0)

    H = min(int(cfg.dyn_roll_horizon), T - 1)
    loss_roll = q.new_zeros(())
    loss_temp_res_roll = q.new_zeros(())
    loss_viol_roll = q.new_zeros(())
    loss_viol_tail = q.new_zeros(())
    loss_temp_persist_roll = q.new_zeros(())
    loss_viol_persist_roll = q.new_zeros(())
    loss_temp_gain_tail = q.new_zeros(())
    loss_temp_gain_consistency = q.new_zeros(())
    loss_temp_roll_bias = q.new_zeros(())
    loss_temp_res_reanchor_roll = q.new_zeros(())
    roll_rmse_sum = q.new_zeros(())
    roll_nrmse_sum = q.new_zeros(())
    temp_roll_rmse_sum = q.new_zeros(())
    temp_roll_nrmse_sum = q.new_zeros(())
    temp_roll_bias_sum = q.new_zeros(())
    temp_reanchor_roll_rmse_sum = q.new_zeros(())
    temp_reanchor_roll_nrmse_sum = q.new_zeros(())
    temp_persist_roll_rmse_sum = q.new_zeros(())
    temp_persist_roll_nrmse_sum = q.new_zeros(())
    temp_rmse_gain_roll_sum = q.new_zeros(())
    viol_roll_rmse_sum = q.new_zeros(())
    viol_roll_nrmse_sum = q.new_zeros(())
    viol_persist_roll_rmse_sum = q.new_zeros(())
    viol_persist_roll_nrmse_sum = q.new_zeros(())
    viol_rmse_gain_roll_sum = q.new_zeros(())
    used_roll = 0
    used_reanchor = 0
    used_roll_weight = q.new_zeros(())
    used_tail_weight = q.new_zeros(())
    used_reanchor_weight = q.new_zeros(())

    horizon_logs = {1: float(dyn_1step_nrmse.detach().cpu())}
    temp_horizon_logs = {1: float(nrmse_temp_dyn1.detach().cpu())}
    temp_rmse_horizon_logs = {1: float(rmse_temp_dyn1.detach().cpu())}
    temp_persist_horizon_logs = {1: float(temp_persist_dyn1_nrmse.detach().cpu())}
    temp_gain_horizon_logs = {1: float((temp_persist_dyn1_nrmse - nrmse_temp_dyn1).detach().cpu())}
    viol_horizon_logs = {1: float(viol_nrmse_dyn1.detach().cpu())}
    viol_rmse_horizon_logs = {1: float(viol_rmse_dyn1.detach().cpu())}
    viol_persist_horizon_logs = {1: float(viol_persist_dyn1_nrmse.detach().cpu())}
    viol_gain_horizon_logs = {1: float((viol_persist_dyn1_nrmse - viol_nrmse_dyn1).detach().cpu())}

    temp_gain_curve = [temp_persist_dyn1_nrmse - nrmse_temp_dyn1]
    viol_gain_curve = [viol_persist_dyn1_nrmse - viol_nrmse_dyn1]

    reanchor_gap = max(1, int(getattr(cfg, "train_reanchor_gap", 3)))
    reanchor_prob = float(getattr(cfg, "train_reanchor_prob", 0.0))
    use_reanchor = bool(reanchor_prob > 0.0 and H > reanchor_gap and torch.rand((), device=q.device).item() < reanchor_prob)
    roll_weights = backloaded_horizon_weights(max(H - 1, 1), device=q.device, dtype=q.dtype)
    tail_weights = backloaded_horizon_weights(max(H - 1, 1), device=q.device, dtype=q.dtype)

    if H >= 2:
        for idx, h in enumerate(range(2, H + 1)):
            w_h = roll_weights[idx]
            tail_w = tail_weights[idx]
            q_hat_open = rollout_latent_for_horizon(q, a, exog_next, dyn_model, h)
            q_hat_open_decoder = rollout_latent_for_horizon(q_decoder, a, exog_next, dyn_model, h)
            q_tgt_h = q[:, h:, :]
            loss_roll = loss_roll + w_h * normalized_huber_loss(q_hat_open, q_tgt_h, latent_scale, delta=huber_delta)
            rmse_h = tensor_rmse(q_hat_open, q_tgt_h)
            nrmse_h = normalized_rmse(q_hat_open, q_tgt_h, latent_scale)
            roll_rmse_sum = roll_rmse_sum + rmse_h
            roll_nrmse_sum = roll_nrmse_sum + nrmse_h
            horizon_logs[h] = float(nrmse_h.detach().cpu())

            temp_tgt_h = temp_next_seq[:, h - 1:T - 1]
            temp_base_h = temp_now[:, :-h]
            exog_target_h = exog_next[:, h - 1:T - 1, :]
            temp_hat_h = decode_temperature_from_q(q_hat_open_decoder, temp_decoder, cfg, temp_base=temp_base_h, exog=exog_target_h)
            temp_res_hat_h = temp_hat_h - temp_base_h
            temp_res_tgt_h = temp_tgt_h - temp_base_h
            viol_hat_h = violation_mag_from_temp(temp_hat_h, comfort_low, comfort_high)
            viol_tgt_h = violation_mag_from_temp(temp_tgt_h, comfort_low, comfort_high)
            viol_base_h = violation_mag_from_temp(temp_base_h, comfort_low, comfort_high)

            loss_temp_res_roll = loss_temp_res_roll + w_h * normalized_huber_loss(temp_res_hat_h, temp_res_tgt_h, float(cfg.temp_residual_scale), delta=huber_delta)
            loss_viol_roll = loss_viol_roll + w_h * normalized_huber_loss(viol_hat_h, viol_tgt_h, float(cfg.violation_loss_scale), delta=huber_delta)
            loss_viol_tail = loss_viol_tail + tail_w * normalized_huber_loss(viol_hat_h, viol_tgt_h, float(cfg.violation_loss_scale), delta=huber_delta)

            temp_rmse_h = tensor_rmse(temp_hat_h, temp_tgt_h)
            temp_nrmse_h = normalized_rmse(temp_hat_h, temp_tgt_h, 10.0)
            temp_bias_h = (temp_hat_h - temp_tgt_h).mean()
            temp_persist_rmse_h = tensor_rmse(temp_base_h, temp_tgt_h)
            temp_persist_nrmse_h = normalized_rmse(temp_base_h, temp_tgt_h, 10.0)
            temp_gain_nrmse_h = temp_persist_nrmse_h - temp_nrmse_h
            loss_temp_persist_roll = loss_temp_persist_roll + tail_w * persistence_surplus_loss(temp_rmse_h, temp_persist_rmse_h, margin=float(cfg.temp_persist_margin))
            loss_temp_gain_tail = loss_temp_gain_tail + tail_w * (temp_nrmse_h - temp_persist_nrmse_h).clamp_min(0.0)
            loss_temp_roll_bias = loss_temp_roll_bias + tail_w * temp_bias_h.abs()

            if use_reanchor and h > reanchor_gap:
                q_hat_reanchor_decoder = rollout_latent_for_horizon_reanchored(q_decoder, a, exog_next, dyn_model, h, reanchor_gap)
                temp_hat_reanchor_h = decode_temperature_from_q(q_hat_reanchor_decoder, temp_decoder, cfg, temp_base=temp_base_h, exog=exog_target_h)
                temp_res_hat_reanchor_h = temp_hat_reanchor_h - temp_base_h
                loss_temp_res_reanchor_roll = loss_temp_res_reanchor_roll + tail_w * normalized_huber_loss(
                    temp_res_hat_reanchor_h,
                    temp_res_tgt_h,
                    float(cfg.temp_residual_scale),
                    delta=huber_delta,
                )
                temp_reanchor_roll_rmse_sum = temp_reanchor_roll_rmse_sum + tensor_rmse(temp_hat_reanchor_h, temp_tgt_h)
                temp_reanchor_roll_nrmse_sum = temp_reanchor_roll_nrmse_sum + normalized_rmse(temp_hat_reanchor_h, temp_tgt_h, 10.0)
                used_reanchor += 1
                used_reanchor_weight = used_reanchor_weight + tail_w

            viol_rmse_h = tensor_rmse(viol_hat_h, viol_tgt_h)
            viol_nrmse_h = normalized_rmse(viol_hat_h, viol_tgt_h, float(cfg.violation_loss_scale))
            viol_persist_rmse_h = tensor_rmse(viol_base_h, viol_tgt_h)
            viol_persist_nrmse_h = normalized_rmse(viol_base_h, viol_tgt_h, float(cfg.violation_loss_scale))
            loss_viol_persist_roll = loss_viol_persist_roll + tail_w * persistence_surplus_loss(viol_rmse_h, viol_persist_rmse_h, margin=0.0)

            temp_roll_rmse_sum = temp_roll_rmse_sum + temp_rmse_h
            temp_roll_nrmse_sum = temp_roll_nrmse_sum + temp_nrmse_h
            temp_roll_bias_sum = temp_roll_bias_sum + temp_bias_h
            temp_persist_roll_rmse_sum = temp_persist_roll_rmse_sum + temp_persist_rmse_h
            temp_persist_roll_nrmse_sum = temp_persist_roll_nrmse_sum + temp_persist_nrmse_h
            temp_rmse_gain_roll_sum = temp_rmse_gain_roll_sum + (temp_persist_rmse_h - temp_rmse_h)
            viol_roll_rmse_sum = viol_roll_rmse_sum + viol_rmse_h
            viol_roll_nrmse_sum = viol_roll_nrmse_sum + viol_nrmse_h
            viol_persist_roll_rmse_sum = viol_persist_roll_rmse_sum + viol_persist_rmse_h
            viol_persist_roll_nrmse_sum = viol_persist_roll_nrmse_sum + viol_persist_nrmse_h
            viol_rmse_gain_roll_sum = viol_rmse_gain_roll_sum + (viol_persist_rmse_h - viol_rmse_h)

            temp_horizon_logs[h] = float(temp_nrmse_h.detach().cpu())
            temp_rmse_horizon_logs[h] = float(temp_rmse_h.detach().cpu())
            temp_persist_horizon_logs[h] = float(temp_persist_nrmse_h.detach().cpu())
            temp_gain_horizon_logs[h] = float(temp_gain_nrmse_h.detach().cpu())
            viol_horizon_logs[h] = float(viol_nrmse_h.detach().cpu())
            viol_rmse_horizon_logs[h] = float(viol_rmse_h.detach().cpu())
            viol_persist_horizon_logs[h] = float(viol_persist_nrmse_h.detach().cpu())
            viol_gain_horizon_logs[h] = float((viol_persist_nrmse_h - viol_nrmse_h).detach().cpu())
            temp_gain_curve.append(temp_gain_nrmse_h)
            viol_gain_curve.append(viol_persist_nrmse_h - viol_nrmse_h)

            used_roll += 1
            used_roll_weight = used_roll_weight + w_h
            used_tail_weight = used_tail_weight + tail_w

        norm_w = used_roll_weight.clamp_min(1.0e-8)
        tail_norm_w = used_tail_weight.clamp_min(1.0e-8)
        loss_roll = loss_roll / norm_w
        loss_temp_res_roll = loss_temp_res_roll / norm_w
        loss_viol_roll = loss_viol_roll / norm_w
        loss_viol_tail = loss_viol_tail / tail_norm_w
        loss_temp_persist_roll = loss_temp_persist_roll / tail_norm_w
        loss_viol_persist_roll = loss_viol_persist_roll / tail_norm_w
        loss_temp_gain_tail = loss_temp_gain_tail / tail_norm_w
        loss_temp_roll_bias = loss_temp_roll_bias / tail_norm_w

        if used_reanchor > 0:
            reanchor_norm_w = used_reanchor_weight.clamp_min(1.0e-8)
            loss_temp_res_reanchor_roll = loss_temp_res_reanchor_roll / reanchor_norm_w
            temp_reanchor_roll_rmse = temp_reanchor_roll_rmse_sum / used_reanchor
            temp_reanchor_roll_nrmse = temp_reanchor_roll_nrmse_sum / used_reanchor
        else:
            loss_temp_res_reanchor_roll = q.new_zeros(())
            temp_reanchor_roll_rmse = q.new_zeros(())
            temp_reanchor_roll_nrmse = q.new_zeros(())
        dyn_roll_rmse = roll_rmse_sum / used_roll
        dyn_roll_nrmse = roll_nrmse_sum / used_roll
        temp_roll_rmse = temp_roll_rmse_sum / used_roll
        temp_roll_nrmse = temp_roll_nrmse_sum / used_roll
        temp_roll_bias = temp_roll_bias_sum / used_roll
        temp_persist_roll_rmse = temp_persist_roll_rmse_sum / used_roll
        temp_persist_roll_nrmse = temp_persist_roll_nrmse_sum / used_roll
        temp_rmse_gain_roll = temp_rmse_gain_roll_sum / used_roll
        viol_roll_rmse = viol_roll_rmse_sum / used_roll
        viol_roll_nrmse = viol_roll_nrmse_sum / used_roll
        viol_persist_roll_rmse = viol_persist_roll_rmse_sum / used_roll
        viol_persistence_roll_nrmse = viol_persist_roll_nrmse_sum / used_roll
        viol_rmse_gain_roll = viol_rmse_gain_roll_sum / used_roll
        temp_gain_curve_t = torch.stack(temp_gain_curve, dim=0)
        viol_gain_curve_t = torch.stack(viol_gain_curve, dim=0)
        temp_gain_nrmse_last_h = temp_gain_curve_t[-1]
        temp_gain_nrmse_min_h = temp_gain_curve_t.min()
        if temp_gain_curve_t.numel() > 1:
            gain_drop = (temp_gain_curve_t[:-1] - temp_gain_curve_t[1:]).clamp_min(0.0)
            gain_drop_weights = backloaded_horizon_weights(int(gain_drop.shape[0]), device=q.device, dtype=q.dtype)
            loss_temp_gain_consistency = (gain_drop * gain_drop_weights).sum() / gain_drop_weights.sum().clamp_min(1.0e-8)
        else:
            loss_temp_gain_consistency = q.new_zeros(())
        viol_gain_nrmse_last_h = viol_gain_curve_t[-1]
        viol_gain_nrmse_min_h = viol_gain_curve_t.min()
    else:
        dyn_roll_rmse = q.new_zeros(())
        dyn_roll_nrmse = q.new_zeros(())
        temp_roll_rmse = rmse_temp_dyn1
        temp_roll_nrmse = nrmse_temp_dyn1
        temp_roll_bias = temp_dyn1_bias
        temp_reanchor_roll_rmse = q.new_zeros(())
        temp_reanchor_roll_nrmse = q.new_zeros(())
        temp_persist_roll_rmse = temp_persist_dyn1_rmse
        temp_persist_roll_nrmse = temp_persist_dyn1_nrmse
        temp_rmse_gain_roll = temp_rmse_gain_dyn1
        viol_roll_rmse = viol_rmse_dyn1
        viol_roll_nrmse = viol_nrmse_dyn1
        viol_persist_roll_rmse = viol_persist_dyn1_rmse
        viol_persistence_roll_nrmse = viol_persist_dyn1_nrmse
        viol_rmse_gain_roll = viol_rmse_gain_dyn1
        temp_gain_nrmse_last_h = temp_persist_dyn1_nrmse - nrmse_temp_dyn1
        temp_gain_nrmse_min_h = temp_gain_nrmse_last_h
        viol_gain_nrmse_last_h = viol_persist_dyn1_nrmse - viol_nrmse_dyn1
        viol_gain_nrmse_min_h = viol_gain_nrmse_last_h

    latent_dynamics_scale = 1.0 if backbone_dyn_on else 0.0
    temp_supervision_scale = 1.0 if temp_supervision_on else 0.0
    total = (
        latent_dynamics_scale * (
            float(cfg.latent_dyn1_coef) * loss_1
            + rollout_progress * float(cfg.latent_roll_coef) * loss_roll
        )
        + temp_supervision_scale * (
            float(cfg.temp_residual_dyn1_coef) * loss_temp_res_dyn1
            + rollout_progress * float(cfg.temp_residual_roll_coef) * loss_temp_res_roll
            + float(cfg.viol_dyn1_coef) * loss_viol_dyn1
            + rollout_progress * float(cfg.viol_roll_coef) * loss_viol_roll
            + rollout_progress * float(cfg.viol_tail_coef) * loss_viol_tail
            + float(cfg.temp_persist_dyn1_coef) * loss_temp_persist_dyn1
            + rollout_progress * float(cfg.temp_persist_roll_coef) * loss_temp_persist_roll
            + float(cfg.viol_persist_dyn1_coef) * loss_viol_persist_dyn1
            + rollout_progress * float(cfg.viol_persist_roll_coef) * loss_viol_persist_roll
            + rollout_progress * float(cfg.temp_gain_tail_coef) * loss_temp_gain_tail
            + rollout_progress * float(cfg.temp_gain_consistency_coef) * loss_temp_gain_consistency
            + rollout_progress * float(cfg.temp_roll_bias_coef) * loss_temp_roll_bias
            + rollout_progress * float(cfg.temp_reanchor_roll_coef) * loss_temp_res_reanchor_roll
        )
    )

    logs = _zero_logs()
    logs.update({
        "dyn_1step": float(loss_1.detach().cpu()),
        "dyn_roll": float(loss_roll.detach().cpu()),
        "regime_maxprob": float(alpha_all.max(dim=-1).values.mean().detach().cpu()),
        "drift_norm": float(drift_all.norm(dim=-1).mean().detach().cpu()),
        "ctrl_norm": float(ctrl_all.norm(dim=-1).mean().detach().cpu()),
        "dyn_1step_rmse": float(dyn_1step_rmse.detach().cpu()),
        "dyn_1step_nrmse": float(dyn_1step_nrmse.detach().cpu()),
        "dyn_roll_rmse": float(dyn_roll_rmse.detach().cpu()),
        "dyn_roll_nrmse": float(dyn_roll_nrmse.detach().cpu()),
        "loss_temp_res_dyn1": float(loss_temp_res_dyn1.detach().cpu()),
        "loss_temp_res_roll": float(loss_temp_res_roll.detach().cpu()),
        "loss_viol_dyn1": float(loss_viol_dyn1.detach().cpu()),
        "loss_viol_roll": float(loss_viol_roll.detach().cpu()),
        "loss_viol_tail": float(loss_viol_tail.detach().cpu()),
        "loss_temp_persist_dyn1": float(loss_temp_persist_dyn1.detach().cpu()),
        "loss_temp_persist_roll": float(loss_temp_persist_roll.detach().cpu()),
        "loss_viol_persist_dyn1": float(loss_viol_persist_dyn1.detach().cpu()),
        "loss_viol_persist_roll": float(loss_viol_persist_roll.detach().cpu()),
        "loss_temp_gain_tail": float(loss_temp_gain_tail.detach().cpu()),
        "loss_temp_gain_consistency": float(loss_temp_gain_consistency.detach().cpu()),
        "loss_temp_roll_bias": float(loss_temp_roll_bias.detach().cpu()),
        "loss_temp_res_reanchor_roll": float(loss_temp_res_reanchor_roll.detach().cpu()),
        "reanchor_active": float(1.0 if use_reanchor else 0.0),
        "decoded_temperature_mean": float(temperature_decoded.mean().detach().cpu()),
        "decoded_temperature_std": float(temperature_decoded.std(unbiased=False).detach().cpu()),
        "decoded_temperature_min": float(temperature_decoded.min().detach().cpu()),
        "decoded_temperature_max": float(temperature_decoded.max().detach().cpu()),
        "latent_state_norm": float(q.norm(dim=-1).mean().detach().cpu()),
        "latent_temperature_direct_rmse": float(rmse_temp_direct.detach().cpu()),
        "latent_temperature_direct_nrmse": float(nrmse_temp_direct.detach().cpu()),
        "latent_temperature_direct_bias": float(temp_direct_bias.detach().cpu()),
        "latent_temperature_direct_corr": float(corr_temp_direct.detach().cpu()),
        "latent_temperature_dyn1_rmse": float(rmse_temp_dyn1.detach().cpu()),
        "latent_temperature_dyn1_nrmse": float(nrmse_temp_dyn1.detach().cpu()),
        "latent_temperature_dyn1_bias": float(temp_dyn1_bias.detach().cpu()),
        "latent_temperature_dyn1_corr": float(corr_temp_dyn1.detach().cpu()),
        "latent_temperature_roll_rmse": float(temp_roll_rmse.detach().cpu()),
        "latent_temperature_roll_nrmse": float(temp_roll_nrmse.detach().cpu()),
        "latent_temperature_roll_bias": float(temp_roll_bias.detach().cpu()),
        "latent_temperature_reanchor_roll_rmse": float(temp_reanchor_roll_rmse.detach().cpu()),
        "latent_temperature_reanchor_roll_nrmse": float(temp_reanchor_roll_nrmse.detach().cpu()),
        "temp_persistence_dyn1_rmse": float(temp_persist_dyn1_rmse.detach().cpu()),
        "temp_persistence_dyn1_nrmse": float(temp_persist_dyn1_nrmse.detach().cpu()),
        "temp_persistence_roll_rmse": float(temp_persist_roll_rmse.detach().cpu()),
        "temp_persistence_roll_nrmse": float(temp_persist_roll_nrmse.detach().cpu()),
        "temp_rmse_gain_dyn1": float(temp_rmse_gain_dyn1.detach().cpu()),
        "temp_rmse_gain_roll": float(temp_rmse_gain_roll.detach().cpu()),
        "temp_gain_nrmse_last_h": float(temp_gain_nrmse_last_h.detach().cpu()),
        "temp_gain_nrmse_min_h": float(temp_gain_nrmse_min_h.detach().cpu()),
        "viol_dyn1_rmse": float(viol_rmse_dyn1.detach().cpu()),
        "viol_dyn1_nrmse": float(viol_nrmse_dyn1.detach().cpu()),
        "viol_roll_rmse": float(viol_roll_rmse.detach().cpu()),
        "viol_roll_nrmse": float(viol_roll_nrmse.detach().cpu()),
        "viol_persistence_dyn1_rmse": float(viol_persist_dyn1_rmse.detach().cpu()),
        "viol_persistence_dyn1_nrmse": float(viol_persist_dyn1_nrmse.detach().cpu()),
        "viol_persistence_roll_rmse": float(viol_persist_roll_rmse.detach().cpu()),
        "viol_persistence_roll_nrmse": float(viol_persistence_roll_nrmse.detach().cpu()),
        "viol_rmse_gain_dyn1": float(viol_rmse_gain_dyn1.detach().cpu()),
        "viol_rmse_gain_roll": float(viol_rmse_gain_roll.detach().cpu()),
        "viol_gain_nrmse_last_h": float(viol_gain_nrmse_last_h.detach().cpu()),
        "viol_gain_nrmse_min_h": float(viol_gain_nrmse_min_h.detach().cpu()),
        "latent_constraint_cost_mean": float(latent_cost_now.mean().detach().cpu()),
        "latent_constraint_cost_std": float(latent_cost_now.std(unbiased=False).detach().cpu()),
        "latent_constraint_cost_max": float(latent_cost_now.max().detach().cpu()),
        "latent_constraint_violation_rate": float(latent_hard_now.mean().detach().cpu()),
        "temp_vs_persist_dyn1_surplus": float((rmse_temp_dyn1 - temp_persist_dyn1_rmse).detach().cpu()),
        "temp_vs_persist_roll_surplus": float((temp_roll_rmse - temp_persist_roll_rmse).detach().cpu()),
        "temp_beats_persist_dyn1": float((nrmse_temp_dyn1 < temp_persist_dyn1_nrmse).float().detach().cpu()),
        "temp_beats_persist_roll": float((temp_roll_nrmse < temp_persist_roll_nrmse).float().detach().cpu()),
        "viol_vs_persist_dyn1_surplus": float((viol_rmse_dyn1 - viol_persist_dyn1_rmse).detach().cpu()),
        "viol_vs_persist_roll_surplus": float((viol_roll_rmse - viol_persist_roll_rmse).detach().cpu()),
        "viol_beats_persist_dyn1": float((viol_nrmse_dyn1 < viol_persist_dyn1_nrmse).float().detach().cpu()),
        "viol_beats_persist_roll": float((viol_roll_nrmse < viol_persistence_roll_nrmse).float().detach().cpu()),
    })

    for h in range(1, H + 1):
        logs[f"dyn_latent_nrmse_h{h:02d}"] = horizon_logs[h]
        logs[f"latent_temperature_nrmse_h{h:02d}"] = temp_horizon_logs[h]
        logs[f"latent_temperature_rmse_h{h:02d}"] = temp_rmse_horizon_logs[h]
        logs[f"latent_temperature_persist_nrmse_h{h:02d}"] = temp_persist_horizon_logs[h]
        logs[f"latent_temperature_gain_nrmse_h{h:02d}"] = temp_gain_horizon_logs[h]
        logs[f"latent_violation_nrmse_h{h:02d}"] = viol_horizon_logs[h]
        logs[f"latent_violation_rmse_h{h:02d}"] = viol_rmse_horizon_logs[h]
        logs[f"latent_violation_persist_nrmse_h{h:02d}"] = viol_persist_horizon_logs[h]
        logs[f"latent_violation_gain_nrmse_h{h:02d}"] = viol_gain_horizon_logs[h]

    if not temp_supervision_on:
        zero_logs = _zero_logs()
        for k in list(logs.keys()):
            if k in zero_logs and k not in {"dyn_rollout_progress", "dynamics_aux_enabled", "dynamics_aux_enable_after_batch", "temp_supervision_enabled", "temp_supervision_enable_after_batch", "backbone_dynamics_enabled", "backbone_dynamics_enable_after_batch", "latent_dynamics_loss_enabled", "dyn_1step", "dyn_roll", "regime_maxprob", "drift_norm", "ctrl_norm", "dyn_1step_rmse", "dyn_1step_nrmse", "dyn_roll_rmse", "dyn_roll_nrmse"}:
                logs[k] = zero_logs[k]
    return total, logs


def make_feature_module(selected_indices: list[int], shift: torch.Tensor, scale: torch.Tensor, clip_value: float, hidden: int, device):
    return TensorDictModule(
        ObservationFeatureExtractor(selected_indices, shift.to(device), scale.to(device), clip_value=clip_value, hidden=hidden).to(device),
        in_keys=["observation"],
        out_keys=["model_observation", "observation_normalized", "actor_embed"],
    ).to(device)


def make_lstm_module(embed_key, h_key, c_key, is_init_key, out_key, next_h_key, next_c_key, in_dim, hid_dim, device):
    return LSTMModule(
        input_size=in_dim,
        hidden_size=hid_dim,
        in_keys=[embed_key, h_key, c_key, is_init_key],
        out_keys=[out_key, next_h_key, next_c_key],
    ).to(device)


def make_rollbuf_module(lstm_out_key, window_key, is_init_key, window_seq_key, next_window_key, window_size, hidden_size, device):
    return TensorDictModule(
        RollingWindowBuffer(window_size, hidden_size).to(device),
        in_keys=[lstm_out_key, window_key, is_init_key],
        out_keys=[window_seq_key, next_window_key],
    ).to(device)


def make_xfm_module(window_seq_key, xfm_latent_key, hidden_size, nhead, layers, ff_dim, dropout, device):
    return TensorDictModule(
        WindowTransformer(hidden_size, nhead, layers, ff_dim, dropout).to(device),
        in_keys=[window_seq_key],
        out_keys=[xfm_latent_key],
    ).to(device)


def save_checkpoint(*, run_dir: str, policy, reward_critic, cost_critic, dynamics_model=None, temperature_decoder=None, actor_optim=None, dynamics_optim=None, reward_critic_optim=None, cost_critic_optim=None, lag_lambda=None, frames=None, batch_idx=None, term_counts=None, running_ep_steps=None, cfg=None):
    os.makedirs(run_dir, exist_ok=True)
    run_dir = Path(run_dir)
    torch.save(policy.state_dict(), run_dir / "actor_policy_state_dict.pt")
    torch.save(reward_critic.state_dict(), run_dir / "reward_critic_state_dict.pt")
    torch.save(cost_critic.state_dict(), run_dir / "cost_critic_state_dict.pt")
    if dynamics_model is not None:
        torch.save(dynamics_model.state_dict(), run_dir / "switching_dynamics_state_dict.pt")
    if temperature_decoder is not None:
        torch.save(temperature_decoder.state_dict(), run_dir / "temperature_decoder_state_dict.pt")
    config_payload = None
    if cfg is not None:
        if hasattr(cfg, "to_dict"):
            config_payload = cfg.to_dict()
        elif hasattr(cfg, "__dict__"):
            config_payload = copy.deepcopy(cfg.__dict__)
    ckpt = {
        "actor_optim": actor_optim.state_dict() if actor_optim is not None else None,
        "dynamics_optim": dynamics_optim.state_dict() if dynamics_optim is not None else None,
        "reward_critic_optim": reward_critic_optim.state_dict() if reward_critic_optim is not None else None,
        "cost_critic_optim": cost_critic_optim.state_dict() if cost_critic_optim is not None else None,
        "lag_lambda": float(lag_lambda.detach().cpu().item()) if isinstance(lag_lambda, torch.Tensor) else lag_lambda,
        "frames": int(frames) if frames is not None else None,
        "batch_idx": int(batch_idx) if batch_idx is not None else None,
        "term_counts": dict(term_counts) if term_counts is not None else None,
        "running_ep_steps": running_ep_steps.tolist() if isinstance(running_ep_steps, np.ndarray) else running_ep_steps,
        "config": config_payload,
    }
    torch.save(ckpt, run_dir / "train_state.pt")
    print(f"[save] checkpoint written to: {run_dir.resolve()}")


class MovingWindowSinergymEnv(gym.Env):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.env = None
        self.current_runperiod = None
        self._rng = random.Random(int.from_bytes(os.urandom(16), "big"))
        self._orig_cwd = os.getcwd()
        self._worker_output_dir: Path | None = None

        if bool(getattr(self.cfg, "sinergym_isolate_output_dirs", True)):
            root = Path(getattr(self.cfg, "sinergym_output_root", None) or tempfile.gettempdir()).expanduser()
            # In ParallelEnv each worker process normally owns one env. The PID + object id
            # makes the sandbox unique enough even when many workers are spawned.
            safe_env_id = str(getattr(self.cfg, "sinergym_env_id", "sinergym")).replace(os.sep, "_")
            self._worker_output_dir = root / f"{safe_env_id}_pid{os.getpid()}_{id(self)}"
            self._worker_output_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(self._worker_output_dir)

        self._rebuild_env(self._select_runperiod())

    def _cleanup_worker_outputs(self):
        if not bool(getattr(self.cfg, "sinergym_isolate_output_dirs", True)):
            return
        root = self._worker_output_dir
        if root is None or not root.exists():
            return
        # Keep the worker sandbox itself as the current working directory; remove only
        # EnergyPlus/Sinergym outputs inside it. This avoids polluting the project folder
        # while still allowing randomized runperiod rebuilds.
        for child in list(root.iterdir()):
            _safe_remove_path(child)

    def _select_runperiod(self) -> tuple[int, int, int, int, int, int]:
        if bool(getattr(self.cfg, "sinergym_randomize_runperiod", False)):
            return sample_moving_window_runperiod(self.cfg, rng=self._rng)

        building_config = copy.deepcopy(self.cfg.sinergym_building_config)
        runperiod = building_config.get("runperiod")
        if runperiod:
            return tuple(runperiod)
        return default_runperiod_for_season(self.cfg)

    def _make_raw_env(self, runperiod: tuple[int, int, int, int, int, int]):
        building_config = copy.deepcopy(self.cfg.sinergym_building_config)
        building_config["runperiod"] = tuple(runperiod)
        if "timesteps_per_hour" not in building_config:
            building_config["timesteps_per_hour"] = int(self.cfg.timesteps_per_hour)

        reward_kwargs = copy.deepcopy(self.cfg.sinergym_reward_kwargs)
        make_kwargs = {
            "reward_kwargs": reward_kwargs,
            "building_config": building_config,
        }
        max_ep_store = getattr(self.cfg, "sinergym_max_ep_store", 1)
        if max_ep_store is not None:
            make_kwargs["max_ep_store"] = int(max_ep_store)
        if self.cfg.sinergym_weather_file is not None:
            make_kwargs["weather_file"] = copy.deepcopy(self.cfg.sinergym_weather_file)
        if self.cfg.sinergym_weather_variability:
            make_kwargs["weather_variability"] = copy.deepcopy(self.cfg.sinergym_weather_variability)

        with suppress_stdout_stderr(bool(self.cfg.suppress_env_stdout_stderr)):
            return gym.make(self.cfg.sinergym_env_id, **make_kwargs)

    def _rebuild_env(self, runperiod: tuple[int, int, int, int, int, int]):
        old_env = getattr(self, "env", None)
        if old_env is not None:
            try:
                old_env.close()
            except Exception:
                pass

        if bool(getattr(self.cfg, "sinergym_cleanup_output_on_rebuild", True)):
            self._cleanup_worker_outputs()

        self.env = self._make_raw_env(runperiod)
        self.current_runperiod = tuple(runperiod)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = getattr(self.env, "metadata", {})

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            try:
                self._rng.seed(int(seed))
            except Exception:
                pass

        if bool(getattr(self.cfg, "sinergym_randomize_runperiod_on_reset", True)) or self.env is None:
            self._rebuild_env(self._select_runperiod())

        with suppress_stdout_stderr(bool(self.cfg.suppress_env_stdout_stderr)):
            out = self.env.reset(seed=seed, options=options)

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        info = dict(info)
        if self.current_runperiod is not None:
            info["runperiod"] = self.current_runperiod
            info["runperiod_start_month"] = int(self.current_runperiod[0])
            info["runperiod_start_day"] = int(self.current_runperiod[1])
            info["runperiod_start_year"] = int(self.current_runperiod[2])
            info["runperiod_end_day"] = int(self.current_runperiod[3])
            info["runperiod_end_month"] = int(self.current_runperiod[4])
            info["runperiod_end_year"] = int(self.current_runperiod[5])
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def render(self):
        if self.env is None:
            return None
        return self.env.render()

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None
        if bool(getattr(self.cfg, "sinergym_cleanup_output_on_close", True)):
            self._cleanup_worker_outputs()
        try:
            if getattr(self, "_orig_cwd", None):
                os.chdir(self._orig_cwd)
        except Exception:
            pass

    def __getattr__(self, name):
        env = self.__dict__.get("env", None)
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)


class SinergymEnergyComfortWrapper(gym.Wrapper):
    def __init__(self, env, cfg: Config):
        super().__init__(env)
        self.cfg = cfg
        self.obs_idx = obs_index_map_from_cfg(cfg)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.prev_action = None
        self.action_low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        self.action_range = np.maximum(self.action_high - self.action_low, 1e-6)

    def _obs_value(self, obs, key: str, default: float = 0.0) -> float:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        idx = self.obs_idx.get(key, None)
        if idx is None or idx >= arr.size:
            return float(default)
        return float(arr[idx])

    def _comfort_band(self, info: dict) -> tuple[float, float]:
        month = info.get("month", None)
        day_of_month = info.get("day_of_month", None)
        return active_comfort_band_from_cfg(self.cfg, month=month, day_of_month=day_of_month)

    def _compute_smoothness_terms(self, action) -> tuple[float, float, float, float]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)

        if self.prev_action is None or self.prev_action.shape != a.shape:
            delta = np.zeros_like(a, dtype=np.float32)
        else:
            delta = a - self.prev_action

        if bool(self.cfg.smoothness_normalize_by_action_range):
            delta_used = delta / self.action_range
        else:
            delta_used = delta

        delta_l1 = float(np.mean(np.abs(delta_used)))
        delta_l2 = float(np.mean(np.square(delta_used)))

        mode = str(self.cfg.smoothness_mode).lower()
        if mode == "l1":
            smoothness_raw = delta_l1
        elif mode == "l2":
            smoothness_raw = delta_l2
        else:
            raise ValueError(f"Unsupported smoothness_mode={self.cfg.smoothness_mode!r}")

        smoothness_penalty = float(self.cfg.smoothness_coef) * smoothness_raw
        return smoothness_penalty, smoothness_raw, delta_l1, delta_l2

    def _augment_info(self, info: dict, obs=None, reward=None, smoothness_penalty: float = 0.0, smoothness_raw: float = 0.0, action_delta_l1: float = 0.0, action_delta_l2: float = 0.0) -> dict:
        info = dict(info)
        runperiod = getattr(self.env, "current_runperiod", None)
        if runperiod is not None:
            info["runperiod_start_month"] = np.float32(int(runperiod[0]))
            info["runperiod_start_day"] = np.float32(int(runperiod[1]))
            info["runperiod_start_year"] = np.float32(int(runperiod[2]))
            info["runperiod_end_day"] = np.float32(int(runperiod[3]))
            info["runperiod_end_month"] = np.float32(int(runperiod[4]))
            info["runperiod_end_year"] = np.float32(int(runperiod[5]))

        comfort_low, comfort_high = self._comfort_band(info)
        air_temperature = self._obs_value(obs, self.cfg.sinergym_temperature_variable)

        hvac_power_obs = self._obs_value(obs, self.cfg.sinergym_power_variable)
        total_power_demand = float(info.get("total_power_demand", hvac_power_obs))

        heating_setpoint = self._obs_value(obs, self.cfg.sinergym_heating_setpoint_variable)
        cooling_setpoint = self._obs_value(obs, self.cfg.sinergym_cooling_setpoint_variable)

        lower_gap = comfort_low - air_temperature
        upper_gap = air_temperature - comfort_high
        violation_mag = max(lower_gap, upper_gap, 0.0)
        hard_violation = float(violation_mag > 0.0)

        raw_power_w = total_power_demand
        step_hours = 1.0 / self.cfg.timesteps_per_hour
        energy_kwh = (raw_power_w / 1000.0) * step_hours
        energy_reward = -energy_kwh

        comfort_reward = float(info.get("comfort_penalty", -violation_mag))
        native_reward = float(reward if reward is not None else energy_reward + comfort_reward)

        training_reward = energy_reward - float(smoothness_penalty)

        info["native_reward"] = np.float32(native_reward)
        info["energy_reward"] = np.float32(energy_reward)
        info["training_reward"] = np.float32(training_reward)

        info["smoothness_penalty"] = np.float32(smoothness_penalty)
        info["smoothness_raw"] = np.float32(smoothness_raw)
        info["action_delta_l1"] = np.float32(action_delta_l1)
        info["action_delta_l2"] = np.float32(action_delta_l2)

        info["comfort_reward"] = np.float32(comfort_reward)
        info["energy_penalty"] = np.float32(float(info.get("energy_penalty", energy_reward)))
        info["comfort_penalty"] = np.float32(float(info.get("comfort_penalty", comfort_reward)))
        info["energy_term"] = np.float32(float(info.get("energy_term", energy_reward)))
        info["comfort_term"] = np.float32(float(info.get("comfort_term", comfort_reward)))
        info["total_power_demand"] = np.float32(total_power_demand)
        info["total_temperature_violation"] = np.float32(float(info.get("total_temperature_violation", violation_mag)))
        info["constraint_violation"] = np.float32(hard_violation)
        info["env_constraint_violation"] = np.float32(hard_violation)
        info["constraint_margin"] = np.float32(violation_mag)
        info["env_constraint_margin"] = np.float32(violation_mag)
        info["constraint_cost"] = np.float32(violation_mag)
        info["constraint_active"] = np.float32(hard_violation)
        info["constraint_violation_magnitude"] = np.float32(violation_mag)
        info["air_temperature"] = np.float32(air_temperature)
        info["hvac_power"] = np.float32(hvac_power_obs)
        info["comfort_low"] = np.float32(comfort_low)
        info["comfort_high"] = np.float32(comfort_high)
        info["heating_setpoint"] = np.float32(heating_setpoint)
        info["cooling_setpoint"] = np.float32(cooling_setpoint)
        info["term_code"] = np.float32(1 if bool(info.get("truncated", False)) else 0)
        return info

    def reset(self, **kwargs):
        self.prev_action = None
        with suppress_stdout_stderr(bool(self.cfg.suppress_env_stdout_stderr)):
            out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return obs, self._augment_info(info, obs, reward=None)

    def step(self, action):
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        smoothness_penalty, smoothness_raw, action_delta_l1, action_delta_l2 = self._compute_smoothness_terms(action_np)

        with suppress_stdout_stderr(bool(self.cfg.suppress_env_stdout_stderr)):
            obs, reward, terminated, truncated, info = self.env.step(action)

        info["truncated"] = bool(truncated)
        info = self._augment_info(
            info,
            obs,
            reward=reward,
            smoothness_penalty=smoothness_penalty,
            smoothness_raw=smoothness_raw,
            action_delta_l1=action_delta_l1,
            action_delta_l2=action_delta_l2,
        )

        self.prev_action = action_np.copy()
        training_reward = float(info["training_reward"])
        return obs, training_reward, bool(terminated), bool(truncated), info


def build_env(cfg: Config):
    configure_sinergym_verbosity(cfg)
    raw_env = MovingWindowSinergymEnv(cfg)
    return SinergymEnergyComfortWrapper(raw_env, cfg)


def main():
    cfg = Config()
    configure_sinergym_verbosity(cfg)

    cfg.model_observation_indices = resolve_model_observation_indices(cfg)
    cfg.dynamics_exog_indices = resolve_dynamics_exog_indices(cfg)
    cfg.dynamics_exog_dim = len(cfg.dynamics_exog_indices)
    cfg.forecast_index_map = infer_forecast_index_map(cfg, horizon=max(int(cfg.dyn_roll_horizon), int(cfg.shield_horizon)))
    cfg.air_temperature_obs_idx = require_obs_index_in_keys(
        cfg.model_observation_keys,
        cfg.sinergym_temperature_variable,
        label="cfg.model_observation_keys",
    )

    print("model_observation_keys =", cfg.model_observation_keys)
    print("model_observation_indices =", cfg.model_observation_indices)
    print("dynamics_exog_keys =", [cfg.model_observation_keys[i] for i in cfg.dynamics_exog_indices])
    print("forecast_index_map =", cfg.forecast_index_map)
    print("air_temperature_obs_idx =", cfg.air_temperature_obs_idx)
    print("runperiod_mode =", describe_runperiod_sampling(cfg))

    os.makedirs(cfg.save_dir, exist_ok=True)
    run_dir = os.path.join(cfg.save_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    env_device = torch.device("cpu")
    storing_device = device if (device.type == "cuda" and bool(cfg.collector_storing_on_gpu)) else env_device

    if bool(cfg.show_startup_prints):
        print(f"policy/train device: {device}")
        print(f"env device: {env_device}")
        print(f"collector storing device: {storing_device}")
        print(f"num_envs: {cfg.num_envs}")
        print(f"model_observation_keys: {cfg.model_observation_keys}")

    INFO_KEYS = [
        "term_code",
        "constraint_violation",
        "env_constraint_violation",
        "constraint_margin",
        "env_constraint_margin",
        "constraint_cost",
        "constraint_active",
        "constraint_violation_magnitude",
        "air_temperature",
        "hvac_power",
        "comfort_low",
        "comfort_high",
        "energy_reward",
        "comfort_reward",
        "native_reward",
        "energy_penalty",
        "comfort_penalty",
        "energy_term",
        "comfort_term",
        "total_power_demand",
        "total_temperature_violation",
        "heating_setpoint",
        "cooling_setpoint",
        "training_reward",
        "runperiod_start_month",
        "runperiod_start_day",
        "runperiod_start_year",
        "runperiod_end_day",
        "runperiod_end_month",
        "runperiod_end_year",
        "smoothness_penalty",
        "smoothness_raw",
        "action_delta_l1",
        "action_delta_l2",
    ]

    prototype_gym_env = build_env(cfg)
    LOW = torch.as_tensor(np.asarray(prototype_gym_env.action_space.low, dtype=np.float32), device=device)
    HIGH = torch.as_tensor(np.asarray(prototype_gym_env.action_space.high, dtype=np.float32), device=device)
    prototype_gym_env.close()

    def make_env():
        gym_env = build_env(cfg)
        base = GymWrapper(gym_env, device=env_device)
        base = base.auto_register_info_dict(info_dict_reader=default_info_dict_reader(INFO_KEYS))
        return TransformedEnv(
            base,
            Compose(
                DoubleToFloat(),
                StepCounter(),
                InitTracker(),
                TensorDictPrimer(
                    actor_h=torchrl.data.Unbounded((1, cfg.actor_lstm_hidden), device=env_device),
                    actor_c=torchrl.data.Unbounded((1, cfg.actor_lstm_hidden), device=env_device),
                    actor_window=torchrl.data.Unbounded((cfg.actor_window_size, cfg.actor_lstm_hidden), device=env_device),
                ),
            ),
        )

    if bool(cfg.check_env_specs_on_single_env):
        with suppress_stdout_stderr(bool(cfg.suppress_env_stdout_stderr)):
            env_probe = make_env()
            check_env_specs(env_probe)
            env_probe.close()

    with suppress_stdout_stderr(bool(cfg.suppress_env_stdout_stderr)):
        if bool(cfg.use_parallel_env) and int(cfg.num_envs) > 1:
            env = ParallelEnv(cfg.num_envs, make_env, mp_start_method=str(cfg.mp_start_method))
        else:
            env = SerialEnv(max(1, int(cfg.num_envs)), make_env)

    obs_dim = int(env.observation_spec["observation"].shape[-1])
    model_obs_dim = len(cfg.model_observation_keys)
    act_dim = int(env.action_spec.shape[-1])
    if bool(cfg.show_startup_prints):
        print(f"env_obs_dim: {obs_dim} | model_obs_dim: {model_obs_dim} | act_dim: {act_dim}")

    obs_shift, obs_scale = observation_affine_tensors_from_cfg(cfg)

    actor_feature = make_feature_module(
        cfg.model_observation_indices,
        obs_shift,
        obs_scale,
        clip_value=float(cfg.observation_norm_clip),
        hidden=cfg.actor_num_cells,
        device=device,
    )
    actor_lstm = make_lstm_module(
        "actor_embed", "actor_h", "actor_c", "is_init",
        "actor_lstm_out", "next_actor_h", "next_actor_c",
        cfg.actor_num_cells, cfg.actor_lstm_hidden, device,
    )
    actor_rollbuf = make_rollbuf_module(
        "actor_lstm_out", "actor_window", "is_init",
        "actor_window_seq", "next_actor_window",
        cfg.actor_window_size, cfg.actor_lstm_hidden, device,
    )
    actor_xfm = make_xfm_module(
        "actor_window_seq", "actor_xfm_latent",
        cfg.actor_lstm_hidden,
        cfg.actor_xfm_nhead,
        cfg.actor_xfm_layers,
        cfg.actor_xfm_ff_dim,
        cfg.actor_xfm_dropout,
        device,
    )

    state_proj_core = LatentStateProjector(
        cfg.actor_lstm_hidden,
        cfg.dyn_state_dim,
        hidden=cfg.dyn_hidden,
    ).to(device)
    temp_decoder = LatentTemperatureDecoder(
        state_dim=cfg.dyn_state_dim,
        hidden=cfg.dyn_hidden,
        exog_dim=int(cfg.dynamics_exog_dim),
    ).to(device)

    actor_state_proj = TensorDictModule(
        state_proj_core,
        in_keys=["actor_xfm_latent", "model_observation"],
        out_keys=["actor_dyn_state"],
    ).to(device)

    dyn_model = SwitchingLatentDynamics(
        state_dim=cfg.dyn_state_dim,
        act_dim=act_dim,
        num_regimes=cfg.dyn_num_regimes,
        hidden=cfg.dyn_hidden,
        exog_dim=int(cfg.dynamics_exog_dim),
    ).to(device)

    actor_dyn_summary = TensorDictModule(
        DynamicsSummaryModule(dyn_model, cfg).to(device),
        in_keys=["actor_dyn_state", "model_observation"],
        out_keys=["actor_dyn_logits", "actor_dyn_alpha", "actor_dyn_drift"],
    ).to(device)

    actor_policy_conditioner = TensorDictModule(
        PolicyConditioner(cfg).to(device),
        in_keys=["actor_dyn_state", "actor_dyn_alpha", "actor_dyn_drift"],
        out_keys=["actor_policy_feat"],
    ).to(device)

    actor_carry_hc = TensorDictModule(
        lambda next_actor_h, next_actor_c: (next_actor_h, next_actor_c),
        in_keys=["next_actor_h", "next_actor_c"],
        out_keys=["actor_h", "actor_c"],
    ).to(device)

    actor_carry_win = TensorDictModule(
        lambda next_actor_window: (next_actor_window,),
        in_keys=["next_actor_window"],
        out_keys=["actor_window"],
    ).to(device)

    policy_in_dim = 2 * cfg.dyn_state_dim + cfg.dyn_num_regimes
    actor_policy_core = ActorLocScaleHead(policy_in_dim, cfg.actor_num_cells, act_dim).to(device)
    actor_head = TensorDictModule(
        actor_policy_core,
        in_keys=["actor_policy_feat"],
        out_keys=["loc", "scale"],
    ).to(device)
    clamp_loc_scale = TensorDictModule(
        ClampLocScale(min_scale=0.05, max_scale=2.0, loc_max=5.0),
        in_keys=["loc", "scale"],
        out_keys=["loc", "scale"],
    ).to(device)

    actor_trunk = TensorDictSequential(
        actor_feature,
        actor_lstm,
        actor_rollbuf,
        actor_xfm,
        actor_state_proj,
        actor_dyn_summary,
        actor_policy_conditioner,
        actor_carry_hc,
        actor_carry_win,
        actor_head,
        clamp_loc_scale,
    ).to(device)

    policy = ShieldedLatentPolicyActor(
        actor_trunk=actor_trunk,
        actor_policy_core=actor_policy_core,
        dyn_model=dyn_model,
        temp_decoder=temp_decoder,
        low=LOW,
        high=HIGH,
        cfg=cfg,
    ).to(device)

    reward_critic = NormalizedCritic(
        cfg.model_observation_indices,
        obs_shift.to(device),
        obs_scale.to(device),
        clip_value=float(cfg.observation_norm_clip),
        obs_dim=model_obs_dim,
    ).to(device)
    cost_critic = NormalizedCritic(
        cfg.model_observation_indices,
        obs_shift.to(device),
        obs_scale.to(device),
        clip_value=float(cfg.observation_norm_clip),
        obs_dim=model_obs_dim,
    ).to(device)

    model_backbone_params = unique_params(actor_feature, actor_lstm, actor_rollbuf, actor_xfm)
    projector_params = unique_params(state_proj_core)
    dynamics_model_params = unique_params(dyn_model)
    temp_decoder_params = unique_params(temp_decoder)
    dynamics_params = unique_params(actor_feature, actor_lstm, actor_rollbuf, actor_xfm, state_proj_core, temp_decoder, dyn_model)
    actor_params = unique_params(policy)

    actor_optim = torch.optim.Adam(actor_params, lr=cfg.actor_lr)
    dynamics_optim = torch.optim.Adam([
        {"params": model_backbone_params, "lr": cfg.backbone_dyn_lr},
        {"params": projector_params, "lr": cfg.projector_lr},
        {"params": dynamics_model_params, "lr": cfg.dynamics_lr},
        {"params": temp_decoder_params, "lr": cfg.dynamics_lr},
    ])
    reward_critic_optim = torch.optim.Adam(reward_critic.parameters(), lr=cfg.reward_critic_lr)
    cost_critic_optim = torch.optim.Adam(cost_critic.parameters(), lr=cfg.cost_critic_lr)
    lag_lambda = torch.tensor(float(cfg.lambda_init), device=device)

    with torch.no_grad():
        td0 = env.reset().to(device)
        _ = policy(td0.clone())
        _ = reward_critic(td0["observation"].float())
        _ = cost_critic(td0["observation"].float())
        td0_probe = td0.clone()
        actor_trunk(td0_probe)
        _ = dyn_model.regime_probs(td0_probe["actor_dyn_state"].float(), extract_dynamics_exog_tensor(td0_probe["model_observation"].float(), cfg, normalize=True))
        _ = temp_decoder(td0_probe["actor_dyn_state"].float(), extract_dynamics_exog_tensor(td0_probe["model_observation"].float(), cfg, normalize=True))

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        split_trajs=False,
        policy_device=device,
        env_device=env_device,
        storing_device=storing_device,
    )

    pbar = NoOpPbar() if bool(cfg.sinergym_disable_progress_bar) else tqdm_pkg.tqdm(total=cfg.total_frames, bar_format="{desc}", disable=False)

    def prepare_recurrent_minibatch(sub_td, *, time_dim: int):
        return strip_recurrent_to_t0(sub_td, ["actor_h", "actor_c", "actor_window"], time_dim=time_dim)

    def eval_new_logp_and_entropy(sub_td):
        loc, scale, a = sub_td["loc"], sub_td["scale"], sub_td["action"]
        require_finite(loc, "actor loc")
        require_finite(scale, "actor scale")
        require_finite(a, "stored action")

        scale_clamped = torch.clamp(scale, 0.05, 2.0)
        dist = TanhNormal(loc, scale_clamped, low=LOW, high=HIGH)

        logp = dist.log_prob(a)
        if logp.ndim == a.ndim:
            logp = logp.sum(-1, keepdim=True)
        elif logp.ndim == a.ndim - 1:
            logp = logp.unsqueeze(-1)
        require_finite(logp, "new_logp")

        ent_per_dim = 0.5 * (1.0 + math.log(2.0 * math.pi)) + torch.log(scale_clamped)
        ent = ent_per_dim.sum(dim=-1, keepdim=True).mean()
        require_finite(ent, "gaussian_entropy")

        return logp, ent

    def ppo_surrogate_terms(sub_td, adv, old_logp):
        time_dim_local = get_time_dim(sub_td)
        new_logp, ent = eval_new_logp_and_entropy(sub_td)

        if new_logp.ndim == 2:
            new_logp = new_logp.unsqueeze(-1)
        if old_logp.ndim == 2:
            old_logp = old_logp.unsqueeze(-1)
        if adv.ndim == 2:
            adv = adv.unsqueeze(-1)

        new_logp = _move_time_last(new_logp, time_dim_local)
        old_logp = _move_time_last(old_logp, time_dim_local)
        adv = _move_time_last(adv, time_dim_local)

        log_ratio = (new_logp - old_logp).clamp(-20, 20)
        ratio = torch.exp(log_ratio)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * adv
        surr = torch.min(surr1, surr2)
        return surr.mean(), (old_logp - new_logp).mean(), (torch.abs(ratio - 1.0) > cfg.clip_epsilon).float().mean(), ratio.mean(), ratio.std(unbiased=False), ent

    def ppo_actor_loss_with_logs(sub_td):
        old_logp = sub_td["action_log_prob"].detach()
        adv = sub_td["advantage"]
        if adv.ndim == 2:
            adv = adv.unsqueeze(-1)
        surr_total, approx_kl, clipfrac, ratio_mean, ratio_std, ent = ppo_surrogate_terms(sub_td, adv, old_logp)
        loss_pi_raw = -surr_total
        loss_ent_raw = -ent
        loss_ent_scaled = cfg.entropy_coef * loss_ent_raw
        logs = {
            "approx_kl": float(approx_kl.detach().cpu()),
            "clipfrac": float(clipfrac.detach().cpu()),
            "ratio_mean": float(ratio_mean.detach().cpu()),
            "ratio_std": float(ratio_std.detach().cpu()),
            "ent": float(ent.detach().cpu()),
        }
        return loss_pi_raw, loss_ent_raw, loss_ent_scaled, loss_pi_raw + loss_ent_scaled, logs

    frames = 0
    last_it = -1
    term_counts = Counter()
    running_ep_steps = np.zeros((cfg.num_envs,), dtype=np.int64)
    policy.set_shield_apply_enabled(shield_apply_enabled_for_batch(0, cfg))

    nan_check_keys = [
        ("next", "observation"),
        ("next", "reward"),
        ("next", "constraint_cost"),
        ("next", "constraint_margin"),
        ("next", "constraint_violation"),
        ("action",),
        ("action_log_prob",),
        ("is_init",),
    ]

    try:
        for it, td in enumerate(collector):
            last_it = it
            current_shield_apply = shield_apply_enabled_for_batch(it, cfg)
            td = td.to(device)
            time_dim = get_time_dim(td)
            frames += int(td.numel())
            pbar.set_description(f"{frames}/{cfg.total_frames} ({100 * frames / cfg.total_frames:.1f}%)")

            if check_batch_for_nans(td, nan_check_keys, label=f"after collect batch {it}"):
                continue

            ended_lengths, running_mean_len = update_episode_length_stats(
                td,
                time_dim=time_dim,
                running_steps_np=running_ep_steps,
                term_counter=term_counts,
            )

            with torch.no_grad():
                state_value_r_norm = reward_critic(td["observation"].float())
                next_state_value_r_norm = reward_critic(td["next", "observation"].float())

                state_value_c_norm = cost_critic(td["observation"].float())
                next_state_value_c_norm = cost_critic(td["next", "observation"].float())

                td["state_value_norm"] = state_value_r_norm
                td["next", "state_value_norm"] = next_state_value_r_norm
                td["cost_value_norm"] = state_value_c_norm
                td["next", "cost_value_norm"] = next_state_value_c_norm

                td["state_value"] = unnormalize_value_prediction(
                    state_value_r_norm,
                    scale=cfg.reward_value_target_scale,
                    shift=cfg.reward_value_target_shift,
                )
                td["next", "state_value"] = unnormalize_value_prediction(
                    next_state_value_r_norm,
                    scale=cfg.reward_value_target_scale,
                    shift=cfg.reward_value_target_shift,
                )

                td["cost_value"] = unnormalize_value_prediction(
                    state_value_c_norm,
                    scale=cfg.cost_value_target_scale,
                    shift=cfg.cost_value_target_shift,
                )
                td["next", "cost_value"] = unnormalize_value_prediction(
                    next_state_value_c_norm,
                    scale=cfg.cost_value_target_scale,
                    shift=cfg.cost_value_target_shift,
                )

                td["next", "constraint"] = get_constraint_cost_tensor(td)

            try:
                compute_gae_inplace(
                    td,
                    reward_key=("next", "reward"),
                    value_key="state_value",
                    next_value_key=("next", "state_value"),
                    out_adv_key="advantage_r",
                    out_tgt_key="value_target_r",
                    gamma=cfg.reward_gamma,
                    lmbda=cfg.reward_lmbda,
                )
                compute_gae_inplace(
                    td,
                    reward_key=("next", "constraint"),
                    value_key="cost_value",
                    next_value_key=("next", "cost_value"),
                    out_adv_key="advantage_c",
                    out_tgt_key="value_target_c",
                    gamma=cfg.cost_gamma,
                    lmbda=cfg.cost_lmbda,
                )
            except RuntimeError as e:
                print("\n[GAE FAILED]", str(e))
                continue

            td["value_target_r_norm"] = normalize_value_target(
                td["value_target_r"],
                scale=cfg.reward_value_target_scale,
                shift=cfg.reward_value_target_shift,
            )
            td["value_target_c_norm"] = normalize_value_target(
                td["value_target_c"],
                scale=cfg.cost_value_target_scale,
                shift=cfg.cost_value_target_shift,
            )

            td["advantage_r_raw"] = td["advantage_r"].detach().clone()
            td["advantage_c_raw"] = td["advantage_c"].detach().clone()

            td["advantage_raw"] = td["advantage_r_raw"] - lag_lambda.detach() * td["advantage_c_raw"]
            td["advantage"] = td["advantage_raw"].clone()
            normalize_advantage_(td, adv_key="advantage", eps=cfg.eps)

            with torch.no_grad():
                batch_lambda_mean = get_lambda_signal_tensor(td, cfg).reshape(-1).mean()
                if it >= cfg.lambda_warmup_batches:
                    lag_lambda = torch.clamp(
                        lag_lambda + cfg.lambda_lr * (batch_lambda_mean - cfg.cost_budget),
                        cfg.lambda_min,
                        cfg.lambda_max,
                    )

            losses = Counter()
            mb_count = 0

            for _epoch in range(cfg.ppo_epochs):
                for sub, _start in contiguous_minibatches(td, cfg.chunk_T, time_dim=time_dim):
                    prepare_recurrent_minibatch(sub, time_dim=time_dim)
                    sub_actor = sub.clone()
                    sub_dyn = sub.clone()

                    with set_recurrent_mode(True):
                        try:
                            actor_trunk(sub_actor)
                            pred_r = reward_critic(sub_actor["observation"].float())
                            pred_c = cost_critic(sub_actor["observation"].float())

                            loss_pi_raw, loss_ent_raw, loss_ent_scaled, actor_total, pi_logs = ppo_actor_loss_with_logs(sub_actor)
                            loss_vr_raw = F.smooth_l1_loss(pred_r, sub_actor["value_target_r_norm"].detach().squeeze(-1))
                            loss_vr_total = cfg.reward_value_coef * loss_vr_raw
                            loss_vc_raw = F.smooth_l1_loss(pred_c, sub_actor["value_target_c_norm"].detach().squeeze(-1))
                            loss_vc_total = cfg.cost_value_coef * loss_vc_raw

                            require_finite(actor_total, "actor_total")
                            require_finite(loss_vr_total, "loss_vr_total")
                            require_finite(loss_vc_total, "loss_vc_total")
                        except RuntimeError as e:
                            print("\n[MINIBATCH SKIP]", str(e))
                            continue

                        actor_optim.zero_grad(set_to_none=True)
                        actor_total.backward()
                        clip_grad_norm_(actor_params, cfg.max_grad_norm)
                        actor_optim.step()

                        reward_critic_optim.zero_grad(set_to_none=True)
                        loss_vr_total.backward()
                        clip_grad_norm_(reward_critic.parameters(), cfg.max_grad_norm)
                        reward_critic_optim.step()

                        cost_critic_optim.zero_grad(set_to_none=True)
                        loss_vc_total.backward()
                        clip_grad_norm_(cost_critic.parameters(), cfg.max_grad_norm)
                        cost_critic_optim.step()

                        with set_recurrent_mode(True):
                            actor_trunk(sub_dyn)

                        try:
                            loss_dyn_raw, dyn_logs = latent_dynamics_loss(sub_dyn, state_proj_core, dyn_model, temp_decoder, cfg, batch_idx=it)
                            loss_dyn_scaled = cfg.dyn_loss_coef * loss_dyn_raw
                            dynamics_total = loss_dyn_scaled
                            require_finite(loss_dyn_raw, "loss_dyn_raw")
                            require_finite(dynamics_total, "dynamics_total")
                        except RuntimeError as e:
                            print("\n[DYNAMICS MINIBATCH SKIP]", str(e))
                            continue

                        dynamics_optim.zero_grad(set_to_none=True)
                        dynamics_total.backward()
                        clip_grad_norm_(dynamics_params, cfg.max_grad_norm)
                        dynamics_optim.step()

                        mb_count += 1
                        losses["actor_loss_total"] += float(actor_total.detach().cpu())
                        losses["reward_value_loss_total"] += float(loss_vr_total.detach().cpu())
                        losses["cost_value_loss_total"] += float(loss_vc_total.detach().cpu())
                        losses["loss_dyn_scaled"] += float(loss_dyn_scaled.detach().cpu())
                        losses["dynamics_total"] += float(dynamics_total.detach().cpu())
                        for k, v in dyn_logs.items():
                            losses[k] += v
                        losses["approx_kl"] += pi_logs["approx_kl"]
                        losses["clipfrac"] += pi_logs["clipfrac"]
                        losses["ratio_mean"] += pi_logs["ratio_mean"]
                        losses["ratio_std"] += pi_logs["ratio_std"]
                        losses["entropy"] += pi_logs["ent"]

                        del sub_actor, sub_dyn
                        del actor_total, loss_vr_total, loss_vc_total, dynamics_total
                        del loss_dyn_raw, loss_dyn_scaled
                        if torch.cuda.is_available() and int(cfg.cuda_empty_cache_every_minibatches) > 0 and (mb_count % int(cfg.cuda_empty_cache_every_minibatches) == 0):
                            torch.cuda.empty_cache()

            if torch.cuda.is_available() and bool(cfg.empty_cache_after_checkpoint):
                torch.cuda.empty_cache()

            with torch.no_grad():
                pred_r_norm = reward_critic(td["observation"].float())
                pred_c_norm = cost_critic(td["observation"].float())

                pred_r = unnormalize_value_prediction(
                    pred_r_norm,
                    scale=cfg.reward_value_target_scale,
                    shift=cfg.reward_value_target_shift,
                )
                pred_c = unnormalize_value_prediction(
                    pred_c_norm,
                    scale=cfg.cost_value_target_scale,
                    shift=cfg.cost_value_target_shift,
                )

                ev_r = explained_variance(pred_r, td["value_target_r"].squeeze(-1))
                ev_c = explained_variance(pred_c, td["value_target_c"].squeeze(-1))
                attach_normalized_observation_(td, reward_critic)

            if it % cfg.log_every_batches == 0:
                denom = max(1, mb_count)
                c_tb = to_TB(get_constraint_cost_tensor(td), time_dim=time_dim)
                if c_tb.ndim > 2:
                    c_tb = c_tb.squeeze(-1)
                lambda_tb = to_TB(get_lambda_signal_tensor(td, cfg), time_dim=time_dim).float()
                if lambda_tb.ndim > 2:
                    lambda_tb = lambda_tb.squeeze(-1)

                viol_mean = mean_of(td, "constraint_violation", time_dim=time_dim)

                train_reward_mean = mean_of(td, "reward", time_dim=time_dim)
                energy_reward_mean = mean_of(td, "energy_reward", time_dim=time_dim)
                native_reward_mean = mean_of(td, "native_reward", time_dim=time_dim)

                smoothness_penalty_mean = mean_of(td, "smoothness_penalty", time_dim=time_dim)
                smoothness_raw_mean = mean_of(td, "smoothness_raw", time_dim=time_dim)
                action_delta_l1_mean = mean_of(td, "action_delta_l1", time_dim=time_dim)
                action_delta_l2_mean = mean_of(td, "action_delta_l2", time_dim=time_dim)

                cost_mean = mean_of(td, "constraint_cost", time_dim=time_dim)
                temp_mean = mean_of(td, "air_temperature", time_dim=time_dim)
                power_mean = mean_of(td, "hvac_power", time_dim=time_dim)
                comfort_low_mean = mean_of(td, "comfort_low", time_dim=time_dim)
                comfort_high_mean = mean_of(td, "comfort_high", time_dim=time_dim)
                total_temp_violation_mean = mean_of(td, "total_temperature_violation", time_dim=time_dim)

                lines = [
                    f"=== batch {it} ===",
                    f"env={cfg.sinergym_env_id} | season={cfg.season} | runperiod_mode={describe_runperiod_sampling(cfg)}",
                    f"lrs: actor={actor_optim.param_groups[0]['lr']:.2e} | backbone_dyn={dynamics_optim.param_groups[0]['lr']:.2e} | projector={dynamics_optim.param_groups[1]['lr']:.2e} | dyn={dynamics_optim.param_groups[2]['lr']:.2e} | Vr={reward_critic_optim.param_groups[0]['lr']:.2e} | Vc={cost_critic_optim.param_groups[0]['lr']:.2e}",
                    f"done={mean_done_rate(td, time_dim=time_dim):.2%} | lambda={float(lag_lambda):.4f} | constraint_cost μ={c_tb.mean().item():.6f} σ={c_tb.std(unbiased=False).item():.6f} max={c_tb.max().item():.6f} | env_violation μ={lambda_tb.mean().item():.6f} | violation={(viol_mean if viol_mean is not None else float('nan')):.2%}",
                    f"train_reward μ={(train_reward_mean if train_reward_mean is not None else float('nan')):.6f} | energy_reward μ={(energy_reward_mean if energy_reward_mean is not None else float('nan')):.6f} | native_reward μ={(native_reward_mean if native_reward_mean is not None else float('nan')):.6f}",
                    f"smoothness: penalty μ={(smoothness_penalty_mean if smoothness_penalty_mean is not None else float('nan')):.6f} | raw μ={(smoothness_raw_mean if smoothness_raw_mean is not None else float('nan')):.6f} | delta_l1 μ={(action_delta_l1_mean if action_delta_l1_mean is not None else float('nan')):.6f} | delta_l2 μ={(action_delta_l2_mean if action_delta_l2_mean is not None else float('nan')):.6f}",
                    f"comfort_cost μ={(cost_mean if cost_mean is not None else float('nan')):.6f} | air_temperature μ={(temp_mean if temp_mean is not None else float('nan')):.4f} | hvac_power μ={(power_mean if power_mean is not None else float('nan')):.4f}",
                    f"comfort_band μ=[{(comfort_low_mean if comfort_low_mean is not None else float('nan')):.3f}, {(comfort_high_mean if comfort_high_mean is not None else float('nan')):.3f}] | total_temperature_violation μ={(total_temp_violation_mean if total_temp_violation_mean is not None else float('nan')):.6f}",
                    f"episodes: ended={len(ended_lengths)} | ended_len_mean={(float(np.mean(ended_lengths)) if ended_lengths else float('nan')):.2f} | running_len_mean={running_mean_len:.2f}",
                ]
                if len(term_counts) > 0:
                    lines.append("term_codes: " + ", ".join([f"{k}:{v}" for k, v in term_counts.most_common(6)]))
                if safe_key(td, ("action",)) is not None:
                    lines.append(action_stats_by_dim(safe_key(td, ("action",)), name="action_sample"))
                obs_stats = mean_std_max(td, "observation", time_dim=time_dim)
                if obs_stats:
                    lines.append(f"obs_raw: μ={obs_stats[0]:.4f} σ={obs_stats[1]:.4f} max={obs_stats[2]:.4f} min={obs_stats[3]:.4f}")
                model_obs_stats = mean_std_max(td, "model_observation", time_dim=time_dim)
                if model_obs_stats:
                    lines.append(f"model_obs_raw: μ={model_obs_stats[0]:.4f} σ={model_obs_stats[1]:.4f} max={model_obs_stats[2]:.4f} min={model_obs_stats[3]:.4f}")
                obs_norm_stats = mean_std_max(td, "observation_normalized", time_dim=time_dim)
                if obs_norm_stats:
                    lines.append(f"obs_norm: μ={obs_norm_stats[0]:.4f} σ={obs_norm_stats[1]:.4f} max={obs_norm_stats[2]:.4f} min={obs_norm_stats[3]:.4f}")
                constraint_margin_td = safe_key(td, ("next", "constraint_margin"))
                if constraint_margin_td is not None:
                    lines.append(rollout_summary_line(to_TB(constraint_margin_td, time_dim).float(), "constraint_margin"))

                lines.append(
                    f"PPO: KL={losses['approx_kl']/denom:.6f} | clipfrac={losses['clipfrac']/denom:.4f} | ratio μ={losses['ratio_mean']/denom:.4f} σ={losses['ratio_std']/denom:.4f} | entropy={losses['entropy']/denom:.4f}"
                )
                lines.append(
                    f"losses: actor={losses['actor_loss_total']/denom:.4f} | Vr={losses['reward_value_loss_total']/denom:.4f} | Vc={losses['cost_value_loss_total']/denom:.4f} | dyn_total={losses['dynamics_total']/denom:.6f} | dyn={losses['loss_dyn_scaled']/denom:.6f}"
                )
                lines.append(
                    f"dreamer schedule: rollout_progress={losses['dyn_rollout_progress']/denom:.4f} | horizon={int(cfg.dyn_roll_horizon)} | dynamics_aux_on={losses['dynamics_aux_enabled']/denom:.2f} | dynamics_aux_after_batch={losses['dynamics_aux_enable_after_batch']/denom:.2f} | latent_dyn1_coef={cfg.latent_dyn1_coef:.3f} | latent_roll_coef={cfg.latent_roll_coef:.3f} | temp_res_dyn1_coef={cfg.temp_residual_dyn1_coef:.3f} | temp_res_roll_coef={cfg.temp_residual_roll_coef:.3f} | viol_dyn1_coef={cfg.viol_dyn1_coef:.3f} | viol_roll_coef={cfg.viol_roll_coef:.3f}"
                )
                lines.append(
                    f"persistence penalties: temp_dyn1={cfg.temp_persist_dyn1_coef:.3f} | temp_roll={cfg.temp_persist_roll_coef:.3f} | viol_dyn1={cfg.viol_persist_dyn1_coef:.3f} | viol_roll={cfg.viol_persist_roll_coef:.3f}"
                )
                lines.append(
                    f"open-loop latent dynamics: one_step_loss={losses['dyn_1step']/denom:.6f} | roll_loss={losses['dyn_roll']/denom:.6f} | one_step_rmse={losses['dyn_1step_rmse']/denom:.6f} | one_step_nrmse={losses['dyn_1step_nrmse']/denom:.6f} | roll_rmse={losses['dyn_roll_rmse']/denom:.6f} | roll_nrmse={losses['dyn_roll_nrmse']/denom:.6f} | regime_maxprob={losses['regime_maxprob']/denom:.4f}"
                )
                lines.append(
                    f"temp/viol supervision: temp_dyn1={losses['loss_temp_res_dyn1']/denom:.6f} | temp_roll={losses['loss_temp_res_roll']/denom:.6f} | viol_dyn1={losses['loss_viol_dyn1']/denom:.6f} | viol_roll={losses['loss_viol_roll']/denom:.6f}"
                )
                lines.append(
                    f"vs persistence penalties: temp_dyn1={losses['loss_temp_persist_dyn1']/denom:.6f} | temp_roll={losses['loss_temp_persist_roll']/denom:.6f} | viol_dyn1={losses['loss_viol_persist_dyn1']/denom:.6f} | viol_roll={losses['loss_viol_persist_roll']/denom:.6f}"
                )
                lines.append(
                    f"gain vs persist (nrmse): temp_last={losses['temp_gain_nrmse_last_h']/denom:.4f} | temp_min={losses['temp_gain_nrmse_min_h']/denom:.4f} | viol_last={losses['viol_gain_nrmse_last_h']/denom:.4f} | viol_min={losses['viol_gain_nrmse_min_h']/denom:.4f}"
                )
                lines.append(
                    f"temperature decoder: pred μ={losses['decoded_temperature_mean']/denom:.4f} σ={losses['decoded_temperature_std']/denom:.4f} min={losses['decoded_temperature_min']/denom:.4f} max={losses['decoded_temperature_max']/denom:.4f} | latent_norm={losses['latent_state_norm']/denom:.4f}"
                )
                lines.append(
                    f"temp fit (open-loop monitor): direct_rmse={losses['latent_temperature_direct_rmse']/denom:.4f} | dyn1_rmse={losses['latent_temperature_dyn1_rmse']/denom:.4f} | roll_rmse={losses['latent_temperature_roll_rmse']/denom:.4f} | direct_nrmse={losses['latent_temperature_direct_nrmse']/denom:.4f} | dyn1_nrmse={losses['latent_temperature_dyn1_nrmse']/denom:.4f} | roll_nrmse={losses['latent_temperature_roll_nrmse']/denom:.4f}"
                )
                lines.append(
                    f"temp persistence baseline: dyn1_rmse={losses['temp_persistence_dyn1_rmse']/denom:.4f} | roll_rmse={losses['temp_persistence_roll_rmse']/denom:.4f} | dyn1_nrmse={losses['temp_persistence_dyn1_nrmse']/denom:.4f} | roll_nrmse={losses['temp_persistence_roll_nrmse']/denom:.4f} | gain_dyn1={losses['temp_rmse_gain_dyn1']/denom:.4f} | gain_roll={losses['temp_rmse_gain_roll']/denom:.4f}"
                )
                lines.append(
                    f"temp vs persistence: dyn1_loss={losses['loss_temp_persist_dyn1']/denom:.6f} | roll_loss={losses['loss_temp_persist_roll']/denom:.6f} | dyn1_surplus={losses['temp_vs_persist_dyn1_surplus']/denom:.4f} | roll_surplus={losses['temp_vs_persist_roll_surplus']/denom:.4f} | beat_dyn1={losses['temp_beats_persist_dyn1']/denom:.4f} | beat_roll={losses['temp_beats_persist_roll']/denom:.4f}"
                )
                lines.append(
                    f"viol vs persistence: dyn1_loss={losses['loss_viol_persist_dyn1']/denom:.6f} | roll_loss={losses['loss_viol_persist_roll']/denom:.6f} | dyn1_surplus={losses['viol_vs_persist_dyn1_surplus']/denom:.4f} | roll_surplus={losses['viol_vs_persist_roll_surplus']/denom:.4f} | beat_dyn1={losses['viol_beats_persist_dyn1']/denom:.4f} | beat_roll={losses['viol_beats_persist_roll']/denom:.4f}"
                )
                lines.append(
                    f"violation fit (open-loop): dyn1_rmse={losses['viol_dyn1_rmse']/denom:.4f} | roll_rmse={losses['viol_roll_rmse']/denom:.4f} | dyn1_nrmse={losses['viol_dyn1_nrmse']/denom:.4f} | roll_nrmse={losses['viol_roll_nrmse']/denom:.4f} | persist_roll_rmse={losses['viol_persistence_roll_rmse']/denom:.4f} | gain_roll={losses['viol_rmse_gain_roll']/denom:.4f}"
                )
                lines.append(
                    f"temp bias/corr: direct_bias={losses['latent_temperature_direct_bias']/denom:.4f} | dyn1_bias={losses['latent_temperature_dyn1_bias']/denom:.4f} | roll_bias={losses['latent_temperature_roll_bias']/denom:.4f} | direct_corr={losses['latent_temperature_direct_corr']/denom:.4f} | dyn1_corr={losses['latent_temperature_dyn1_corr']/denom:.4f}"
                )
                lines.append(
                    f"latent_constraint_from_decoded_temp: μ={losses['latent_constraint_cost_mean']/denom:.6f} σ={losses['latent_constraint_cost_std']/denom:.6f} max={losses['latent_constraint_cost_max']/denom:.6f} | violation={(losses['latent_constraint_violation_rate']/denom):.2%}"
                )

                adv_r_raw = td["advantage_r_raw"]
                adv_c_raw = td["advantage_c_raw"]
                adv_raw = td["advantage_raw"]

                adv_r_norm = td["advantage_r"]
                adv_c_norm = td["advantage_c"]
                adv_norm = td["advantage"]

                lines.append(tensor_stats_line(adv_r_raw, "adv_r_raw"))
                lines.append(tensor_stats_line(adv_c_raw, "adv_c_raw"))
                lines.append(tensor_stats_line(adv_raw, "adv_combined_raw"))

                lines.append(tensor_stats_line(adv_r_norm, "adv_r_norm"))
                lines.append(tensor_stats_line(adv_c_norm, "adv_c_norm"))
                lines.append(tensor_stats_line(adv_norm, "adv_combined_norm"))

                lines.append(tensor_corr_line(adv_r_raw, adv_c_raw, "adv_r_raw vs adv_c_raw"))
                lines.append(tensor_corr_line(adv_r_norm, adv_c_norm, "adv_r_norm vs adv_c_norm"))

                temp_rmse_h_parts = [f"h{h:02d}={losses[f'latent_temperature_rmse_h{h:02d}']/denom:.4f}" for h in range(1, int(cfg.dyn_roll_horizon) + 1)]
                for chunk_idx in range(0, len(temp_rmse_h_parts), 10):
                    prefix = "dreamed temperature rmse (open-loop): " if chunk_idx == 0 else "                                     "
                    lines.append(prefix + " | ".join(temp_rmse_h_parts[chunk_idx:chunk_idx + 10]))

                temp_h_parts = [f"h{h:02d}={losses[f'latent_temperature_nrmse_h{h:02d}']/denom:.4f}" for h in range(1, int(cfg.dyn_roll_horizon) + 1)]
                for chunk_idx in range(0, len(temp_h_parts), 10):
                    prefix = "dreamed temperature nrmse (open-loop): " if chunk_idx == 0 else "                                      "
                    lines.append(prefix + " | ".join(temp_h_parts[chunk_idx:chunk_idx + 10]))

                temp_persist_h_parts = [f"h{h:02d}={losses[f'latent_temperature_persist_nrmse_h{h:02d}']/denom:.4f}" for h in range(1, int(cfg.dyn_roll_horizon) + 1)]
                for chunk_idx in range(0, len(temp_persist_h_parts), 10):
                    prefix = "persistence temperature nrmse: " if chunk_idx == 0 else "                              "
                    lines.append(prefix + " | ".join(temp_persist_h_parts[chunk_idx:chunk_idx + 10]))

                temp_gain_h_parts = [f"h{h:02d}={losses[f'latent_temperature_gain_nrmse_h{h:02d}']/denom:.4f}" for h in range(1, int(cfg.dyn_roll_horizon) + 1)]
                for chunk_idx in range(0, len(temp_gain_h_parts), 10):
                    prefix = "temperature gain vs persist (nrmse): " if chunk_idx == 0 else "                                  "
                    lines.append(prefix + " | ".join(temp_gain_h_parts[chunk_idx:chunk_idx + 10]))

                viol_h_parts = [f"h{h:02d}={losses[f'latent_violation_nrmse_h{h:02d}']/denom:.4f}" for h in range(1, int(cfg.dyn_roll_horizon) + 1)]
                for chunk_idx in range(0, len(viol_h_parts), 10):
                    prefix = "dreamed violation nrmse (open-loop): " if chunk_idx == 0 else "                                   "
                    lines.append(prefix + " | ".join(viol_h_parts[chunk_idx:chunk_idx + 10]))

                viol_gain_h_parts = [f"h{h:02d}={losses[f'latent_violation_gain_nrmse_h{h:02d}']/denom:.4f}" for h in range(1, int(cfg.dyn_roll_horizon) + 1)]
                for chunk_idx in range(0, len(viol_gain_h_parts), 10):
                    prefix = "violation gain vs persist (nrmse): " if chunk_idx == 0 else "                                "
                    lines.append(prefix + " | ".join(viol_gain_h_parts[chunk_idx:chunk_idx + 10]))

                shield_eval_td = safe_key(td, ("shield_eval_enabled",))
                shield_apply_td = safe_key(td, ("shield_apply_enabled",))
                shield_active_td = safe_key(td, ("shield_active",))
                shield_changed_td = safe_key(td, ("shield_changed",))
                shield_would_changed_td = safe_key(td, ("shield_would_change",))
                shield_base_v_td = safe_key(td, ("shield_base_violation",))
                shield_best_v_td = safe_key(td, ("shield_best_violation",))
                shield_final_v_td = safe_key(td, ("shield_chosen_violation",))
                shield_delta_td = safe_key(td, ("shield_delta_norm",))
                shield_would_delta_td = safe_key(td, ("shield_would_delta_norm",))
                if shield_active_td is not None:
                    se = to_TB(shield_eval_td.float(), time_dim).squeeze(-1) if shield_eval_td is not None else None
                    sp = to_TB(shield_apply_td.float(), time_dim).squeeze(-1) if shield_apply_td is not None else None
                    sa = to_TB(shield_active_td.float(), time_dim).squeeze(-1)
                    sc = to_TB(shield_changed_td.float(), time_dim).squeeze(-1) if shield_changed_td is not None else None
                    sw = to_TB(shield_would_changed_td.float(), time_dim).squeeze(-1) if shield_would_changed_td is not None else None
                    sb = to_TB(shield_base_v_td.float(), time_dim).squeeze(-1) if shield_base_v_td is not None else None
                    sx = to_TB(shield_best_v_td.float(), time_dim).squeeze(-1) if shield_best_v_td is not None else None
                    sf = to_TB(shield_final_v_td.float(), time_dim).squeeze(-1) if shield_final_v_td is not None else None
                    sd = to_TB(shield_delta_td.float(), time_dim).squeeze(-1) if shield_delta_td is not None else None
                    swd = to_TB(shield_would_delta_td.float(), time_dim).squeeze(-1) if shield_would_delta_td is not None else None
                    lines.append(
                        f"shield: eval_on={(se.mean().item() if se is not None else 0.0):.2%} | apply_on={(sp.mean().item() if sp is not None else float(current_shield_apply)):.2%} | apply_after_batch={int(cfg.shield_apply_after_batch)} | active_rate={sa.mean().item():.2%} | changed_rate={(sc.mean().item() if sc is not None else 0.0):.2%} | would_change_rate={(sw.mean().item() if sw is not None else 0.0):.2%} | base_pred_v={(sb.mean().item() if sb is not None else 0.0):.4f} | best_pred_v={(sx.mean().item() if sx is not None else 0.0):.4f} | final_pred_v={(sf.mean().item() if sf is not None else 0.0):.4f} | delta_norm={(sd.mean().item() if sd is not None else 0.0):.4f} | would_delta_norm={(swd.mean().item() if swd is not None else 0.0):.4f}"
                    )
                lines.append(f"critics: EV_r={ev_r:.4f} | EV_c={ev_c:.4f}")
                log_text = "\n".join(lines)
                if bool(cfg.sinergym_disable_progress_bar):
                    print(log_text, flush=True)
                else:
                    pbar.write(log_text)

            policy.set_shield_apply_enabled(shield_apply_enabled_for_batch(it + 1, cfg))

            if (it + 1) % cfg.save_every_batches == 0:
                save_checkpoint(
                    run_dir=run_dir,
                    policy=policy,
                    reward_critic=reward_critic,
                    cost_critic=cost_critic,
                    dynamics_model=dyn_model,
                    temperature_decoder=temp_decoder,
                    actor_optim=actor_optim,
                    dynamics_optim=dynamics_optim,
                    reward_critic_optim=reward_critic_optim,
                    cost_critic_optim=cost_critic_optim,
                    lag_lambda=lag_lambda,
                    frames=frames,
                    batch_idx=it,
                    term_counts=term_counts,
                    running_ep_steps=running_ep_steps,
                    cfg=cfg,
                )
                if torch.cuda.is_available() and bool(cfg.empty_cache_after_checkpoint):
                    torch.cuda.empty_cache()

            if frames >= cfg.total_frames:
                break

    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl+C received. Saving current models/checkpoint...")

    finally:
        try:
            collector.shutdown()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        try:
            pbar.close()
        except Exception:
            pass

        save_checkpoint(
            run_dir=run_dir,
            policy=policy,
            reward_critic=reward_critic,
            cost_critic=cost_critic,
            dynamics_model=dyn_model,
            temperature_decoder=temp_decoder,
            actor_optim=actor_optim,
            dynamics_optim=dynamics_optim,
            reward_critic_optim=reward_critic_optim,
            cost_critic_optim=cost_critic_optim,
            lag_lambda=lag_lambda,
            frames=frames,
            batch_idx=last_it,
            term_counts=term_counts,
            running_ep_steps=running_ep_steps,
            cfg=cfg,
        )
        if torch.cuda.is_available() and bool(cfg.empty_cache_after_checkpoint):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
