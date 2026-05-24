import difflib as _difflib
import numpy as _np2
import gymnasium as _gym
from gymnasium import spaces as _spaces

from fmpy import read_model_description as _read_model_description, extract as _extract
from fmpy.simulation import instantiate_fmu as _instantiate_fmu

import pandas as _pd


def _np(x) -> _np2.float32:
    return _np2.float32(x)


class SupermarketFMUEnv(_gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        fmu_path: str,
        horizon: int = 720,
        sim_dt: float = 1.0,
        ctrl_dt: float = 120.0,
        Psuc_ref_bounds=(0.0, 38.0),
        Troom_ref_bounds_C=(0.0, 10.0),
        T_amb_const_C: float = 30.0,
        Psuc_ref_init: float = 20.0,
        Troom_ref_init_C: float = 4.0,
        tcabinet_var: str = "Tcabinet",
        wcomp_var: str = "Wcomp",
        qevap_var: str | None = "Qevap",
        OUT_NAMES=("Tcabinet", "Wcomp", "Qevap", "T_amb_out"),
        electricity_price_eur_per_kwh: float = 10.0,
        fan_power_coeff: float = 0.0,
        price_reward_weight: float = 5.0,
        T_wmean_min_C: float = 3.9,
        T_wmean_max_C: float = 4.1,
        temp_window_seconds: float = 14400.0 * 3,
        temp_weighted_plateau_hours: float = 4.0,
        temp_weighted_decay_hours: float = 1.0,
        temp_weighted_floor: float = 0.02,
        include_weighted_temp_in_obs: bool = True,
        T_weighted_div: float | None = None,
        terminate_on_window_mean_violation: bool = False,
        termination_buffer_seconds: float = 14400.0 * 5,
        alive_reward_per_ctrl_step: float = 0.1,
        early_end_penalty: float = 0.0,
        seed: int | None = None,
        strict_io: bool = True,
        print_available_vars: bool = False,
        scale_observations: bool = True,
        obs_divisors: dict[str, float] | None = None,
        T_wmean_div: float = 10.0,
        include_applied_action_in_obs: bool = False,
        include_window_fill_in_obs: bool = False,
        Psuc_applied_div: float | None = None,
        Troom_applied_div: float | None = None,
        action_lpf_alpha: float = 0.8,
        action_smoothness_coef: float = 0.01,
        action_smoothness_type: str = "l2",
        action_smoothness_use_scaled: bool = True,
        truncation_penalty: float = 0.0,
        fmu_failure_reward: float = 0.0,
        max_total_power_W: float = 50_000_000.0,
        max_step_cost_eur: float = 1_000.0,
        kill_if_step_cost_gt: float = 10_000.0,
        kill_if_total_power_gt: float = 50_000_000.0,
        debug_print_on_fail: bool = False,
        debug_print_max_vars: int = 20,
        terminate_on_obs_nan: bool = True,
        max_reset_retries: int = 3,
        use_excel_profile: bool = True,
        excel_path: str = "Price_Temp.xlsx",
        time_col: str = "HourDK",
        price_col: str = "SpotPriceEUR",
        temp_col: str = "Temperature_C",
        episode_hours: int = 24,
        forecast_hours: int = 24,
        price_is_eur_per_mwh: bool = True,
        scale_time_feature: bool = True,
        scale_forecast: bool = True,
        forecast_temp_div: float = 50.0,
        forecast_price_div: float = 0.5,
        exo_hold_at_ctrl_dt: bool = True,
        amb_smooth_mode: str = "interp",
        amb_ema_beta: float = 0.05,
        clamp_T_amb: bool = False,
        T_amb_min_C: float = -5.0,
        T_amb_max_C: float = 40.0,
        price_neg_floor_eur_per_kwh: float = 0.05,
        alpha_power: float = 0.0,
        max_profit_eur_per_step: float = 1.0,
        violation_clip_C: float | None = 4.0,
    ):
        super().__init__()

        self.fmu_path = fmu_path
        self.horizon = int(horizon)

        self.sim_dt = float(sim_dt)
        self.ctrl_dt = float(ctrl_dt)
        if self.sim_dt <= 0 or self.ctrl_dt <= 0:
            raise ValueError("sim_dt and ctrl_dt must be > 0")

        ratio = self.ctrl_dt / self.sim_dt
        self.substeps = int(round(ratio))
        if abs(ratio - self.substeps) > 1e-9:
            raise ValueError(f"ctrl_dt must be integer multiple of sim_dt. Got ctrl_dt/sim_dt={ratio}")

        self.rng = _np2.random.default_rng(seed)

        self.T_amb_const_C = float(T_amb_const_C)

        self.Psuc_ref_init = float(Psuc_ref_init)
        self.Troom_ref_init_C = float(Troom_ref_init_C)

        self.Psuc_ref_bounds = (float(Psuc_ref_bounds[0]), float(Psuc_ref_bounds[1]))
        self.Troom_ref_bounds_C = (float(Troom_ref_bounds_C[0]), float(Troom_ref_bounds_C[1]))

        self.tcabinet_var = str(tcabinet_var)
        self.wcomp_var = str(wcomp_var)
        self.qevap_var = None if qevap_var is None else str(qevap_var)
        self.OUT_NAMES = list(OUT_NAMES)

        self.c_elec = float(electricity_price_eur_per_kwh)
        self.fan_power_coeff = float(fan_power_coeff)
        self.price_reward_weight = float(price_reward_weight)
        if self.price_reward_weight < 0.0:
            raise ValueError("price_reward_weight must be >= 0")

        self.T_wmean_min_C = float(T_wmean_min_C)
        self.T_wmean_max_C = float(T_wmean_max_C)

        self.temp_window_seconds = float(temp_window_seconds)
        self.temp_window_size_steps = max(1, int(round(self.temp_window_seconds / self.sim_dt)))

        self.temp_weighted_plateau_hours = float(temp_weighted_plateau_hours)
        self.temp_weighted_decay_hours = float(temp_weighted_decay_hours)
        self.temp_weighted_floor = float(temp_weighted_floor)
        self.include_weighted_temp_in_obs = bool(include_weighted_temp_in_obs)
        self.T_weighted_div = (
            float(T_weighted_div)
            if (T_weighted_div is not None and float(T_weighted_div) > 0)
            else float(T_wmean_div)
        )

        if self.temp_weighted_plateau_hours < 0.0:
            raise ValueError("temp_weighted_plateau_hours must be >= 0")
        if self.temp_weighted_decay_hours <= 0.0:
            raise ValueError("temp_weighted_decay_hours must be > 0")
        if self.temp_weighted_floor < 0.0:
            raise ValueError("temp_weighted_floor must be >= 0")

        self.terminate_on_window_mean_violation = bool(terminate_on_window_mean_violation)

        self.termination_buffer_seconds = float(termination_buffer_seconds)
        self.termination_buffer_steps = max(0, int(round(self.termination_buffer_seconds / self.ctrl_dt)))

        self.alive_reward_per_ctrl_step = float(alive_reward_per_ctrl_step)
        self.early_end_penalty = float(early_end_penalty)

        self.strict_io = bool(strict_io)
        self.truncation_penalty = float(truncation_penalty)
        self.fmu_failure_reward = float(fmu_failure_reward)

        self.max_total_power_W = float(max_total_power_W)
        self.max_step_cost_eur = float(max_step_cost_eur)
        self.kill_if_step_cost_gt = float(kill_if_step_cost_gt)
        self.kill_if_total_power_gt = float(kill_if_total_power_gt)

        self.debug_print_on_fail = bool(debug_print_on_fail)
        self.debug_print_max_vars = int(debug_print_max_vars)
        self.terminate_on_obs_nan = bool(terminate_on_obs_nan)
        self.max_reset_retries = int(max_reset_retries)

        self.use_excel_profile = bool(use_excel_profile)
        self.excel_path = str(excel_path)
        self.time_col = str(time_col)
        self.price_col = str(price_col)
        self.temp_col = str(temp_col)

        self.episode_hours = int(episode_hours)
        self.forecast_hours = int(forecast_hours)
        if self.episode_hours <= 0:
            raise ValueError("episode_hours must be > 0")
        if self.forecast_hours <= 0:
            raise ValueError("forecast_hours must be > 0")

        self.price_is_eur_per_mwh = bool(price_is_eur_per_mwh)

        self.scale_time_feature = bool(scale_time_feature)
        self.scale_forecast = bool(scale_forecast)
        self.forecast_temp_div = float(forecast_temp_div) if float(forecast_temp_div) > 0 else 50.0
        self.forecast_price_div = float(forecast_price_div) if float(forecast_price_div) > 0 else 1.0

        self.exo_hold_at_ctrl_dt = bool(exo_hold_at_ctrl_dt)
        self.amb_smooth_mode = str(amb_smooth_mode).lower().strip()
        if self.amb_smooth_mode not in ("none", "ema", "interp"):
            raise ValueError("amb_smooth_mode must be one of: 'none', 'ema', 'interp'")
        self.amb_ema_beta = float(amb_ema_beta)
        if not (0.0 < self.amb_ema_beta <= 1.0):
            raise ValueError(f"amb_ema_beta must be in (0,1], got {self.amb_ema_beta}")

        self.clamp_T_amb = bool(clamp_T_amb)
        self.T_amb_min_C = float(T_amb_min_C)
        self.T_amb_max_C = float(T_amb_max_C)
        if self.T_amb_min_C >= self.T_amb_max_C:
            raise ValueError("T_amb_min_C must be < T_amb_max_C")

        self.price_neg_floor_eur_per_kwh = float(price_neg_floor_eur_per_kwh)
        if self.price_neg_floor_eur_per_kwh < 0.0:
            raise ValueError("price_neg_floor_eur_per_kwh must be >= 0")
        self.alpha_power = float(alpha_power)
        if self.alpha_power < 0.0:
            raise ValueError("alpha_power must be >= 0")
        self.max_profit_eur_per_step = float(max_profit_eur_per_step)
        if self.max_profit_eur_per_step < 0.0:
            raise ValueError("max_profit_eur_per_step must be >= 0")

        self.violation_clip_C = None if (violation_clip_C is None) else float(violation_clip_C)
        if (self.violation_clip_C is not None) and (self.violation_clip_C <= 0.0):
            raise ValueError("violation_clip_C must be > 0 or None")

        self._day_start_idxs = None
        self._times = None
        self._temps_C_all = None
        self._prices_eur_per_kwh_all = None
        self._profile_date_str = None
        self._profile_start_idx = None
        self._temps_C = None
        self._prices_eur_per_kwh = None
        self._episode_ctrl_steps = None

        if self.use_excel_profile:
            self._load_excel_profiles()

        self.IN_NAMES = ["Psuc_ref", "T_amb", "Troom_ref_in"]

        self.action_space = _spaces.Box(
            low=_np2.array([self.Psuc_ref_bounds[0], self.Troom_ref_bounds_C[0]], dtype=_np2.float32),
            high=_np2.array([self.Psuc_ref_bounds[1], self.Troom_ref_bounds_C[1]], dtype=_np2.float32),
            shape=(2,),
            dtype=_np2.float32,
        )

        self.md = _read_model_description(self.fmu_path)
        self.unzipdir = _extract(self.fmu_path)
        self.vr = {v.name: v.valueReference for v in self.md.modelVariables}

        if print_available_vars:
            ins = [v.name for v in self.md.modelVariables if getattr(v, "causality", None) == "input"]
            outs = [v.name for v in self.md.modelVariables if getattr(v, "causality", None) == "output"]
            print("\n[FMU] Inputs:", ins[:200], ("...(truncated)" if len(ins) > 200 else ""))
            print("[FMU] Outputs:", outs[:200], ("...(truncated)" if len(outs) > 200 else ""))
            print()

        required = []
        required += self.IN_NAMES
        required += [self.tcabinet_var, self.wcomp_var]
        if self.qevap_var is not None:
            required.append(self.qevap_var)
        required += list(self.OUT_NAMES)

        missing = [n for n in required if n not in self.vr]
        if missing and self.strict_io:
            self._raise_missing(missing)

        if not self.strict_io:
            self.OUT_NAMES = [n for n in self.OUT_NAMES if n in self.vr]

        self.in_vr = [self.vr[n] for n in self.IN_NAMES if n in self.vr]
        self.out_vr = [self.vr[n] for n in self.OUT_NAMES if n in self.vr]

        self.scale_observations = bool(scale_observations)

        default_divisors = {
            self.wcomp_var: 200_000.0,
            (self.qevap_var or ""): 200_000.0,
            "Tcabinet": 10.0,
            "T_amb_out": 50.0,
        }
        if obs_divisors is not None:
            default_divisors.update({str(k): float(v) for k, v in obs_divisors.items()})

        self._obs_div = _np2.ones((len(self.OUT_NAMES),), dtype=_np2.float32)
        self._obs_div_map = {}
        for i, name in enumerate(self.OUT_NAMES):
            div = float(default_divisors.get(name, 1.0))
            if div <= 0:
                raise ValueError(f"obs_divisors['{name}'] must be > 0, got {div}")
            self._obs_div[i] = _np2.float32(div)
            self._obs_div_map[name] = div

        self.T_wmean_div = float(T_wmean_div) if float(T_wmean_div) > 0 else 10.0

        self.include_applied_action_in_obs = bool(include_applied_action_in_obs)
        self.include_window_fill_in_obs = bool(include_window_fill_in_obs)

        if Psuc_applied_div is None:
            Psuc_applied_div = (self.Psuc_ref_bounds[1] - self.Psuc_ref_bounds[0])
        if Troom_applied_div is None:
            Troom_applied_div = (self.Troom_ref_bounds_C[1] - self.Troom_ref_bounds_C[0])
        self.Psuc_applied_div = float(Psuc_applied_div) if float(Psuc_applied_div) > 0 else 1.0
        self.Troom_applied_div = float(Troom_applied_div) if float(Troom_applied_div) > 0 else 1.0

        self.action_lpf_alpha = float(action_lpf_alpha)
        if not (0.0 < self.action_lpf_alpha <= 1.0):
            raise ValueError(f"action_lpf_alpha must be in (0,1], got {self.action_lpf_alpha}")

        self.action_smoothness_coef = float(action_smoothness_coef)
        self.action_smoothness_type = str(action_smoothness_type)
        self.action_smoothness_use_scaled = bool(action_smoothness_use_scaled)
        if self.action_smoothness_type not in ("l1", "l2"):
            raise ValueError("action_smoothness_type must be 'l1' or 'l2'")

        self._time_dim = 1
        self._forecast_dim = 2 * self.forecast_hours

        extra_state_dim = 1
        if self.include_weighted_temp_in_obs:
            extra_state_dim += 1
        if self.include_window_fill_in_obs:
            extra_state_dim += 1

        action_obs_dim = 2 if self.include_applied_action_in_obs else 0
        base_dim = len(self.out_vr) + extra_state_dim + action_obs_dim

        self.observation_space = _spaces.Box(
            low=-_np2.inf,
            high=_np2.inf,
            shape=(base_dim + self._time_dim + self._forecast_dim,),
            dtype=_np2.float32,
        )

        self.fmu = None
        self.t = 0.0
        self.k = 0

        self._T_window = None
        self._T_widx = 0
        self._T_wcount = 0

        self._last_good_obs = None

        self._last_u = None
        self._last_a = None

        self._a_applied = None
        self._prev_a_for_smooth = None

        self._held_applied = None

        self._held_T_amb_C = None
        self._held_c_elec_eur_per_kwh = None

        self._amb_applied_C = None

        self._idx_Tcab_obs = self.OUT_NAMES.index(self.tcabinet_var) if self.tcabinet_var in self.OUT_NAMES else None
        self._idx_Wcomp_obs = self.OUT_NAMES.index(self.wcomp_var) if self.wcomp_var in self.OUT_NAMES else None
        self._idx_Qevap_obs = (
            self.OUT_NAMES.index(self.qevap_var)
            if (self.qevap_var is not None and self.qevap_var in self.OUT_NAMES)
            else None
        )

    def _load_excel_profiles(self):
        df = _pd.read_excel(self.excel_path)

        for col in (self.time_col, self.price_col, self.temp_col):
            if col not in df.columns:
                raise KeyError(f"Excel missing column '{col}'. Found columns: {list(df.columns)}")

        df = df[[self.time_col, self.price_col, self.temp_col]].copy()
        df[self.time_col] = _pd.to_datetime(df[self.time_col], errors="coerce")
        df = df.dropna(subset=[self.time_col])
        df = df.sort_values(self.time_col).reset_index(drop=True)

        df[self.price_col] = _pd.to_numeric(df[self.price_col], errors="coerce")
        df[self.temp_col] = _pd.to_numeric(df[self.temp_col], errors="coerce")
        df = df.dropna(subset=[self.price_col, self.temp_col]).reset_index(drop=True)

        self._times = df[self.time_col].to_numpy()
        self._temps_C_all = df[self.temp_col].to_numpy(dtype=_np2.float32)

        prices = df[self.price_col].to_numpy(dtype=_np2.float32)
        if self.price_is_eur_per_mwh:
            self._prices_eur_per_kwh_all = prices / 1000.0
        else:
            self._prices_eur_per_kwh_all = prices

        ts = _pd.Series(df[self.time_col])
        is_midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)
        day_start_idxs = _np2.where(is_midnight.to_numpy())[0].tolist()

        need_hours = self.episode_hours + self.forecast_hours + 1
        valid = []
        for i in day_start_idxs:
            j = i + need_hours
            if j >= len(df):
                continue
            t0 = df.loc[i, self.time_col]
            tj = df.loc[j, self.time_col]
            if (tj - t0) == _pd.Timedelta(hours=need_hours):
                valid.append(i)

        if len(valid) == 0:
            raise RuntimeError(
                f"No valid midnight day-starts found with {need_hours} contiguous hourly rows available."
            )

        self._day_start_idxs = _np2.asarray(valid, dtype=_np2.int64)

    def _select_random_day(self):
        idx = int(self.rng.choice(self._day_start_idxs))
        self._profile_start_idx = idx

        t0 = _pd.Timestamp(self._times[idx])
        self._profile_date_str = t0.strftime("%Y-%m-%d")

        total_hours = self.episode_hours + self.forecast_hours + 1
        sl = slice(idx, idx + total_hours)

        self._temps_C = _np2.asarray(self._temps_C_all[sl], dtype=_np2.float32)
        self._prices_eur_per_kwh = _np2.asarray(self._prices_eur_per_kwh_all[sl], dtype=_np2.float32)

        self._episode_ctrl_steps = int(round((self.episode_hours * 3600.0) / self.ctrl_dt))
        self._episode_ctrl_steps = max(1, self._episode_ctrl_steps)

    def _current_hour_index(self) -> int:
        h = int(_np2.floor(float(self.t) / 3600.0))
        return int(_np2.clip(h, 0, self.episode_hours - 1))

    def _hour_feature(self) -> float:
        h = float(self._current_hour_index())
        if self.scale_time_feature:
            denom = float(max(1, self.episode_hours - 1))
            return h / denom
        return h

    def _raw_hourly_T_amb_target(self) -> float:
        if not self.use_excel_profile:
            return float(self.T_amb_const_C)
        return float(self._temps_C[self._current_hour_index()])

    def _raw_hourly_price_target(self) -> float:
        if not self.use_excel_profile:
            return float(self.c_elec)
        return float(self._prices_eur_per_kwh[self._current_hour_index()])

    def _interp_T_amb_target(self) -> float:
        if not self.use_excel_profile:
            return float(self.T_amb_const_C)

        h = int(_np2.floor(float(self.t) / 3600.0))
        h = int(_np2.clip(h, 0, self.episode_hours - 1))
        h2 = min(h + 1, len(self._temps_C) - 1)

        t0 = 3600.0 * float(h)
        frac = float((self.t - t0) / 3600.0)
        frac = float(_np2.clip(frac, 0.0, 1.0))

        T1 = float(self._temps_C[h])
        T2 = float(self._temps_C[h2])
        return (1.0 - frac) * T1 + frac * T2

    def _compute_T_amb_now(self) -> float:
        if self.amb_smooth_mode == "interp":
            target = self._interp_T_amb_target()
        else:
            target = self._raw_hourly_T_amb_target()

        if self.amb_smooth_mode == "ema":
            if self._amb_applied_C is None:
                self._amb_applied_C = float(target)
            else:
                beta = float(self.amb_ema_beta)
                self._amb_applied_C = beta * float(target) + (1.0 - beta) * float(self._amb_applied_C)
            out = float(self._amb_applied_C)
        else:
            out = float(target)

        if self.clamp_T_amb:
            out = float(_np2.clip(out, self.T_amb_min_C, self.T_amb_max_C))

        return out

    def _compute_price_now(self) -> float:
        return float(self._raw_hourly_price_target())

    def _forecast_vec(self) -> _np2.ndarray:
        H = int(self.forecast_hours)

        if not self.use_excel_profile:
            temps = _np2.full((H,), float(self.T_amb_const_C), dtype=_np2.float32)
            prices = _np2.full((H,), float(self.c_elec), dtype=_np2.float32)
            if self.scale_forecast:
                temps = temps / _np2.float32(self.forecast_temp_div)
                prices = prices / _np2.float32(self.forecast_price_div)
            return _np2.concatenate([temps, prices], axis=0).astype(_np2.float32)

        h = self._current_hour_index()
        start = h + 1
        end = start + H

        end = int(_np2.clip(end, 0, len(self._temps_C)))
        start = int(_np2.clip(start, 0, max(0, end)))

        temps = self._temps_C[start:end]
        prices_kwh = self._prices_eur_per_kwh[start:end]

        if temps.size < H:
            pad_n = H - temps.size
            last_T = temps[-1] if temps.size > 0 else self._temps_C[min(h, len(self._temps_C) - 1)]
            last_P = (
                prices_kwh[-1]
                if prices_kwh.size > 0
                else self._prices_eur_per_kwh[min(h, len(self._prices_eur_per_kwh) - 1)]
            )
            temps = _np2.concatenate([temps, _np2.full((pad_n,), last_T, dtype=_np2.float32)], axis=0)
            prices_kwh = _np2.concatenate([prices_kwh, _np2.full((pad_n,), last_P, dtype=_np2.float32)], axis=0)

        if self.scale_forecast:
            temps = temps / _np2.float32(self.forecast_temp_div)
            prices = prices_kwh / _np2.float32(self.forecast_price_div)
        else:
            prices = prices_kwh

        return _np2.concatenate([temps.astype(_np2.float32), prices.astype(_np2.float32)], axis=0)

    def get_observation_divisors(self) -> dict[str, float]:
        return dict(self._obs_div_map)

    def _raise_missing(self, missing):
        names = list(self.vr.keys())
        lines = [f"Missing FMU variables: {missing}"]
        for m in missing:
            s = _difflib.get_close_matches(m, names, n=6, cutoff=0.6)
            if s:
                lines.append(f"  suggestions for '{m}': {s}")
        lines.append("Fix: variable names must EXACTLY match the FMU variable names.")
        raise KeyError("\n".join(lines))

    @staticmethod
    def _viol(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo - x
        if x > hi:
            return x - hi
        return 0.0

    @staticmethod
    def _viol_flag(v: float) -> float:
        return 1.0 if v > 0.0 else 0.0

    def _viol_clipped(self, x: float, lo: float, hi: float) -> float:
        v = self._viol(x, lo, hi)
        if self.violation_clip_C is None:
            return float(v)
        return float(min(float(v), float(self.violation_clip_C)))

    def _instantiate(self):
        fmu = _instantiate_fmu(
            self.unzipdir,
            self.md,
            fmi_type="CoSimulation",
            visible=False,
        )
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode()
        fmu.exitInitializationMode()
        return fmu

    def _set_fmu_inputs(self, Psuc_ref: float, Troom_ref_in: float, T_amb_C: float):
        u = [float(Psuc_ref), float(T_amb_C), float(Troom_ref_in)]
        self.fmu.setReal(self.in_vr, u)
        self._last_u = tuple(u)
        self._last_a = (float(Psuc_ref), float(Troom_ref_in))

    def _apply_action(self, action: _np2.ndarray):
        a = _np2.asarray(action, dtype=_np2.float32).reshape(-1)
        if a.size != 2:
            raise ValueError(f"Expected action shape (2,), got {a.shape}")

        Psuc_cmd = float(_np2.clip(a[0], self.Psuc_ref_bounds[0], self.Psuc_ref_bounds[1]))
        Troom_cmd = float(_np2.clip(a[1], self.Troom_ref_bounds_C[0], self.Troom_ref_bounds_C[1]))
        a_cmd = _np2.array([Psuc_cmd, Troom_cmd], dtype=_np2.float32)

        if self._a_applied is None:
            self._a_applied = a_cmd.copy()
        else:
            alpha = float(self.action_lpf_alpha)
            self._a_applied = alpha * a_cmd + (1.0 - alpha) * self._a_applied

        Psuc_ref = float(_np2.clip(self._a_applied[0], self.Psuc_ref_bounds[0], self.Psuc_ref_bounds[1]))
        Troom_ref_in = float(_np2.clip(self._a_applied[1], self.Troom_ref_bounds_C[0], self.Troom_ref_bounds_C[1]))

        self._held_applied = _np2.array([Psuc_ref, Troom_ref_in], dtype=_np2.float32)

        T_amb_C = float(self._compute_T_amb_now())
        self._set_fmu_inputs(Psuc_ref, Troom_ref_in, T_amb_C)

    def _get_obs_raw(self) -> _np2.ndarray:
        y = self.fmu.getReal(self.out_vr)
        return _np2.asarray(y, dtype=_np2.float32)

    def _extract_var(self, obs_raw: _np2.ndarray, varname: str, idx_hint: int | None) -> float:
        if idx_hint is not None:
            return float(obs_raw[idx_hint])
        if varname in self.vr:
            return float(self.fmu.getReal([self.vr[varname]])[0])
        return 0.0

    def _get_T_window_chronological(self) -> _np2.ndarray:
        if self._T_window is None or self._T_wcount == 0:
            return _np2.empty((0,), dtype=_np2.float32)

        if self._T_wcount < self.temp_window_size_steps:
            return self._T_window[:self._T_wcount].astype(_np2.float32, copy=True)

        return _np2.concatenate(
            [self._T_window[self._T_widx:], self._T_window[:self._T_widx]], axis=0
        ).astype(_np2.float32, copy=False)

    def _compute_weighted_temp_signal(self, T_hist: _np2.ndarray) -> float:
        if T_hist.size == 0:
            return 0.0

        lags_sec = (_np2.arange(T_hist.size - 1, -1, -1, dtype=_np2.float32) * float(self.sim_dt))
        lags_h = lags_sec / 3600.0

        plateau_h = float(self.temp_weighted_plateau_hours)
        decay_h = float(self.temp_weighted_decay_hours)
        floor = float(self.temp_weighted_floor)

        w = _np2.ones_like(lags_h, dtype=_np2.float32)
        older = lags_h > plateau_h
        if older.any():
            w[older] = _np2.exp(-(lags_h[older] - plateau_h) / decay_h)

        w = floor + w
        ws = float(w.sum())
        if ws <= 0.0 or not _np2.isfinite(ws):
            return float(T_hist.mean())

        return float(_np2.dot(T_hist.astype(_np2.float32, copy=False), w) / ws)

    def _update_window(self, Tcab_C: float):
        if self._T_window is None or self._T_window.size != self.temp_window_size_steps:
            self._T_window = _np2.zeros((self.temp_window_size_steps,), dtype=_np2.float32)
            self._T_widx = 0
            self._T_wcount = 0

        self._T_window[self._T_widx] = float(Tcab_C)
        self._T_widx = (self._T_widx + 1) % self.temp_window_size_steps
        self._T_wcount = min(self._T_wcount + 1, self.temp_window_size_steps)

        window_full = (self._T_wcount >= self.temp_window_size_steps)

        if self._T_wcount > 0:
            T_hist = self._get_T_window_chronological()
            Twmean_C = float(T_hist.mean())
            T_weighted_C = self._compute_weighted_temp_signal(T_hist)
        else:
            Twmean_C = float(Tcab_C)
            T_weighted_C = float(Tcab_C)

        T_weighted_violation = self._viol_clipped(T_weighted_C, self.T_wmean_min_C, self.T_wmean_max_C)

        return Twmean_C, T_weighted_C, T_weighted_violation, window_full

    def _economic_cost_and_powers(self, obs_raw: _np2.ndarray, dt_seconds: float):
        Wcomp_W = float(self._extract_var(obs_raw, self.wcomp_var, self._idx_Wcomp_obs))

        if self.qevap_var is not None and self.fan_power_coeff != 0.0:
            Qevap_W = float(self._extract_var(obs_raw, self.qevap_var, self._idx_Qevap_obs))
            Wfan_W = float(self.fan_power_coeff) * Qevap_W
        else:
            Qevap_W = (
                float(self._extract_var(obs_raw, self.qevap_var, self._idx_Qevap_obs))
                if self.qevap_var is not None
                else 0.0
            )
            Wfan_W = 0.0

        Wtotal_W = Wcomp_W + Wfan_W
        kill_power = (Wtotal_W > self.kill_if_total_power_gt)
        Wtotal_W_clip = float(_np2.clip(Wtotal_W, 0.0, self.max_total_power_W))

        if self._held_c_elec_eur_per_kwh is not None:
            price_raw = float(self._held_c_elec_eur_per_kwh)
        else:
            price_raw = float(self._compute_price_now())

        price_used = max(price_raw, -float(self.price_neg_floor_eur_per_kwh))

        Wtotal_kW = Wtotal_W_clip / 1000.0
        dt_hours = float(dt_seconds) / 3600.0

        step_cost_elec_raw_eur = price_used * Wtotal_kW * dt_hours

        power_reg_eur = 0.0
        if self.alpha_power > 0.0:
            power_reg_eur = float(self.alpha_power) * (Wtotal_kW ** 2) * dt_hours

        step_cost_total_raw_eur = float(step_cost_elec_raw_eur + power_reg_eur)
        step_cost_total_raw_eur = float(
            _np2.clip(
                step_cost_total_raw_eur,
                -float(self.max_profit_eur_per_step),
                float(self.max_step_cost_eur),
            )
        )

        step_cost_used_eur = float(self.price_reward_weight) * step_cost_total_raw_eur
        kill_cost = (step_cost_total_raw_eur > self.kill_if_step_cost_gt)

        return (
            Wcomp_W,
            Wfan_W,
            Wtotal_W_clip,
            Wtotal_kW,
            step_cost_elec_raw_eur,
            power_reg_eur,
            step_cost_total_raw_eur,
            step_cost_used_eur,
            float(Qevap_W),
            bool(kill_power or kill_cost),
            price_raw,
            price_used,
        )

    def _build_obs(self, obs_raw: _np2.ndarray, Twmean_C: float, T_weighted_C: float) -> _np2.ndarray:
        obs_scaled = (obs_raw / self._obs_div).astype(_np2.float32) if self.scale_observations else obs_raw.astype(_np2.float32)
        tw = float(Twmean_C) / self.T_wmean_div if self.scale_observations else float(Twmean_C)

        extras = [_np2.float32(tw)]

        if self.include_weighted_temp_in_obs:
            tw_weighted = float(T_weighted_C) / self.T_weighted_div if self.scale_observations else float(T_weighted_C)
            extras.append(_np2.float32(tw_weighted))

        if self.include_window_fill_in_obs:
            fill_ratio = float(self._T_wcount) / float(max(1, self.temp_window_size_steps))
            extras.append(_np2.float32(fill_ratio))

        if self.include_applied_action_in_obs:
            if self._held_applied is None:
                ps_applied = 0.0
                tr_applied = 0.0
            else:
                ps_applied = float(self._held_applied[0]) / float(self.Psuc_applied_div) if self.scale_observations else float(self._held_applied[0])
                tr_applied = float(self._held_applied[1]) / float(self.Troom_applied_div) if self.scale_observations else float(self._held_applied[1])

            extras.append(_np2.float32(ps_applied))
            extras.append(_np2.float32(tr_applied))

        extras.append(_np2.float32(self._hour_feature()))
        fvec = self._forecast_vec().astype(_np2.float32)

        extra = _np2.asarray(extras, dtype=_np2.float32)
        return _np2.concatenate([obs_scaled, extra, fvec], axis=0).astype(_np2.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = _np2.random.default_rng(seed)

        if self.use_excel_profile:
            self._select_random_day()

        last_err = None
        for attempt in range(max(1, self.max_reset_retries + 1)):
            if self.fmu is not None:
                try:
                    self.fmu.terminate()
                except Exception:
                    pass
                try:
                    self.fmu.freeInstance()
                except Exception:
                    pass

            self.fmu = self._instantiate()
            self.t = 0.0
            self.k = 0

            self._T_window = None
            self._T_widx = 0
            self._T_wcount = 0
            self._last_good_obs = None
            self._last_u = None
            self._last_a = None

            self._a_applied = None
            self._prev_a_for_smooth = None
            self._held_applied = None

            self._held_T_amb_C = None
            self._held_c_elec_eur_per_kwh = None
            self._amb_applied_C = None

            init_action = _np2.asarray([self.Psuc_ref_init, self.Troom_ref_init_C], dtype=_np2.float32)
            self._apply_action(init_action)

            if self._last_a is not None:
                self._prev_a_for_smooth = _np2.array([self._last_a[0], self._last_a[1]], dtype=_np2.float32)
            else:
                self._prev_a_for_smooth = None

            if self.use_excel_profile and self.exo_hold_at_ctrl_dt:
                self._held_T_amb_C = float(self._compute_T_amb_now())
                self._held_c_elec_eur_per_kwh = float(self._compute_price_now())

            try:
                obs_raw = self._get_obs_raw()
                if self.terminate_on_obs_nan and (not _np2.all(_np2.isfinite(obs_raw))):
                    last_err = RuntimeError("Non-finite obs on reset")
                    continue

                Tcab_C = float(self._extract_var(obs_raw, self.tcabinet_var, self._idx_Tcab_obs))
                Twmean_C, T_weighted_C, T_weighted_violation, window_full = self._update_window(Tcab_C)

                obs = self._build_obs(obs_raw, Twmean_C, T_weighted_C)
                self._last_good_obs = obs.copy()

                v_w_always = self._viol_clipped(Twmean_C, self.T_wmean_min_C, self.T_wmean_max_C)
                v_w_full = v_w_always if window_full else 0.0

                (
                    Wcomp_W, Wfan_W, Wtotal_W, Wtotal_kW,
                    step_cost_elec_raw_eur, power_reg_eur, step_cost_total_raw_eur, step_cost_used_eur,
                    Qevap_W, _kill, price_raw, price_used
                ) = self._economic_cost_and_powers(obs_raw, dt_seconds=self.sim_dt)

                info = {
                    "t": _np(self.t),
                    "k": _np(self.k),
                    "sim_dt": _np(self.sim_dt),
                    "ctrl_dt": _np(self.ctrl_dt),
                    "substeps": _np(float(self.substeps)),

                    "Tcabinet_C": _np(Tcab_C),
                    "T_wmean_C": _np(Twmean_C),
                    "T_weighted_C": _np(T_weighted_C),

                    "temp_window_size_steps": _np(float(self.temp_window_size_steps)),
                    "temp_window_seconds": _np(float(self.temp_window_seconds)),
                    "temp_window_filled": _np(1.0 if window_full else 0.0),
                    "include_weighted_temp_in_obs": _np(1.0 if self.include_weighted_temp_in_obs else 0.0),

                    "termination_buffer_steps": _np(float(self.termination_buffer_steps)),
                    "termination_buffer_seconds": _np(float(self.termination_buffer_seconds)),
                    "termination_buffer_active": _np(1.0 if (self.k < self.termination_buffer_steps) else 0.0),

                    "T_wmean_violation": _np(v_w_always),
                    "T_wmean_violation_always": _np(v_w_always),
                    "T_wmean_violation_full": _np(v_w_full),
                    "T_wmean_violated_always": _np(self._viol_flag(v_w_always)),
                    "T_wmean_violated_full": _np(self._viol_flag(v_w_full)),

                    "T_weighted_violation": _np(T_weighted_violation),
                    "T_weighted_violation_always": _np(T_weighted_violation),
                    "T_weighted_violated_always": _np(self._viol_flag(T_weighted_violation)),

                    "Twmean_min_C": _np(self.T_wmean_min_C),
                    "Twmean_max_C": _np(self.T_wmean_max_C),
                    "temp_weighted_plateau_hours": _np(float(self.temp_weighted_plateau_hours)),
                    "temp_weighted_decay_hours": _np(float(self.temp_weighted_decay_hours)),
                    "temp_weighted_floor": _np(float(self.temp_weighted_floor)),

                    "Twmean_violation_clip_C": _np(-1.0 if self.violation_clip_C is None else float(self.violation_clip_C)),

                    "Wcomp_W": _np(Wcomp_W),
                    "Wfan_W": _np(Wfan_W),
                    "Wtotal_W": _np(Wtotal_W),
                    "Wtotal_kW": _np(Wtotal_kW),
                    "Qevap_W": _np(Qevap_W),

                    "c_elec_eur_per_kwh": _np(float(price_raw)),
                    "c_elec_used_eur_per_kwh": _np(float(price_used)),

                    "step_cost_eur_raw": _np(float(step_cost_elec_raw_eur)),
                    "step_power_reg_eur": _np(float(power_reg_eur)),
                    "step_cost_eur_total_raw": _np(float(step_cost_total_raw_eur)),
                    "step_cost_eur_used": _np(float(step_cost_used_eur)),
                    "step_cost_eur": _np(float(step_cost_used_eur)),

                    "price_reward_weight": _np(float(self.price_reward_weight)),

                    "alive_reward_per_ctrl_step": _np(self.alive_reward_per_ctrl_step),
                    "early_end_penalty": _np(self.early_end_penalty),

                    "term_code": _np(0.0),
                    "episode_end": _np(0.0),
                    "episode_len_steps": _np(0.0),
                    "episode_len_seconds": _np(0.0),

                    "truncation_penalty": _np(self.truncation_penalty),
                    "reward_total": _np(0.0),

                    "action_lpf_alpha": _np(float(self.action_lpf_alpha)),
                    "smooth_pen": _np(0.0),
                    "smooth_coef": _np(float(self.action_smoothness_coef)),
                    "smooth_reward_contrib": _np(0.0),

                    "reset_retry_attempt": _np(float(attempt)),

                    "held_T_amb_C": _np(-999.0 if self._held_T_amb_C is None else float(self._held_T_amb_C)),
                    "held_price_eur_per_kwh": _np(-999.0 if self._held_c_elec_eur_per_kwh is None else float(self._held_c_elec_eur_per_kwh)),
                }

                if self.use_excel_profile:
                    info.update({
                        "profile_day": self._profile_date_str,
                        "hour_idx": _np(float(self._current_hour_index())),
                    })

                return obs, info

            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"reset() failed after {self.max_reset_retries + 1} attempts. last_err={last_err}")

    def step(self, action):
        if self.fmu is None:
            raise RuntimeError("Call reset() before step().")

        term_code = 0
        terminated = False
        truncated = False

        self._apply_action(action)

        if self.use_excel_profile and self.exo_hold_at_ctrl_dt:
            self._held_T_amb_C = float(self._compute_T_amb_now())
            self._held_c_elec_eur_per_kwh = float(self._compute_price_now())
        else:
            self._held_T_amb_C = None
            self._held_c_elec_eur_per_kwh = None

        smooth_pen = 0.0
        if self.action_smoothness_coef > 0.0 and self._last_a is not None:
            a_now = _np2.array([self._last_a[0], self._last_a[1]], dtype=_np2.float32)

            if self._prev_a_for_smooth is None:
                self._prev_a_for_smooth = a_now.copy()

            a_prev = self._prev_a_for_smooth

            if self.action_smoothness_use_scaled:
                ps_range = (self.Psuc_ref_bounds[1] - self.Psuc_ref_bounds[0]) + 1e-8
                tr_range = (self.Troom_ref_bounds_C[1] - self.Troom_ref_bounds_C[0]) + 1e-8

                a_now_s = _np2.array([
                    (a_now[0] - self.Psuc_ref_bounds[0]) / ps_range,
                    (a_now[1] - self.Troom_ref_bounds_C[0]) / tr_range,
                ], dtype=_np2.float32)
                a_prev_s = _np2.array([
                    (a_prev[0] - self.Psuc_ref_bounds[0]) / ps_range,
                    (a_prev[1] - self.Troom_ref_bounds_C[0]) / tr_range,
                ], dtype=_np2.float32)
                da = a_now_s - a_prev_s
            else:
                da = a_now - a_prev

            if self.action_smoothness_type == "l1":
                smooth_pen = float(_np2.abs(da).sum())
            else:
                smooth_pen = float((da * da).sum())

            self._prev_a_for_smooth = a_now.copy()

        reward_total = 0.0
        step_cost_eur_raw_sum = 0.0
        step_cost_eur_total_raw_sum = 0.0
        step_cost_eur_used_sum = 0.0

        obs_raw_last = None
        Tcab_C = 0.0
        Twmean_C = 0.0
        T_weighted_C = 0.0
        T_weighted_violation = 0.0
        window_full = False

        Wcomp_W_last = 0.0
        Wfan_W_last = 0.0
        Wtotal_W_last = 0.0
        Wtotal_kW_last = 0.0
        Qevap_W_last = 0.0
        price_raw_last = float(self._held_c_elec_eur_per_kwh) if (self._held_c_elec_eur_per_kwh is not None) else float(self._compute_price_now())
        price_used_last = max(price_raw_last, -float(self.price_neg_floor_eur_per_kwh))

        try:
            for sub in range(self.substeps):
                if self._held_applied is not None:
                    if self._held_T_amb_C is not None:
                        T_amb_C = float(self._held_T_amb_C)
                    else:
                        T_amb_C = float(self._compute_T_amb_now())
                    self._set_fmu_inputs(float(self._held_applied[0]), float(self._held_applied[1]), T_amb_C)

                self.fmu.doStep(currentCommunicationPoint=self.t, communicationStepSize=self.sim_dt)
                self.t += self.sim_dt

                obs_raw_last = self._get_obs_raw()

                if self.terminate_on_obs_nan and (not _np2.all(_np2.isfinite(obs_raw_last))):
                    term_code = 97
                    terminated = True
                    reward_total += float(self.fmu_failure_reward)
                    break

                Tcab_C = float(self._extract_var(obs_raw_last, self.tcabinet_var, self._idx_Tcab_obs))
                Twmean_C, T_weighted_C, T_weighted_violation, window_full = self._update_window(Tcab_C)

                (
                    Wcomp_W_last, Wfan_W_last, Wtotal_W_last, Wtotal_kW_last,
                    step_cost_elec_raw_eur, power_reg_eur, step_cost_total_raw_eur, step_cost_used_eur,
                    Qevap_W_last, kill_absurd, price_raw_last, price_used_last
                ) = self._economic_cost_and_powers(obs_raw_last, dt_seconds=self.sim_dt)

                step_cost_eur_raw_sum += float(step_cost_elec_raw_eur)
                step_cost_eur_total_raw_sum += float(step_cost_total_raw_eur)
                step_cost_eur_used_sum += float(step_cost_used_eur)

                reward_total += -float(step_cost_used_eur)

                if kill_absurd:
                    term_code = 98
                    terminated = True
                    reward_total += float(self.fmu_failure_reward)
                    break

        except Exception:
            term_code = 99
            terminated = True
            reward_total += float(self.fmu_failure_reward)

        self.k += 1
        reward_total += self.alive_reward_per_ctrl_step

        smooth_reward_contrib = 0.0
        if self.action_smoothness_coef > 0.0:
            smooth_reward_contrib = -float(self.action_smoothness_coef) * float(smooth_pen)
            reward_total += smooth_reward_contrib

        wmean_v_always = self._viol_clipped(Twmean_C, self.T_wmean_min_C, self.T_wmean_max_C)
        wmean_v_full = wmean_v_always if window_full else 0.0
        buffer_active = (self.k < self.termination_buffer_steps)

        if (not terminated) and (not buffer_active) and self.terminate_on_window_mean_violation and (wmean_v_always > 0.0):
            terminated = True
            term_code = 3

        if self.use_excel_profile and (self._episode_ctrl_steps is not None):
            truncated = (self.k >= min(self.horizon, self._episode_ctrl_steps))
        else:
            truncated = (self.k >= self.horizon)

        if terminated and self.early_end_penalty != 0.0:
            remaining = float(max(0, self.horizon - self.k))
            reward_total += self.early_end_penalty * remaining

        if truncated and self.truncation_penalty != 0.0:
            reward_total += self.truncation_penalty

        if (not terminated) and (obs_raw_last is not None):
            obs = self._build_obs(obs_raw_last, Twmean_C, T_weighted_C)
            self._last_good_obs = obs.copy()
        else:
            obs = self._last_good_obs.copy() if self._last_good_obs is not None else _np2.zeros(
                (int(self.observation_space.shape[0]),), dtype=_np2.float32
            )

        episode_end = float(terminated or truncated)

        info = {
            "t": _np(self.t),
            "k": _np(self.k),
            "sim_dt": _np(self.sim_dt),
            "ctrl_dt": _np(self.ctrl_dt),
            "substeps": _np(float(self.substeps)),

            "Tcabinet_C": _np(Tcab_C),
            "T_wmean_C": _np(Twmean_C),
            "T_weighted_C": _np(T_weighted_C),

            "temp_window_size_steps": _np(float(self.temp_window_size_steps)),
            "temp_window_seconds": _np(float(self.temp_window_seconds)),
            "temp_window_filled": _np(1.0 if window_full else 0.0),
            "include_weighted_temp_in_obs": _np(1.0 if self.include_weighted_temp_in_obs else 0.0),

            "termination_buffer_steps": _np(float(self.termination_buffer_steps)),
            "termination_buffer_seconds": _np(float(self.termination_buffer_seconds)),
            "termination_buffer_active": _np(1.0 if buffer_active else 0.0),

            "T_wmean_violation": _np(wmean_v_always),
            "T_wmean_violation_always": _np(wmean_v_always),
            "T_wmean_violation_full": _np(wmean_v_full),
            "T_wmean_violated_always": _np(self._viol_flag(wmean_v_always)),
            "T_wmean_violated_full": _np(self._viol_flag(wmean_v_full)),

            "T_weighted_violation": _np(T_weighted_violation),
            "T_weighted_violation_always": _np(T_weighted_violation),
            "T_weighted_violated_always": _np(self._viol_flag(T_weighted_violation)),

            "Twmean_min_C": _np(self.T_wmean_min_C),
            "Twmean_max_C": _np(self.T_wmean_max_C),
            "temp_weighted_plateau_hours": _np(float(self.temp_weighted_plateau_hours)),
            "temp_weighted_decay_hours": _np(float(self.temp_weighted_decay_hours)),
            "temp_weighted_floor": _np(float(self.temp_weighted_floor)),

            "Twmean_violation_clip_C": _np(-1.0 if self.violation_clip_C is None else float(self.violation_clip_C)),

            "Wcomp_W": _np(Wcomp_W_last),
            "Wfan_W": _np(Wfan_W_last),
            "Wtotal_W": _np(Wtotal_W_last),
            "Wtotal_kW": _np(Wtotal_kW_last),
            "Qevap_W": _np(Qevap_W_last),

            "c_elec_eur_per_kwh": _np(float(price_raw_last)),
            "c_elec_used_eur_per_kwh": _np(float(price_used_last)),

            "step_cost_eur_raw": _np(float(step_cost_eur_raw_sum)),
            "step_cost_eur_total_raw": _np(float(step_cost_eur_total_raw_sum)),
            "step_cost_eur_used": _np(float(step_cost_eur_used_sum)),
            "step_cost_eur": _np(float(step_cost_eur_used_sum)),

            "price_reward_weight": _np(float(self.price_reward_weight)),

            "alive_reward_per_ctrl_step": _np(self.alive_reward_per_ctrl_step),
            "early_end_penalty": _np(self.early_end_penalty),

            "term_code": _np(float(term_code)),
            "episode_end": _np(episode_end),
            "episode_len_steps": _np(float(self.k)),
            "episode_len_seconds": _np(float(self.k * self.ctrl_dt)),

            "truncation_penalty": _np(self.truncation_penalty),
            "reward_total": _np(float(reward_total)),

            "action_lpf_alpha": _np(float(self.action_lpf_alpha)),
            "smooth_pen": _np(float(smooth_pen)),
            "smooth_coef": _np(float(self.action_smoothness_coef)),
            "smooth_reward_contrib": _np(float(smooth_reward_contrib)),

            "reset_retry_attempt": _np(0.0),

            "held_T_amb_C": _np(-999.0 if self._held_T_amb_C is None else float(self._held_T_amb_C)),
            "held_price_eur_per_kwh": _np(-999.0 if self._held_c_elec_eur_per_kwh is None else float(self._held_c_elec_eur_per_kwh)),
        }

        if self.use_excel_profile:
            info.update({
                "profile_day": self._profile_date_str,
                "hour_idx": _np(float(self._current_hour_index())),
            })

        return obs, float(reward_total), terminated, truncated, info

    def close(self):
        if self.fmu is not None:
            try:
                self.fmu.terminate()
            except Exception:
                pass
            try:
                self.fmu.freeInstance()
            except Exception:
                pass
        self.fmu = None