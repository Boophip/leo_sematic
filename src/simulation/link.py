"""SNR-AMC link lookup and communication delay primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


class IllegalActionError(ValueError):
    """Raised when an offload action violates physical link constraints."""


@dataclass(frozen=True)
class AmcEntry:
    """One SNR threshold and its fixed AMC spectral efficiency."""

    min_snr_db: float
    spectral_efficiency_bps_hz: float


@dataclass(frozen=True)
class LinkState:
    """Per-slot offload link state exposed to scheduling policies."""

    visible: bool
    snr_db: float
    bandwidth_hz: float
    propagation_delay_ms: float = 0.0


DEFAULT_AMC_TABLE: tuple[AmcEntry, ...] = (
    AmcEntry(min_snr_db=0.0, spectral_efficiency_bps_hz=0.5),
    AmcEntry(min_snr_db=4.0, spectral_efficiency_bps_hz=1.0),
    AmcEntry(min_snr_db=8.0, spectral_efficiency_bps_hz=2.0),
    AmcEntry(min_snr_db=12.0, spectral_efficiency_bps_hz=3.0),
    AmcEntry(min_snr_db=16.0, spectral_efficiency_bps_hz=4.5),
    AmcEntry(min_snr_db=20.0, spectral_efficiency_bps_hz=6.0),
)


def spectral_efficiency_for_snr(
    snr_db: float,
    table: Sequence[AmcEntry] = DEFAULT_AMC_TABLE,
) -> float:
    """O(1)-style SNR-to-AMC lookup used instead of continuous power solving."""

    if not table:
        raise ValueError("AMC table must not be empty")
    selected = 0.0
    for entry in sorted(table, key=lambda item: item.min_snr_db):
        if snr_db >= entry.min_snr_db:
            selected = entry.spectral_efficiency_bps_hz
        else:
            break
    return selected


def throughput_bps(
    link: LinkState,
    table: Sequence[AmcEntry] = DEFAULT_AMC_TABLE,
) -> float:
    """Return link throughput; invisible or unsupported links are illegal actions."""

    if not link.visible:
        raise IllegalActionError("offload link is not visible")
    if link.bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    spectral_efficiency = spectral_efficiency_for_snr(link.snr_db, table)
    if spectral_efficiency <= 0:
        raise IllegalActionError("SNR is below the supported AMC table")
    return link.bandwidth_hz * spectral_efficiency


def transmission_delay_ms(
    byte_count: int | float,
    link: LinkState,
    table: Sequence[AmcEntry] = DEFAULT_AMC_TABLE,
) -> float:
    """Transmit byte_count over the fixed-power AMC link."""

    if byte_count < 0:
        raise ValueError("byte_count must be non-negative")
    if byte_count == 0:
        return 0.0
    return (float(byte_count) * 8.0 / throughput_bps(link, table)) * 1000.0


def offload_communication_delay_ms(
    payload_bytes: int | float,
    link: LinkState,
    *,
    result_bytes: int | float = 64,
    table: Sequence[AmcEntry] = DEFAULT_AMC_TABLE,
) -> float:
    """Forward ROI payload plus propagation and lightweight result return."""

    forward_ms = transmission_delay_ms(payload_bytes, link, table)
    return_ms = transmission_delay_ms(result_bytes, link, table)
    return forward_ms + link.propagation_delay_ms + return_ms


def communication_energy_joules(tx_power_w: float, transmit_delay_ms: float) -> float:
    """Linear transmit energy under the fixed-power radio assumption."""

    if tx_power_w < 0:
        raise ValueError("tx_power_w must be non-negative")
    if transmit_delay_ms < 0:
        raise ValueError("transmit_delay_ms must be non-negative")
    return tx_power_w * (transmit_delay_ms / 1000.0)
