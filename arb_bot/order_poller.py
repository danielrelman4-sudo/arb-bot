"""Post-submission order status polling (Phase 0D).

After placing an order, poll ``get_order_status()`` until the order reaches
a terminal state or timeout.  On timeout with ``cancel_on_poll_timeout``
enabled, attempt to cancel the order.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from arb_bot.exchanges.base import ExchangeAdapter
from arb_bot.models import OrderState, OrderStatus

LOGGER = logging.getLogger(__name__)

_TERMINAL_STATES = frozenset({
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.EXPIRED,
})


@dataclass(frozen=True)
class PollResult:
    """Outcome of polling an order to terminal state."""
    order_id: str
    final_status: Optional[OrderStatus]
    timed_out: bool = False
    cancelled_on_timeout: bool = False


async def poll_order_until_terminal(
    adapter: ExchangeAdapter,
    order_id: str,
    *,
    poll_interval: float = 1.0,
    timeout: float = 15.0,
    cancel_on_timeout: bool = True,
) -> PollResult:
    """Poll an order's status until it reaches a terminal state or timeout.

    Parameters
    ----------
    adapter:
        The exchange adapter that placed the order.
    order_id:
        The order ID returned from ``place_single_order``.
    poll_interval:
        Seconds between status checks.
    timeout:
        Maximum seconds to wait before giving up.
    cancel_on_timeout:
        If True and polling times out, attempt to cancel the order.

    Returns
    -------
    PollResult with the final observed status and whether timeout/cancel occurred.
    """
    deadline = time.monotonic() + timeout
    last_status: Optional[OrderStatus] = None

    while time.monotonic() < deadline:
        try:
            status = await adapter.get_order_status(order_id)
        except Exception as exc:
            LOGGER.warning("poll_order error for %s: %s", order_id, exc)
            await asyncio.sleep(poll_interval)
            continue

        if status is None:
            # Adapter doesn't support status polling; return immediately.
            return PollResult(order_id=order_id, final_status=None, timed_out=False)

        last_status = status

        if status.state in _TERMINAL_STATES:
            return PollResult(order_id=order_id, final_status=status)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(poll_interval, remaining))

    # Timed out — order is still open or in unknown state.
    LOGGER.warning("Order %s polling timed out after %.1fs", order_id, timeout)

    cancelled = False
    if cancel_on_timeout:
        try:
            cancelled = await adapter.cancel_order(order_id)
            if cancelled:
                LOGGER.info("Order %s cancelled on poll timeout", order_id)
                # Fetch final status after cancel.
                try:
                    last_status = await adapter.get_order_status(order_id)
                except Exception:
                    pass
        except Exception as exc:
            LOGGER.warning("Failed to cancel order %s on timeout: %s", order_id, exc)

    return PollResult(
        order_id=order_id,
        final_status=last_status,
        timed_out=True,
        cancelled_on_timeout=cancelled,
    )
