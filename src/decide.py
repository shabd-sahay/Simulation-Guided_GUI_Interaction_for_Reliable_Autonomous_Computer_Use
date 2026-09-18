"""
The decision layer.

This is a business rule, not a model. It takes a forecast and turns it into an
order quantity and a safety buffer. Keeping it in its own file is deliberate:
the forecast is learned from data and can be evaluated, whereas the thresholds
below are policy choices someone picked. Blurring the two is how a project ends
up claiming a model "decides" when really an if-statement does.

Change these numbers freely - just do not call the result a model output.
"""

import numpy as np
import pandas as pd

# Policy knobs. Stated here rather than buried inline so they can be argued
# about, tuned, or replaced by whoever actually owns the inventory budget.
FAST_MOVER_THRESHOLD = 500.0   # forecast units above which an item is "fast"
SLOW_COVER = 0.90              # order 90% of forecast for slow movers
FAST_COVER = 1.10              # order 110% of forecast for fast movers
SLOW_BUFFER = 0.25             # larger relative buffer: harder to restock fast
FAST_BUFFER = 0.10


def decide(forecast, uncertainty=None, min_order=1):
    """
    forecast    : predicted demand for the coming period
    uncertainty : optional spread estimate; when supplied, the safety buffer
                  grows with it, because an uncertain forecast deserves more
                  slack than a confident one.
    min_order   : smallest order worth placing at all, in whole units. A
                  forecast that rounds to a fraction of a unit isn't a real
                  order - it means "not worth stocking," which is reported
                  as its own tier rather than a fractional quantity.
    """
    forecast = np.asarray(forecast, dtype=float)
    fast = forecast >= FAST_MOVER_THRESHOLD

    order = np.where(fast, forecast * FAST_COVER, forecast * SLOW_COVER)
    buffer_frac = np.where(fast, FAST_BUFFER, SLOW_BUFFER)

    if uncertainty is not None:
        unc = np.asarray(uncertainty, dtype=float)
        # Scale the buffer by relative uncertainty, capped so a single noisy
        # item cannot demand an unbounded reserve.
        rel = np.clip(unc / np.maximum(forecast, 1.0), 0, 1.0)
        buffer_frac = buffer_frac * (1.0 + rel)

    # Round to whole units - you can't order 0.0078 bottles - and anything
    # forecast below a full unit is reclassified rather than shown as a
    # fraction, or as "order 1" for something that isn't really moving.
    negligible = forecast < min_order
    order_whole = np.where(negligible, 0, np.ceil(order).astype(int))
    buffer_whole = np.where(negligible, 0, np.ceil(order * buffer_frac).astype(int))

    tier = np.where(fast, "fast-mover", "slow-mover")
    tier = np.where(negligible, "negligible demand", tier)
    policy = np.where(fast, "stock ahead", "lean stock, quick reorder")
    policy = np.where(negligible, "do not stock / order on demand", policy)
    order_whole = np.where(negligible, 0, order_whole)
    buffer_whole = np.where(negligible, 0, buffer_whole)

    return pd.DataFrame({
        "forecast": np.round(forecast, 1),
        "order_quantity": order_whole,
        "safety_buffer": buffer_whole,
        "tier": tier,
        "policy": policy,
    })
