"""Crypto prediction trading module for Kalshi short-term binary markets.

Standalone subpackage that uses Monte Carlo price path simulation to find
edge on Kalshi crypto up/down and above/below contracts.  Reuses the
existing Kalshi adapter, Kelly sizing, and risk infrastructure.

Usage::

    python3 -m arb_bot.crypto --paper --duration-minutes 30
"""
