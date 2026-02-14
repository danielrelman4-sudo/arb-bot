from __future__ import annotations

from fastapi import APIRouter

from db_reader import DbReader

router = APIRouter(prefix="/api/analytics", tags=["analytics"])
db = DbReader()


@router.get("/daily-pnl")
async def daily_pnl(limit: int = 30):
    return db.daily_pnl(limit)


@router.get("/by-lane")
async def pnl_by_lane():
    return db.pnl_by_lane()


@router.get("/by-bucket")
async def pnl_by_bucket(limit: int = 50):
    return db.pnl_by_bucket(limit)


@router.get("/equity-curve")
async def equity_curve():
    return db.equity_curve()


@router.get("/positions")
async def positions():
    return db.open_positions()


@router.get("/recent-trades")
async def recent_trades(limit: int = 50):
    return db.recent_trades(limit)
