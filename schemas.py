from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional


class TradeOrder(BaseModel):
    """Validated trade order — the orchestrator must provide all required fields."""
    ticker: str = Field(pattern=r'^[A-Z]{1,5}(-[A-Z])?$')
    qty: int = Field(gt=0, le=100)
    side: Literal['buy', 'sell']
    order_type: Literal['market', 'limit', 'stop_loss'] = 'market'
    limit_price: Optional[float] = Field(default=None, gt=0)
    stop_price: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode='after')
    def check_prices(self):
        if self.order_type == 'limit' and self.limit_price is None:
            raise ValueError('limit_price is required for limit orders')
        if self.order_type == 'stop_loss' and self.stop_price is None:
            raise ValueError('stop_price is required for stop_loss orders')
        return self


class RiskApproval(BaseModel):
    """Structured response from the Risk Manager when evaluating a trade."""
    decision: Literal['APPROVE', 'REJECT']
    recommended_qty: Optional[int] = Field(default=None, gt=0, le=100)
    reason: str
    warnings: list[str] = []
