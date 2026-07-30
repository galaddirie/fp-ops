"""Executable example of the FP-Ops 0.3 unary pipeline interface."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypedDict

from fp_ops import (
    EnvKey,
    Environment,
    Ok,
    build,
    environment_operation,
    operation,
    traverse,
)


@dataclass
class Pricing:
    tax_rate: float


class Item(TypedDict):
    name: str
    price: float


class Payload(TypedDict):
    items: list[Item]


class PricedItem(TypedDict):
    name: str
    total: float


@dataclass
class Summary:
    name: str
    total: float


PRICING = EnvKey("pricing", Pricing)


@operation
def read_items(payload: Payload) -> list[Item]:
    return payload["items"]


@environment_operation(PRICING)
def price_item(item: Item, pricing: Pricing) -> PricedItem:
    subtotal = item["price"]
    return {
        "name": item["name"],
        "total": round(subtotal * (1 + pricing.tax_rate), 2),
    }


def read_name(item: PricedItem) -> str:
    return item["name"]


def read_total(item: PricedItem) -> float:
    return item["total"]


summarize_item = build(
    {
        "name": read_name,
        "total": read_total,
    },
    Summary,
)

pipeline = read_items.then(traverse(price_item.then(summarize_item)))


async def main() -> None:
    environment = Environment().with_value(PRICING, Pricing(tax_rate=0.13))
    result = await pipeline.run(
        {"items": [{"name": "Book", "price": 10.0}]},
        environment=environment,
    )
    assert result == Ok([Summary(name="Book", total=11.3)])
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
