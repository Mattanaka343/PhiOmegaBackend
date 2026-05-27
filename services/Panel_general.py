from db import queries
from utils import period_to_date, period_to_previous_dates



async def get_kpi(period:str,) -> dict:
    since, until = period_to_date(period)
    prev_since, prev_until = period_to_previous_dates(period)

    current = queries.fetch_kpi(since,until)
    previous = queries.fetch_kpi(prev_since,prev_until)

    def val(row, key):
        return row[key] or 0 if row else 0

    kpi_delta = _pct_change(val(previous, "kpi"),   val(current, "kpi"))

    return {
        "kpi": val(current, "kpi"),
        "kpi_delta": kpi_delta
    }



async def get_top_client_bar_chart(period:str,limit:int) -> dict:
    since, until = period_to_date(period)

    return await queries.fetch_top_client_bar_chart(since,until,limit)

async def get_status_pie_graph(period:str,limit:int) -> dict:
    since, until = period_to_date(period)

    return await queries.fetch_status_pie_graph(since,until,limit)

async def get_best_sales_reps(period:str,limit:int) -> dict:
    since, until = period_to_date(period)

    return await queries.fetch_best_sales_reps(since,until,limit)


async def get_most_orders_clients(period:str,limit:int) -> dict:
    since, until = period_to_date(period)

    return await queries.fetch_most_orders_clients(since,until,limit)




def _pct_change(old: float, new: float) -> float:
    if not old:
        return 0.0
    return round((new - old) / old * 100, 1)

