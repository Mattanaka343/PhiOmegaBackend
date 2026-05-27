from db import queries
from utils import period_to_date, period_to_previous_dates
from datetime import date

async def get_displayed_orders(status:str,date:date,min_sale:float,client:str,sales_rep:str) -> dict:

    return await queries.fetch_displayed_orders(status,date,min_sale,client,sales_rep)


async def get_clients_db(order_by:str,order_how:str) -> dict:
    return await queries.fetch_clients_db(order_by,order_how)


async def get_users_db(order_by:str,order_how:str) -> dict:
    return await queries.fetch_users_db(order_by,order_how)
