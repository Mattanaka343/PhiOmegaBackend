import aiomysql
import os 


from dotenv import load_dotenv
from datetime import date

DATABASE_HOST = os.getenv('DATABASE_HOST')
DATABASE_USER = os.getenv('DATABASE_USERNAME')
DATABASE_NAME = os.getenv('DATABASE_NAME')
DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD')


async def get_conn():
    """
    Connects to the database stored in a mysql instance.
    """
    return await aiomysql.connect(
        host = DATABASE_HOST,
        user = DATABASE_USER,
        password = DATABASE_PASSWORD,
        db = DATABASE_NAME,
        cursorclass = aiomysql.DictCursor)

async def fetch_displayed_orders(status:str, date:date, min_sale:float, client:str, sales_rep: str) -> dict:
    """ 
    
    """
    conn = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute("""
        SELECT 
            t.row_hash                          AS ID,
            t.fecha_creacion                    AS Fecha,
            e.estatus                           AS Estatus,
            t.venta_mxn                         AS Venta_MXN,
            CONCAT(o.ciudad,', ',o.estado)      AS Origen,
            CONCAT(d.ciudad,', ',d.estado)      AS Destino,
            c.nombre                            AS Cliente,
            p.nombre                            AS SalesRep
        FROM transacciones t
        JOIN clientes c
            ON t.id_cliente = c.id
        JOIN estatus e
            ON t.id_estatus = e.id
        JOIN lugares o
            ON t.id_origen = o.id
        JOIN lugares d 
            ON t.id_destino = d.id
        JOIN personas p
            ON t.sales_rep_id = p.id
                          
        WHERE (%s = 'all' OR e.estatus = %s)
            AND (%s = 'all' OR t.fecha_creacion = %s)
            AND (%s = 'none' OR t.venta_mxn > %s)
            AND (%s = 'all' OR  c.nombre = %s)
            AND (%s = 'all' OR p.nombre = %s)
        """,(status,status,date,date,min_sale,min_sale,client,client,sales_rep,sales_rep))
        rows = await cur.fetchall()
        conn.close()

        return rows
    
async def fetch_kpi():
    """
    """
    conn = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute(
        """
        SELECT
            COUNT( DISTINCT)
            
        """
        )