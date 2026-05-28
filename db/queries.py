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
            t.row_hash                          AS id,
            t.fecha_creacion                    AS fecha,
            e.estatus                           AS estatus,
            t.venta_mxn                         AS venta_mxn,
            CONCAT(o.ciudad,', ',o.estado)      AS origen,
            CONCAT(d.ciudad,', ',d.estado)      AS destino,
            c.nombre                            AS cliente,
            p.nombre                            AS sales_rep
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
            AND (%s = 'all' OR p.nombre = %s); 
        """,(status,status,date,date,min_sale,min_sale,client,client,sales_rep,sales_rep))
        rows = await cur.fetchall()
    conn.close()

    return rows
    
async def fetch_kpi(since: date, until:date) -> dict:
    """
    """
    conn = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute(
        """
        SELECT COUNT(DISTINCT t.id_cliente) AS unique_clients_month,
            COUNT(CASE WHEN e.estatus = 'Delivered' OR e.estatus = 'Delivered Without POD' THEN 1 END) AS delivered_orders,
            SUM(t.utilidad_mx) AS month_profit

        FROM transacciones t
        JOIN estatus e
            ON t.id_estatus = e.id

        WHERE t.fecha_creacion >= %s
            AND t.fecha_cración < %s;
        """, (since,until)
        )

async def fetch_top_client_bar_chart(since: date, until: date, limit:int) -> dict:
    """
    """
    conn = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute(
        """
        SELECT SUM(t.utilidad_mxn) as utilidad,
            c.nombre as cliente

        FROM transacciones t
        JOIN clientes c
            ON t.id_cliente = c.id 

        WHERE t.fecha_creacion >= %s
            AND t.fecha_creacion < %s

        GROUP BY c.nombre

        ORDER BY utilidad DESC
        LIMIT {limit} ;
        """, (since, until))

        rows = await cur.fetchall()
    
    conn.close()
    return rows

async def fetch_status_pie_graph(since: date, until:date) -> dict:
    """
    """

    conn = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute(
        """
        SELECT e.estatus    AS estatus,
            COUNT(*)        AS total
        FROM transacciones t
        JOIN estatus e 
            ON t.estatus_id = e.id
        
        WHERE t.fecha_creacion >= %s
            AND t.fecha_creacion < %s

        GROUP BY e.estatus
        ORDER BY total DESC;
        """, (since, until)
        )

        rows = await cur.fetchall()
    
    conn.close()
    return rows 

async def fetch_best_sales_reps(since: date, until: date, limit:int) -> dict:
    """
    """

    conn = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute(
        """
        SELECT SUM(t.utilidad_mxn) as utilidad,
            u.nombre AS nombre

        FROM transacciones t 
        JOIN usuarios u
            ON t.sales_rep_id = u.id
        
        WHERE t.fecha_creacion >= %s
            AND t.fecha_creacion < %s
        
        GROUP BY nombre
        ORDER BY utilidad DESC
        LIMIT {limit} ;
        """, (since,until,limit)
        )

        rows = await cur.fetchall()

    conn.close()
    return rows

async def fetch_most_orders_clients(since: date, until: date, limit: int) -> dict:
    """
    """

    conn = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute(
        """
        SELECT COUNT(*) AS total_orders,
            c.nombre as cliente

        FROM transacciones t
        JOIN clientes c
            ON t.id_cliente = c.id

        WHERE t.fecha_creacion >= %s
            AND t.fecha_creacion < %s

        GROUP BY cliente
        ORDER BY total_orders DESC
        LIMIT {limit} ;
        """, (since,until,limit)
        )

        rows = await cur.fetchall()

    conn.close()
    return rows

async def fetch_clients_db( order_by: str, order_how: str):
    """
    """
    conn = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute(
        f"""
        SELECT c.nombre             AS cliente,
            COUNT(*)                AS ordenes,
            SUM(t.utilidad_mxn)     AS utilidad


        FROM transacciones t
        JOIN clientes c
            ON t.id_cliente = c.id

        GROUP BY cliente
        ORDER BY {order_by} {order_how};
        """
        )

        rows = await cur.fetchall()
    
    conn.close()
    return rows

async def fetch_users_db(order_by: str, order_how:str):
    """
    """
    conn  = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute(
        """
        SELECT u.nombre         AS sales_rep,
            COUNT(*)            AS ordenes,
            SUM(t.utilidad_mxn) AS utilidad,
            u.id                AS id,
            u.correo            AS correo
        
        FROM transacciones t
        JOIN usuarios u
            ON t.sales_rep_id = u.id
        
        GROUP BY sales_rep
        ORDER BY {order_by} {order_how};
        """
        )

        rows = await cur.fetchall()

    conn.close()
    return rows



async def add_client_query(name: str):
    """
    """
    conn = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute(
        """
        INSERT INTO clientes (nombre)
        VALUES
        (%s);
        """, (name,)
        )
    
    await conn.commit()
    conn.close()




async def add_sales_rep_query(name: str, mail:str):
    """
    """
    conn = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute(
        """
        INSERT INTO usuarios (nombre, correo, cargo)
        VALUES
        (%s,%s,'sales rep');
        """, (name, mail)
        )
    
    await conn.commit()
    conn.close()




async def add_place_query(ciudad: str, estado:str, latitud:float, longitud:float):
    """
    """
    conn = await get_conn()

    async with conn.cursor() as cur:
        await cur.execute(
        """
        INSERT INTO lugares (ciudad,estado,latitud,longitud)
        VALUES
        (%s, %s, %s, %s)
        """,(ciudad,estado,latitud,longitud)
        )
    
    await conn.commit()
    conn.close()
 
 