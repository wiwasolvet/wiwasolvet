# Data stored and retrieved in database

# Template for sqlalchemy engine to connect to YOUR database, requires a user/password, server/port at minimum
# engine = create_engine('mysql+mysqlconnector://[user]:[pass]@[host]:[port]/[schema]', echo=False)
# data.to_sql(name='sample_table2', con=engine, if_exists = 'append', index=False)


def storedatabase(dataFull):
    import pymysql
    from sqlalchemy import create_engine

    engine = create_engine('ADD ENGINE DETAILS HERE',
                           echo=False)
    dataFull.to_sql(name='wp_trials', con=engine, if_exists='append', index=False)


def storedatabaseNEW(dataFull,vartablen):
    import pymysql
    from sqlalchemy import create_engine, MetaData

    metadata = MetaData()
    vartable = vartablen
    get_table_object(vartable,metadata)
    engine = create_engine('ADD ENGINE DETAILS HERE',
                           echo=True)
    metadata.create_all(engine)
    dataFull.to_sql(name=vartable, con=engine, if_exists = 'append', index=False)


def get_table_object(vartable, metadata):
    from sqlalchemy import Table, Column, String, Float  # , Integer, MetaData, ForeignKey

    table_name = vartable
    table_object = Table(table_name, metadata,
        # Switch to Datetime to avoid more type conversions if pymyql/sqlalchemy can handle the datetime object?
        Column('time', String(19), nullable=True),
        Column('winds', Float, nullable=True),
        Column('windd', Float, nullable=True),
        Column('preint', Float, nullable=True),
        Column('prepro', Float, nullable=True),
        Column('pretyp', String(256), nullable=True),
        Column('temp', Float, nullable=True),
        Column('humid', Float, nullable=True),
        Column('cloud', Float, nullable=True),
        Column('press', Float, nullable=True),
        Column('correctedwindspeed', Float, nullable=True)
        #mysql_charset='utf8',
    )
    return table_object

def retrievedatabase():
    import pymysql
    from sqlalchemy import create_engine
    import pandas as pd

    engine = create_engine('ADD ENGINE DETAILS HERE',
                           echo=False)
    con = engine.connect()
    rs = con.execute("SELECT * FROM `site_002_46.054_-60.329_1986a`", index_col=0)
    dataFull = pd.DataFrame(rs.fetchall())
    dataFull.columns = rs.keys()
    con.close()
    return dataFull
