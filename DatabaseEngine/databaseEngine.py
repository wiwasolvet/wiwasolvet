# Data stored and retrieved in database

import pymysql
import pandas as pd
from sqlalchemy import Table, Column, String, Float  # , Integer, MetaData, ForeignKey
from sqlalchemy import create_engine, MetaData

# Template for sqlalchemy engine to connect to YOUR database, requires a user/password, server/port at minimum
# engine = create_engine('mysql+mysqlconnector://[user]:[pass]@[host]:[port]/[schema]', echo=False)
# data.to_sql(name='sample_table2', con=engine, if_exists = 'append', index=False)


def store_database(data_full, env_variables):
    # import pymysql
    # from sqlalchemy import create_engine

    engine = create_engine(env_variables["ENGINE_DATABASE"],
                           echo=False)
    data_full.to_sql(name='wp_trials', con=engine, if_exists='append', index=False)


def store_database_new(data_full, vartablen, env_variables):
    # import pymysql
    # from sqlalchemy import create_engine, MetaData

    metadata = MetaData()
    vartable = vartablen
    get_table_object(vartable, metadata)
    engine = create_engine(env_variables["ENGINE_DATABASE"],
                           echo=True)
    metadata.create_all(engine)
    data_full.to_sql(name=vartable, con=engine, if_exists='append', index=False)


def get_table_object(vartable, metadata):
    # from sqlalchemy import Table, Column, String, Float  # , Integer, MetaData, ForeignKey

    table_name = vartable
    table_object = \
        Table(table_name, metadata,
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
              # mysql_charset='utf8',
              )
    return table_object


def retrieve_database(env_variables, input_variables):
    # import pymysql
    # from sqlalchemy import create_engine
    # import pandas as pd

    engine = create_engine(env_variables["ENGINE_DATABASE"],
                           echo=False)
    con = engine.connect()
    rs = con.execute("SELECT * FROM " + input_variables["DB_TABLE_NAME_SITE"], index_col=0)
    data_full = pd.DataFrame(rs.fetchall())
    data_full.columns = rs.keys()
    con.close()
    return data_full
