import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database():
    # Database configuration
    dbname = "semusi" 
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")

    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        dbname="postgres",
        user=user,
        password=password,
        host=host,
        port=port
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Create database if it doesn't exist
    try:
        cur.execute(f"CREATE DATABASE {dbname}")
        print(f"Database '{dbname}' created successfully")
    except psycopg2.errors.DuplicateDatabase:
        print(f"Database '{dbname}' already exists")
    finally:
        cur.close()
        conn.close()

    # Connect to the new database and create tables
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    cur = conn.cursor()

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(current_dir, 'migrations', 'schema.sql')
        
        print(f"Schema path: {schema_path}") 
        print(f"current dir: {current_dir}")
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found at: {schema_path}")
            
        with open(schema_path, 'r') as schema_file:
            cur.execute(schema_file.read())
        conn.commit()
        print("Schema created successfully")
    except FileNotFoundError as e:
        print(f"Schema file error: {e}")
        conn.rollback()
    except Exception as e:
        print(f"Error creating schema: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    create_database()
