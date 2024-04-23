import sys

from config import DEFAULT_TABLE
from logs import LOGGER
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper


def do_delete(table_name: str, vector_id: str, milvus_client: MilvusHelper, mysql_cli: MySQLHelper):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        status = milvus_client.delete(table_name, vector_id)
        mysql_cli.delete_single_record(table_name, vector_id)
        return status
    except Exception as e:
        LOGGER.error(f"Error with deleting vector: {e}")
        sys.exit(1)