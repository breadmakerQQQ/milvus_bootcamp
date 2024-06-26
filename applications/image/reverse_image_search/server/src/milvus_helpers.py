import sys
from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from logs import LOGGER


class MilvusHelper:
    """
    Milvus Helper
    """
    def __init__(self):
        try:
            self.collection = None
            self.collectionMap = dict()

            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            LOGGER.debug(f"Successfully connect to Milvus with IP:{MILVUS_HOST} and PORT:{MILVUS_PORT}")
        except Exception as e:
            LOGGER.error(f"Failed to connect Milvus: {e}")
            sys.exit(1)

    # Deprecate: this is thread-unsafe
    def set_collection(self, collection_name):
        try:
            self.collection = Collection(name=collection_name)
        except Exception as e:
            LOGGER.error(f"Failed to set collection in Milvus: {e}")
            sys.exit(1)

    def get_collection(self, collection_name):
        try:
            collection = self.collectionMap.get(collection_name)
            if collection is None:
                collection = Collection(name=collection_name)
                self.collectionMap[collection_name] = collection
            return collection
        except Exception as e:
            LOGGER.error(f"Failed to get collection in Milvus: {e}")
            sys.exit(1)

    def has_collection(self, collection_name):
        # Return if Milvus has the collection
        try:
            return utility.has_collection(collection_name)
        except Exception as e:
            LOGGER.error(f"Failed to get collection info to Milvus: {e}")
            sys.exit(1)

    def create_collection(self, collection_name):
        # Create milvus collection if not exists
        try:
            field1 = FieldSchema(name="id", dtype=DataType.INT64, description="int64", is_primary=True, auto_id=True)
            field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="float vector",
                                    dim=VECTOR_DIMENSION, is_primary=False)
            schema = CollectionSchema(fields=[field1, field2], description="collection description")
            self.collection = Collection(name=collection_name, schema=schema)
            LOGGER.debug(f"Create Milvus collection: {collection_name}")
            return "OK"
        except Exception as e:
            LOGGER.error(f"Failed create collection in Milvus: {e}")
            sys.exit(1)

    def insert(self, collection_name, vectors):
        # Batch insert vectors to milvus collection
        try:
            # self.set_collection(collection_name)
            data = [vectors]
            mr = self.get_collection(collection_name).insert(data)
            ids = mr.primary_keys
            LOGGER.debug(
                    f"Insert vectors to Milvus in collection: {collection_name} with {len(vectors)} rows")
            return ids
        except Exception as e:
            LOGGER.error(f"Failed to insert data to Milvus: {e}")
            sys.exit(1)

    def delete(self, collection_name, vector_id):
        # Single delete vector in milvus collection
        try:
            collection = self.get_collection(collection_name)
            collection.delete(f"id == {vector_id}")
            return "OK"
        except Exception as e:
            LOGGER.error(f"Failed to delete vector in Milvus: {e}")
            sys.exit(1)

    def get(self, collection_name, vector_id):
        # Query single record in milvus collection
        try:
            collection = self.get_collection(collection_name)
            results = collection.query(expr=f"id == {vector_id}", output_fields=["id", "embedding"])
            return results[0]["embedding"]
        except Exception as e:
            LOGGER.error(f"Failed to get vector in Milvus: {e}")
            sys.exit(1)

    def create_index(self, collection_name):
        # Create IVF_FLAT index on milvus collection
        try:
            collection = self.get_collection(collection_name)
            default_index = {"metric_type": METRIC_TYPE, "index_type": "IVF_FLAT", "params": {"nlist": 2048}}
            status = collection.create_index(field_name="embedding", index_params=default_index)
            if not status.code:
                LOGGER.debug(
                    f"Successfully create index in collection:{collection_name} with param:{default_index}")
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error(f"Failed to create index: {e}")
            sys.exit(1)

    def delete_collection(self, collection_name):
        # Delete Milvus collection
        try:
            collection = self.get_collection(collection_name)
            collection.drop()
            LOGGER.debug("Successfully drop collection!")
            return "ok"
        except Exception as e:
            LOGGER.error(f"Failed to drop collection: {e}")
            sys.exit(1)

    def search_vectors(self, collection_name, vectors, top_k):
        # Search vector in milvus collection
        try:
            # self.set_collection(collection_name)
            collection = self.get_collection(collection_name)
            collection.load()
            search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": 16}}
            res = collection.search(vectors, anns_field="embedding", param=search_params, limit=top_k)
            LOGGER.debug(f"Successfully search in collection: {res}")
            return res
        except Exception as e:
            LOGGER.error(f"Failed to search vectors in Milvus: {e}")
            sys.exit(1)

    def count(self, collection_name):
        # Get the number of milvus collection
        try:
            collection = self.get_collection(collection_name)
            collection.flush()
            num = collection.num_entities
            LOGGER.debug(f"Successfully get the num:{num} of the collection:{collection_name}")
            return num
        except Exception as e:
            LOGGER.error(f"Failed to count vectors in Milvus: {e}")
            sys.exit(1)
