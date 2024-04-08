import redis
from config import config


class RedisClient:
    def __init__(self):
        self.client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=config.REDIS_DECODE_RESPONSES,
        )

    def connect(self):
        return self.client

    def disconnect(self):
        self.client.close()

    def set(self, key, value):
        return self.client.set(key, value)

    def get(self, key):
        return self.client.get(key)

    def delete(self, key):
        return self.client.delete(key)

    def pipeline(self):
        return self.client.pipeline()


redis_client = RedisClient()
