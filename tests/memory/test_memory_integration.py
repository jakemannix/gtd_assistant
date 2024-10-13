import os
import pytest
import docker
import redis
import json
from datetime import datetime, timedelta
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.simple import SimpleVectorStore
from gtd_assistant.agent.memory import SortedDocStore, DurableSemanticMemory, pack_entry, unpack_entry
import logging
from contextlib import ExitStack
import time
import socket

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()
# print("Debugger attached.")

def get_docker_client():
    # Try to get the Docker host from an environment variable
    docker_host = os.environ.get('DOCKER_HOST')
    
    if docker_host:
        logger.debug(f"Using DOCKER_HOST: {docker_host}")
        return docker.DockerClient(base_url=docker_host)
    
    # If no DOCKER_HOST is set, try common socket locations
    common_socket_paths = [
        'unix:///var/run/docker.sock',  # Standard Linux path
        f'unix://{os.path.expanduser("~")}/Library/Containers/com.docker.docker/Data/docker-cli.sock',  # macOS Docker Desktop (older versions)
        f'unix://{os.path.expanduser("~")}/.docker/run/docker.sock',  # macOS Docker Desktop (newer versions)
        os.path.join('npipe:', '', '', '.', 'pipe', 'docker_engine')  # Windows Docker Desktop
    ]
    
    for socket_path in common_socket_paths:
        logger.debug(f"Trying socket path: {socket_path}")
        try:
            client = docker.DockerClient(base_url=socket_path)
            client.ping()  # Test the connection
            logger.debug(f"Successfully connected using: {socket_path}")
            return client
        except Exception as e:
            logger.debug(f"Failed to connect using {socket_path}: {str(e)}")
    
    logger.debug("Falling back to docker.from_env()")
    return docker.from_env()

def is_redis_available(host='localhost', port=6379):
    try:
        client = redis.Redis(host=host, port=port, socket_connect_timeout=1)
        client.ping()
        logger.debug("Existing Redis instance is reachable")
        return True
    except (redis.exceptions.ConnectionError, socket.timeout):
        logger.debug("No existing Redis instance found")
        return False

@pytest.fixture(scope="session")
def docker_client():
    return get_docker_client()

@pytest.fixture(scope="module")
def redis_container(docker_client):
    if is_redis_available():
        logger.debug("Using existing Redis instance")
        yield None  # No container to yield if we're using an existing instance
        return

    logger.debug("Attempting to start Redis container")
    try:
        container = docker_client.containers.run(
            "redis:latest",
            ports={'6379/tcp': 6379},
            detach=True
        )
        logger.debug(f"Container started with ID: {container.id}")
    except docker.errors.APIError as e:
        logger.error(f"Failed to start Redis container: {e}")
        raise

    # Wait for the container to be ready
    for i in range(30):  # Try for 30 seconds
        if is_redis_available():
            logger.debug("Redis in container is now available")
            break
        time.sleep(1)
    else:
        logger.error("Redis in container did not become available in time")
        raise TimeoutError("Redis in container did not become available in time")

    yield container
    
    if container:
        logger.debug("Stopping container")
        container.stop()
        logger.debug("Removing container")
        container.remove()

@pytest.fixture(scope="module")
def redis_client(redis_container):
    logger.debug("Attempting to connect to Redis")
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        redis_client.ping()  # Test the connection
        logger.debug("Successfully connected to Redis")        
        # Flush the database to ensure it's empty
        logger.debug("Flushing Redis database to ensure it's empty")
        redis_client.flushdb()
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise
    yield redis_client
    logger.debug("Closing Redis connection")
    redis_client.close()

@pytest.fixture
def sorted_doc_store(redis_client):
    return SortedDocStore(redis_client)

def test_add_and_retrieve_entries(sorted_doc_store):
    # Create test nodes
    now = datetime.now().timestamp()
    nodes = [
        TextNode(text=f"Test message {i}", metadata={"timestamp": now - i, "sub_dicts": []})
        for i in range(5)
    ]

    # Add nodes to the store
    for node in nodes:
        sorted_doc_store.add_entry(node)

    # Retrieve recent entries
    retrieved_nodes = sorted_doc_store.get_recent_entries(now + 1, 10)

    # Check if retrieved nodes match the original nodes
    assert len(retrieved_nodes) == 5
    for original, retrieved in zip(nodes, retrieved_nodes):
        assert original.text == retrieved.text
        assert original.metadata["timestamp"] == retrieved.metadata["timestamp"]

def test_durable_semantic_memory(redis_client):
    max_recent_memories = 3
    memory = DurableSemanticMemory(
        redis_client=redis_client,
        embed_model=None,
        max_recent_memories=max_recent_memories,
        max_memory_age=timedelta(hours=1)
    )

    # Add messages
    now = datetime.now().timestamp()
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello", metadata={"timestamp": now - 3600}),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!", metadata={"timestamp": now - 3500}),
        ChatMessage(role=MessageRole.USER, content="How are you?", metadata={"timestamp": now - 3400}),
        ChatMessage(role=MessageRole.ASSISTANT, content="I'm doing well, thanks!", metadata={"timestamp": now - 3300}),
        ChatMessage(role=MessageRole.USER, content="What's the weather like?", metadata={"timestamp": now - 3200}),
        ChatMessage(role=MessageRole.ASSISTANT, content="It's sunny and warm.", metadata={"timestamp": now - 3100}),
        ChatMessage(role=MessageRole.USER, content="Great, thanks!", metadata={"timestamp": now - 3000}),
        ChatMessage(role=MessageRole.ASSISTANT, content="You're welcome!", metadata={"timestamp": now - 2900}),
    ]

    for msg in messages:
        memory.put(msg)

    retrieved_messages = memory.get_all()
    pretty_messages = json.dumps([msg.dict() for msg in retrieved_messages], indent=4)
    logger.debug(f"Retrieved messages: {pretty_messages}")

    # Get only USER messages from the retrieved messages
    retrieved_user_messages = [msg for msg in retrieved_messages if msg.role == MessageRole.USER]

    # Check the number of retrieved messages
    assert len(retrieved_user_messages) == max_recent_memories, f"Expected {max_recent_memories} user messages, got {len(retrieved_user_messages)}"

    # Define the expected messages (last 3 USER messages)
    expected_user_messages = [messages[i] for i in [2, 4, 6]]

    # Check content and order in one loop
    for i, (expected, retrieved) in enumerate(zip(expected_user_messages, retrieved_user_messages)):
        assert retrieved.content == expected.content, f"Mismatch at position {i}. Expected: {expected.content}, Got: {retrieved.content}"


def test_pack_and_unpack_entry():
    original_entry = {
        "text": "Test message",
        "metadata": {
            "timestamp": 1234567890.123456,
            "sub_dicts": [{"role": "user", "content": "Hello"}]
        }
    }

    packed = pack_entry(original_entry)
    unpacked = unpack_entry(packed)

    assert original_entry == unpacked

@pytest.fixture(scope="session", autouse=True)
def cleanup_redis_containers():
    docker_client = get_docker_client()
    # Stop and remove any existing Redis containers
    for container in docker_client.containers.list(all=True):
        if 'redis:latest' in container.image.tags:
            container.stop()
            container.remove()
    yield
    # Cleanup after all tests
    for container in docker_client.containers.list(all=True):
        if 'redis:latest' in container.image.tags:
            container.stop()
            container.remove()

@pytest.fixture(scope="session", autouse=True)
def cleanup_docker_images():
    yield
    docker_client = get_docker_client()
    docker_client.images.prune(filters={'dangling': True})

if __name__ == "__main__":
    pytest.main()
