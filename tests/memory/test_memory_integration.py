import pytest
import json
from datetime import datetime, timedelta
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.simple import SimpleVectorStore
from gtd_assistant.agent.memory import SortedDocStore, DurableSemanticMemory, pack_entry, unpack_entry
import logging
from contextlib import ExitStack

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()
# print("Debugger attached.")


@pytest.fixture
def sorted_doc_store(redis_client):
    return SortedDocStore(redis_client)


def test_durable_semantic_memory_get_all(redis_client):
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
        ChatMessage(role=MessageRole.USER, content="Hello", additional_kwargs={"timestamp": now - 3600}),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!", additional_kwargs={"timestamp": now - 3500}),
        ChatMessage(role=MessageRole.USER, content="How are you?", additional_kwargs={"timestamp": now - 3400}),
        ChatMessage(role=MessageRole.ASSISTANT, content="I'm doing well, thanks!", additional_kwargs={"timestamp": now - 3300}),
        ChatMessage(role=MessageRole.USER, content="What's the weather like?", additional_kwargs={"timestamp": now - 3200}),
        ChatMessage(role=MessageRole.ASSISTANT, content="It's sunny and warm.", additional_kwargs={"timestamp": now - 3100}),
        ChatMessage(role=MessageRole.USER, content="Great, thanks!", additional_kwargs={"timestamp": now - 3000}),
        ChatMessage(role=MessageRole.ASSISTANT, content="You're welcome!", additional_kwargs={"timestamp": now - 2900}),
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


def test_durable_semantic_memory_semantic_get(redis_client):
    now = datetime.now().timestamp()
    messages = [
        ChatMessage(role=MessageRole.USER, content="What's interesting about octopi?", additional_kwargs={"timestamp": now - 3600}),
        ChatMessage(role=MessageRole.ASSISTANT, content="Octopi are fascinating creatures! They have three hearts, blue blood, and can change color to blend with their surroundings. They're also known for their intelligence and problem-solving abilities.", additional_kwargs={"timestamp": now - 3500}),
        ChatMessage(role=MessageRole.USER, content="How many arms do octopi have?", additional_kwargs={"timestamp": now - 3400}),
        ChatMessage(role=MessageRole.ASSISTANT, content="Octopi actually have eight arms, not tentacles. Each arm has suction cups and can operate independently, allowing them to multitask effectively!", additional_kwargs={"timestamp": now - 3300}),
        ChatMessage(role=MessageRole.USER, content="I'm planning a trip to London. What are some must-see attractions?", additional_kwargs={"timestamp": now - 3200}),
        ChatMessage(role=MessageRole.ASSISTANT, content="London has many iconic attractions! Some must-see places include Big Ben, the Tower of London, Buckingham Palace, the British Museum, and the London Eye. The city also has beautiful parks like Hyde Park and great shopping areas like Oxford Street.", additional_kwargs={"timestamp": now - 3100}),
        ChatMessage(role=MessageRole.USER, content="How many days should I plan for my London trip?", additional_kwargs={"timestamp": now - 3000}),
        ChatMessage(role=MessageRole.ASSISTANT, content="For a comprehensive London experience, I'd recommend planning for at least 5-7 days. This will give you enough time to see the major attractions, explore different neighborhoods, and maybe take a day trip to nearby places like Windsor Castle or Stonehenge.", additional_kwargs={"timestamp": now - 2900}),
        ChatMessage(role=MessageRole.USER, content="What's the best way to get around in London?", additional_kwargs={"timestamp": now - 2800}),
        ChatMessage(role=MessageRole.ASSISTANT, content="The best way to get around London is by using the public transportation system. The Underground (also known as the Tube) is efficient and covers most of the city. Buses are also a great option and offer scenic routes. For convenience, get an Oyster card or use contactless payment for all public transport. Walking is also great for shorter distances and allows you to discover hidden gems in the city.", additional_kwargs={"timestamp": now - 2700}),
        ChatMessage(role=MessageRole.USER, content="Can you recommend some good areas to stay in London?", additional_kwargs={"timestamp": now - 2600}),
        ChatMessage(role=MessageRole.ASSISTANT, content="""Certainly! Some popular areas to stay in London include:
1. Covent Garden: Central location with great shopping and dining options.
2. South Kensington: Near museums and Hyde Park, upscale area.
3. Shoreditch: Trendy area with vibrant nightlife and street art.
4. Westminster: Close to major attractions like Big Ben and Westminster Abbey.
5. Camden: Known for its markets and alternative vibe.
Choose based on your interests and budget. Central areas are more expensive but offer convenience, while staying a bit further out can be more affordable.""", additional_kwargs={"timestamp": now - 2500}),
    ]

    memory = DurableSemanticMemory(
        redis_client=redis_client,
        embed_model=None,
        max_memory_age=timedelta(hours=1),
        max_recent_memories=3,  # this should ensure that the octopus messages are not included in get_all()
        batch_by_user_message=True,
    )
    logger.debug(f"embed model: {memory.vector_index._embed_model}")
    for message in messages:
        memory.put(message)

    # Check that octopi messages are not included in get_all(), since they're too old
    all_results = memory.get_all()
    content = " ".join([memory.content for memory in all_results])
    assert "octopi" not in content


    semantic_memory_results = memory.get("What animals have we talked about?")
    content = " ".join([memory.content for memory in semantic_memory_results])
    assert "octopi" in content


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


if __name__ == "__main__":
    pytest.main()
