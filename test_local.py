import asyncio
import os
from models import DataCleanAction
from client import get_client

async def test_env():
    # Test without docker first
    client = await get_client(None)
    
    try:
        print("Resetting...")
        result = await client.reset(task="easy_clean")
        print("Reset result:", result)
        
        print("Sending action...")
        action = DataCleanAction(action_type="fill_na", column_name="age", value="0")
        result = await client.step(action)
        print("Step result:", result)
        
        print("Submitting...")
        action = DataCleanAction(action_type="submit")
        result = await client.step(action)
        print("Submit result:", result)
        print("Success!")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_env())