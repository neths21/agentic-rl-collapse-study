import ollama

def generate_plan():

    prompt = """
You are planning steps for solving a grid-world puzzle.

The puzzle usually requires:
- exploring the map
- finding a key
- opening a door
- reaching a goal tile

Return a short ordered list of subtasks.
"""

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature":0}
    )

    plan = response["message"]["content"]

    print("Generated Plan:")
    print(plan)

    return plan