import ollama

# Mapping of MiniGrid actions to text
ACTION_MEANING = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up object",
    4: "drop object",
    5: "toggle door",
    6: "done"
}


def llm_reward(state, action, next_state):
    """
    Returns:
        +1 if LLM thinks action is helpful
        -1 if LLM thinks action is harmful
    """

    action_text = ACTION_MEANING.get(action, "unknown action")

    prompt = f"""
An agent is exploring a grid-world environment.

The goal is usually to:
- explore the map
- find a key
- open a door
- reach a goal tile

The agent performed this action:

{action_text}

Label the action:

GOOD → helps exploration or movement
BAD → wastes time (e.g., repeated turning)

Respond ONLY with:
GOOD
BAD
"""

    try:
        response = ollama.chat(
            model="phi3",   # change to llama3 if installed
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0
            }
        )

        result = response["message"]["content"].strip().upper()

        # Extract first word only
        result = result.split()[0]

        print("LLM decision:", result)

        if result == "GOOD":
            return 1

        else:
            return -1

    except Exception as e:
        print("LLM error:", e)
        return 0