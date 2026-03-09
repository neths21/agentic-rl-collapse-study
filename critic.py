import ollama

ACTION_MEANING = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up object",
    4: "drop object",
    5: "toggle door",
    6: "done"
}

def llm_reward(action):

    action_text = ACTION_MEANING.get(action, "unknown action")

    prompt = f"""
Evaluate this action in a grid-world navigation task.

Action: {action_text}

Respond ONLY with:

GOOD
BAD
"""

    try:
        response = ollama.chat(
            model="phi3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        result = response["message"]["content"].strip().upper()
        result = result.split()[0]

        if result == "GOOD":
            return 1
        else:
            return 0

    except:
        return 0