import ollama


def llm_reward(prev_state, action, new_state):

    prompt = f"""
You are evaluating progress in a reinforcement learning task.

Goal:
1. Pick up the key
2. Open the door
3. Reach the goal

Return ONLY a number between 0 and 1 representing progress.

Previous state: {prev_state}
Action: {action}
Current state: {new_state}
"""

    try:

        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        text = response["message"]["content"].strip()

        value = float(text)

        value = max(0, min(1, value))

    except:
        value = 0.0

    return value