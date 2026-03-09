from collections import deque

class CollapseDetector:

    def __init__(self):
        self.last_actions = deque(maxlen=10)

    def update(self, action):

        self.last_actions.append(action)

        if len(self.last_actions) < 10:
            return False

        if len(set(self.last_actions)) == 1:
            return True

        return False