import json


class Dialog(object):
    ACTOR_SYSTEM = 0
    ACTOR_USER = 1

    def __init__(self, id):
        self.messages = []
        self.states = []  # Each message has one state.
        self.actors = []  # Each message has an actor id associated.
        self.id = id

    def add_message(self, text, state, actor):
        self.messages.append(text)
        self.states.append(state)
        self.actors.append(actor)

    def serialize(self):
        return json.dumps(
            {
                'messages': self.messages,
                'states': self.states,
                'actors': self.actors,
                'id': self.id
            }, indent=4)

    @classmethod
    def deserialize(cls, input_data):
        data = json.loads(input_data)

        obj = Dialog(data['id'])
        obj.messages = data['messages']
        obj.states = data['states']
        obj.actors= data['actors']

        return obj
