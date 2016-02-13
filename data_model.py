import json


class Dialog(object):
    ACTOR_SYSTEM = 0
    ACTOR_USER = 1

    def __init__(self, object_id, session_id):
        self.messages = []
        #self.wcn = []
        self.states = []  # Each message has one state.
        self.actors = []  # Each message has an actor id associated.
        self.slots_mentioned = []
        self.object_id = object_id
        self.session_id = session_id

    def add_message(self, text, wcn, state, actor, slots_mentioned):
        self.messages.append(text)
        self.states.append(state)
        self.actors.append(actor)
        #self._add_wcn(wcn)
        self.slots_mentioned.append(slots_mentioned)

    #def _add_wcn(self, wcn):
    #    new_wcn = []
    #    for hyp in wcn:
    #        new_wcn.append((hyp.hyps, hyp.scores))
    #
    #    self.wcn.append(new_wcn)

    def copy(self):
        return Dialog.deserialize(self.serialize())

    def serialize(self):
        return json.dumps(
            {
                'messages': self.messages,
                #'wcn': self.wcn,
                'states': self.states,
                'actors': self.actors,
                'object_id': self.object_id,
                'session_id': self.session_id,
                'slots_mentioned': self.slots_mentioned
            }, indent=4)

    @classmethod
    def deserialize(cls, input_data):
        data = json.loads(input_data)

        obj = Dialog(data['object_id'], data['session_id'])
        obj.messages = data['messages']
        obj.wcn = data['wcn']
        obj.states = data['states']
        obj.actors = data['actors']
        obj.slots_mentioned = data['slots_mentioned']

        return obj
