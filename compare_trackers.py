from collections import defaultdict
import json

from xtrack_data2 import XTrackData2


def main():
    data_file = '/xdisk/data/dstc/xtrack/e2_food/valid.json'
    data = XTrackData2.load(data_file)

    data_map = {}
    for seq in data.sequences:
        labeling = set()
        for label in seq['labels']:
            pred_label = {}

            labeling.add(label['time'])

        texts = []
        scores = []
        curr = []
        for i, (d, score) in enumerate(zip(seq['data'], seq['data_score'])):
            curr.append(data.vocab_rev[d])
            if i in labeling:
                texts.append((score, " ".join(curr)))
                curr = []
        assert len(curr) == 0
        print seq['id']
        data_map[seq['id']] = texts

    tracker_files = [
        '/xdisk/data/dstc/dstc2/scripts/baseline_output.json',
        '/xdisk/data/dstc/dstc2/scripts/xtrack_output.json'
    ]

    tracker_data = []
    for t_file in tracker_files:
        tracker_data.append(json.load(open(t_file)))

    sess_map = defaultdict(dict)
    sess_ids = set()
    for s1, s2 in zip(tracker_data[0]['sessions'], tracker_data[1][
        'sessions']):
        sess_map[0][s1['session-id']] = s1
        sess_map[1][s2['session-id']] = s2

        sess_ids.add(s1['session-id'])
        sess_ids.add(s2['session-id'])

    for id in sess_ids:
        texts =  data_map[id]
        s1, s2 = sess_map[0][id], sess_map[1][id]
        for (score, text), turn1, turn2 in zip(texts, s1['turns'], s2['turns']):
            print "%.2f" % score, text
            print turn1['goal-labels'].get('food', {})
            print turn2['goal-labels'].get('food', {})
            print
        print '--------'




if __name__ == '__main__':
    main()