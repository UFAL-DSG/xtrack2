import json
from jinja2 import Environment, FileSystemLoader
import logging

from model import Model
from xtrack_data2 import XTrackData2


def add_common_context(context):
    for key, val in __builtins__.__dict__.iteritems():
        context[key] = val
    #context['enumerate'] = enumerate
    #context['zip'] = zip


def compare(label, track, slots, classes_rev):
    res = {}
    for slot in slots:
        lbl_val = label['slots'].get(slot)
        lbl_val = classes_rev[slot][lbl_val]
        track_val = track['debug'].get(slot)

        res[slot] = lbl_val == track_val

    return res


def map_to_words(what, vocab_rev):
    res = []
    for word_id in what:
        res.append(vocab_rev[word_id])

    return res


def concat_system_and_user(system, user):
    system = " ".join("@%s" % w for w in system.split())
    return system + " " + user


def visualize(track_file, dataset):
    logging.info('Loading data: %s' % dataset)
    data = XTrackData2.load(dataset)
    classes_rev = {}
    for slot in data.slots:
        classes_rev[slot] = {val: key
                             for key, val in data.classes[slot].iteritems()}
    for slot in data.slots:
        classes_rev[slot][0] = None

    logging.info('Loading tracking results: %s' % track_file)
    with open(track_file) as f_in:
        track = json.load(f_in)

    track_sessions = {}
    for session in track['sessions']:
        track_sessions[session['session-id']] = session['turns']

    sequences = []
    for i, seq in enumerate(data.sequences):
        seq_track = track_sessions[seq['id']]
        times = [lbl['time'] for lbl in seq['labels']]
        labels = {lbl['time']: lbl for lbl in seq['labels']}
        tracks = {lbl['time']: seq_track[i] for i, lbl in enumerate(seq['labels'])}
        true_inputs = {lbl['time']: concat_system_and_user(
                           seq['true_input'][i * 2],
                           seq['true_input'][i * 2 + 1])
                       for i, lbl in enumerate(seq['labels'])}

        results = {}
        for t in times:
            results[t] = compare(labels[t], tracks[t], data.slots, classes_rev)

        sequences.append({
            'id': seq['id'],
            'labels': labels,
            'tracks': tracks,
            'results': results,
            'true_inputs': true_inputs,
            'tokens': map_to_words(seq['data'], data.vocab_rev)
        })

    context = {
        'sequences': sequences,
        'classes': data.classes,
        'classes_rev': classes_rev,
        'slots': data.slots
    }

    return context


def visualize_html(track_file, dataset, dest_file):
    context = visualize(track_file, dataset)
    add_common_context(context)

    env = Environment(loader=FileSystemLoader('xtrack2_vis'))
    tpl = env.get_template('xtrack2_track_index.html')

    with open(dest_file, 'w') as f_out:
        f_out.write(tpl.render(**context))






if __name__ == '__main__':
    import argparse
    import utils
    utils.pdb_on_error()

    parser = argparse.ArgumentParser()
    parser.add_argument('--track_file')
    parser.add_argument('--dataset')
    parser.add_argument('--dest_file')

    args = parser.parse_args()

    visualize_html(**vars(args))
