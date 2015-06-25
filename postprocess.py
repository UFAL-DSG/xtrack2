import json

N_AHEAD = 3

def main(track_file, output_track_file):
    with open(track_file) as f_in:
        data = json.load(f_in)

    smooth(data)

    with open(output_track_file, 'w') as f_out:
        json.dump(data, f_out, indent=4)


def smooth(data):
    for session in data['sessions']:
        turns = session['turns']
        for i, curr_turn in enumerate(turns):
            curr_goals = curr_turn['goal-labels']
            prev_goals = None
            if i > 0:
                prev_goals = turns[i - 1]['goal-labels']

            for goal in curr_goals:
                if i > 0:
                    if prev_goals and curr_goals.get(goal) != prev_goals.get(goal):
                        if prev_goals.get(goal) == {'dontcare': 1.0}:
                            continue
                        if prev_goals.get(goal) == {}:
                            continue
                        keep_y = None
                        for y, next_turn in enumerate(turns[i + 1: i + 1 + N_AHEAD]):
                            next_goals = next_turn['goal-labels']
                            if next_goals.get(goal) == prev_goals.get(goal):
                                keep_y = y
                                break

                        if keep_y:
                            for next_turn in turns[i:i + keep_y + 1]:
                                next_goals = next_turn['goal-labels']
                                next_goals[goal] = prev_goals[goal]






if __name__ == '__main__':
    import utils
    utils.pdb_on_error()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('track_file')
    parser.add_argument('output_track_file')

    args = parser.parse_args()

    main(**vars(args))