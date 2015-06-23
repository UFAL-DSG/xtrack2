import json

def main(f):
    data = json.loads(open(f).read())
    for sess in data['sessions']:
        for turn in sess['turns']:
            mval, mp = zip(*sorted(turn['goal-labels']['name'].items(), key=lambda x: x[1]))

            print (1 - sum(mp)) < 0.5


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('f')

    args = parser.parse_args()

    main(**vars(args))