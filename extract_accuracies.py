import json


with open('/xdisk/tmp/dstc2_results/team4/entry0.test.json') as f_in:
    data = json.load(f_in)

for session in data['sessions']:
    id = session['session-id']
    for turn in session['turns']:
        v, p = sorted(turn['goal-labels']['food'].items(), key=lambda x: x[
            1])[-1]
        p_mass = sum(p for _, p in turn['goal-labels']['food'].items())
        p_null = 1.0 - p_mass
        if p > p_null:
            print v
        else:
            print '_null_'