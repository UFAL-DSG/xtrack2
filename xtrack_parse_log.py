from collections import defaultdict
import re


def parse(ln, part=1):
    return ln.rsplit(' ', 2)[part]


def parse_col(ln):
    return ln[59:69].strip()


def remove_multiple_spaces(ln):
    return re.sub(' +', ' ', ln)


def parse_valid_type(ln):
    ln = remove_multiple_spaces(ln)
    return ln.rsplit(' ', 3)[1].replace(':', '')


def parse_param(ln):
    param, val = ln[43:].strip().split(': ')
    return param, val


def parse_n_params(ln):
    return int(ln.split('has')[1].split('parameters')[0].strip())


def add_best_stats_to_row(best_acc, best_epoch, best_param_file, row):
    for stat in best_acc:
        row['a_best_acc_%s' % stat] = best_acc[stat]
        row['b_best_epoch_%s' % stat] = best_epoch[stat]
        row['p_%s' % stat] = best_param_file[stat]


def main(log_file, print_header, sep_chr):
    best_acc = defaultdict(float)
    best_epoch = defaultdict(lambda: -1)
    best_param_file = {}
    epoch = -1

    row = {}
    with open(log_file) as f_in:
        f_lines = iter(f_in)
        acc = defaultdict(float)
        param_file = None
        while True:
            try:
                ln = next(f_lines)
                ln = ln.strip()
            except StopIteration:
                break
            if 'This model has' in ln:
                row['c_nparams'] = parse_n_params(ln)
            if 'Epoch' in ln:
                epoch = int(parse_col(ln))
            elif 'Saving parameters' in ln:
                param_file = parse(ln, part=-1)
            elif 'Valid tracking acc' in ln:
                acc['goals'] = float(parse_col(ln))
            elif 'Valid acc' in ln:
                valid_type = parse_valid_type(ln)
                acc[valid_type] = float(parse(ln))
            elif 'Effective args' in ln:
                while not 'Experiment path' in ln:
                    ln = next(f_lines)
                    param, val = parse_param(ln)
                    row[param] = val
            elif 'Example' in ln:
                for acc_type in acc:
                    if acc[acc_type] > best_acc[acc_type]:
                        best_acc[acc_type] = acc[acc_type]
                        best_epoch[acc_type] = epoch
                        best_param_file[acc_type] = param_file

    add_best_stats_to_row(best_acc, best_epoch, best_param_file, row)


    keys = sorted(row.keys())
    if print_header:
        print sep_chr.join(keys)

    print sep_chr.join(str(row[key]) for key in keys)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    parser.add_argument('--print_header', default=False, action='store_true')
    parser.add_argument('--sep_chr', default='\t')

    args = parser.parse_args()

    main(**vars(args))
