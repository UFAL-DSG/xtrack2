def parse(ln):
    return ln[59:69].strip()


def parse_param(ln):
    param, val = ln[43:].strip().split(': ')
    return param, val


def parse_n_params(ln):
    return int(ln.split('has')[1].split('parameters')[0].strip())


def main(log_file, print_header):
    row = {
        'best_epoch': -1,
        'best_acc': 0.0
    }
    with open(log_file) as f_in:
        f_lines = iter(f_in)
        acc = 0.0

        while True:
            try:
                ln = next(f_lines)
            except StopIteration:
                break
            if 'This model has' in ln:
                row['a_model_parameters'] = parse_n_params(ln)
            if 'Epoch' in ln:
                epoch = int(parse(ln))
            elif 'Valid tracking acc' in ln:
                acc = float(parse(ln))
            elif 'Effective args' in ln:
                while not 'Experiment path' in ln:
                    ln = next(f_lines)
                    param, val = parse_param(ln)
                    row[param] = val
            elif 'Example' in ln:
                if acc > row['best_acc']:
                    row['best_acc'] = acc
                    row['best_epoch'] = epoch



    keys = sorted(row.keys())
    if print_header:
        print ';'.join(keys)

    print ';'.join(str(row[key]) for key in keys)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    parser.add_argument('--print_header', default=False, action='store_true')

    args = parser.parse_args()

    main(**vars(args))
