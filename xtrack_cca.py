import csv
from sklearn.cross_decomposition import CCA


def atfield_parse(ftext):
    f, fvals  = ftext[1:].split('=')

    return f, fvals.split(',')


def main(results_csv, xf):
    x = []
    y = []
    with open(results_csv) as f_in:
        for i, ln in enumerate(csv.reader(f_in, delimiter=';')):
            if i == 0:
                header = {}
                for j, key in enumerate(ln):
                    header[key] = j

            else:
                y.append([float(ln[header['best_acc']])])
                row = []
                for f in xf:
                    if f.startswith('@'):
                        f, fvals  = atfield_parse(f)
                        val = ln[header[f]]
                        for fval in fvals:
                            if fval == val:
                                row.append(1.0)
                            else:
                                row.append(0.0)
                    else:
                        row.append(float(ln[header[f]]))
                x.append(row)

                print row, y[-1]


    cca = CCA(n_components=1)
    cca.fit(x, y)

    print cca.x_rotations_
    print cca.x_mean_
    print cca.x_std_
    print cca.y_rotations_

    outstr = '%40s=%6.2f'
    coef_ndx = 0
    for f in xf:
        if f.startswith('@'):
            fname, fvals = atfield_parse(f)
            for fval in fvals:
                ff = "%s__%s" % (fname, fval)
                print outstr % (ff, cca.x_rotations_[coef_ndx])
                coef_ndx += 1
        else:
            print outstr % (f, cca.x_rotations_[coef_ndx])
            coef_ndx += 1



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('results_csv')
    parser.add_argument('--xf', action='append')

    args = parser.parse_args()

    main(**vars(args))