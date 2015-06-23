from collections import defaultdict

def main(input_file):
    scnt = defaultdict(lambda: defaultdict(int))
    sct = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    with open(input_file) as f_in:
        for ln in f_in:
            ln = ln.strip()
            if "lbl(" in ln:
                for x in ['asian oriental', 'modern european', 'north american']:
                    ln = ln.replace(x, x.replace(' ', '_'))

                slot, lbl, pred = ln.split()
                pred = pred.replace('pred', 'lbl')
                sct[slot][lbl][pred] += 1
                scnt[slot][lbl] += 1

    for slot in scnt:
        print '#' * 10, slot, '#' * 10
        cnt = scnt[slot]
        ct = sct[slot]
        res = []
        for k in cnt:
            res.append((k, (ct[k][k] * 1.0 / cnt[k]), cnt[k], ct[k][k], cnt[k] - ct[k][k], ct['lbl(_null_)'][k] - ct[k][k]))


        res_list = []
        for i in sorted(res, key=lambda x: x[2]):
            print "%20s %.2f cnt(%3d) cnt_good_pred(%3d) err(%3d) null_gain(%3d)" % i
            if i[2] < 30:
                res_list.append(i[0].replace('lbl(', '').replace(')', ''))

        for x in res_list:
            print "'%s'," % x,

        print
    # cnt_null = 0
    # for i in res:
    #     if i[-1] > 0:
    #         cnt_null += i[-1]
    #         print "'%s'," % i[0].replace('lbl(', '').replace(')', ''),
    # print cnt_null



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')

    args = parser.parse_args()

    main(**vars(args))