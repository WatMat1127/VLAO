

def txtfile2list(path, keyword_list=None):
    with open(path, mode='r') as f:
        lines = f.readlines()
        l = [line.strip() for line in lines]

        if keyword_list == None:
            return l
        else:
            dat_idx = {}
            for ikey in keyword_list:
                idx = [i for i, line in enumerate(l) if '{}'.format(ikey) in line]
                dat_idx[ikey] = idx
            return l, dat_idx

def str2textfile(path, str_write):
    with open(path, mode='w') as g:
        g.write(str_write)

def list2textfile(path, list_write):
    tmp_str = '\n'.join(list_write)
    str2textfile(path, tmp_str)