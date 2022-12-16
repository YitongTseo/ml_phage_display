
def find_3mer(seq, tmer):
    for i in range(len(seq) - 2):
        if seq[i] == tmer[0] and seq[i + 1] == tmer[1] and seq[i + 2] == tmer[2]:
            return True
    return False

def cnt_c(seq):
    cnt = 0
    for i in range(len(seq)):
        if seq[i] == "C":
            cnt += 1
    return cnt
