import autocomplete


def getacc(s):
    print(s)

    preds = autocomplete.split_predict(s)
    print(preds)
    if(len(preds) == 0):
        # return s.trim().split(" ").splice(-1)
        return s.strip().split()[-1]
    return preds[0][0]


autocomplete.load()

print(getacc('I am a huge agi'))
