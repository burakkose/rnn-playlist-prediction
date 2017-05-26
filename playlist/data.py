def _read(path):
    with open(path) as f:
        return list(map(lambda x: x.split(" ")[:-1], f.readlines()[2:]))


def generate_data(path):
    filtered_data = list(filter(lambda p: len(p) > 1, _read(path)))
    x, y, vocabulary = [], [], set()

    def _generate(line):
        line = list(map(lambda x: int(x) + 1, line))
        vocabulary.update(set(line))
        for i in range(0, len(line) - 1):
            x.append(list(map(lambda p: int(p) + 1, line[0:i + 1])))
            y.append(int(line[i + 1]) + 1)
        return len(line)

    return x, y, vocabulary, max(list(map(_generate, filtered_data)))
