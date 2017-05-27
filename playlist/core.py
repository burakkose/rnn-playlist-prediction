from model import PlaylistGeneration

if __name__ == '__main__':
    PlaylistGeneration() \
        .process() \
        .evaluate()
