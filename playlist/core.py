from model import PlaylistGeneration

if __name__ == '__main__':
    PlaylistGeneration("/home/burak/Desktop/dataset/yes_small/train.txt") \
        .process() \
        .report()
