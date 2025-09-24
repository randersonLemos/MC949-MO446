from classes.imagemisc import  ImageMisc

class Data:
    @classmethod
    def get_casa(cls, n: int):
        ROOT_DIR = '../SampleSet/MVS Data'

        if n == 0:
            folder = f"{ROOT_DIR}/scan6_max_all"
        else:
            folder = f"{ROOT_DIR}/scan6_{n}_1"

        paths = ImageMisc.get_paths(folder, '*max.png')
        return paths

    @classmethod
    def get_chaleira(cls, n: int):
        ROOT_DIR = '../SampleSet/Chaleira'

        if n == 0:
            folder = f"{ROOT_DIR}/cl_all"
        else:
            folder = f"{ROOT_DIR}/cl_{n}"

        paths = ImageMisc.get_paths(folder, '*')
        return paths

    @classmethod
    def get_banana(cls, n: int):
        ROOT_DIR = '../SampleSet/Banana'

        if n == 0:
            folder = f"{ROOT_DIR}/ba_all"
        else:
            folder = f"{ROOT_DIR}/ba_{n}"

        paths = ImageMisc.get_paths(folder, '*')
        return paths

    @classmethod
    def get_banana2(cls, n: int):
        ROOT_DIR = '../SampleSet/Banana2'

        if n == 0:
            folder = f"{ROOT_DIR}/ba_all"
        else:
            folder = f"{ROOT_DIR}/ba_{n}"

        paths = ImageMisc.get_paths(folder, '*')
        return paths