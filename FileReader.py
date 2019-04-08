class DataSetFileReader():
    @staticmethod
    def read_dataset_file(filepath):
        with open(filepath, 'r', encoding='utf8') as dataset:
            data = []
            target = []
            for line in dataset:
                label = line.split(maxsplit=1)[0]
                msg = line.split(maxsplit=1)[1]

                target.append(label)
                data.append(msg)

            return data, target
