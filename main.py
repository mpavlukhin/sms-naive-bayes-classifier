from FileReader import DataSetFileReader
from NaiveBayes import NaiveBayes

def main():
    data, target = DataSetFileReader.read_dataset_file(
        'data/SMSSpamCollection'
    )

    classifier = NaiveBayes()
    classifier.fit(data, target)

    input_data = DataSetFileReader.read_input_data_file('data/inputdata')


    result = NaiveBayes.predict(classifier, input_data)

    for pred, msg in zip(result, input_data):
        print('{0} -> {1}'.format(pred.upper(), msg))
    
if __name__ == '__main__':
    main()