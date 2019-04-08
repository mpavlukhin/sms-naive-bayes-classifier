from FileReader import DataSetFileReader
from NaiveBayes import NaiveBayes

def main():
    data, target = DataSetFileReader.read_dataset_file(
        'data/SMSSpamCollection'
    )
    classifier = NaiveBayes.fit(data, target)

    result = NaiveBayes.predict(
        classifier,
        'SMS. ac Sptv: The New Jersey Devils and the Detroit Red Wings play '
        'Ice Hockey. Correct or Incorrect? End? Reply END SPTV'
    )
    print(result)

    result = NaiveBayes.predict(classifier, 'Fair enough, anything going on?')
    print(result)


if __name__ == '__main__':
    main()