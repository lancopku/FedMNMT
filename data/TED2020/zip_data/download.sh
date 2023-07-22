for pair in en-zh_cn en-th ar-en en-he en-fi en-et en-ru en-sl
do
    FILE=$pair.txt.zip
    URL=https://opus.nlpl.eu/download.php?f=TED2020/v1/moses/$FILE
    wget $URL -O $FILE
done
