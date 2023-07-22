for pair in de-en en-nl en-fr en-it en-es en-pl en-sl en-lt en-lv de-fr nl-pl fr-nl it-sl es-lv es-sl lt-sl de-lt it-lv lv-pl
do
    FILE=$pair.txt.zip
    URL=https://opus.nlpl.eu/download.php?f=Europarl/v8/moses/$FILE
    wget $URL -O $FILE
done