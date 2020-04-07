To run the ELA program, use the following command line:

- Run 30 times calculating statistics (without generating graph with global explanation):

python code.py trainingSet testSet Numfeatures GPdepth generateGraph > outputFile

Example: python code.py Dataset/keijzer-1-train-0.csv Dataset/keijzer-1-test-0.csv 1 6 0 > output_keijzer-1_30

or

-To run only once generating a graph of stacked areas with global explanation:

python code.py trainingSet testSet Numfeatures GPdepth generateGraph FileFeatureNames TotalIntervalsGraph > outputFile

Example: python code.py Dataset/wineRed-train-0.csv Dataset/wineRed-test-0.csv 11 6 1 Dataset/featuresWineRed.txt 5 > output_wineRed

note: in this second example the graph will be created in the "graphic_Dataset" directory










