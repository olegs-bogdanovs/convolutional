package com.tsi.bak.convolution;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.FrameFilter;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.File;
import java.util.Random;

public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);

    private static int height=32;
    private static int width=32;
    private static long seed = 123;
    private static Random rng = new Random(seed);
    private static int batchSize = 250;
    private static double splitTrainTest = 0.8;
    private static int iterations = 1;
    private static int nEpochs = 150;
    protected static int channels = 1;


    private int numLabels;

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }


    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }


    public void execute(String[] args) throws Exception{

        log.info("Loading data...");

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File("dataset/");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(fileSplit.length());
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);

        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        log.info("Fitting to dataset");
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false).l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .learningRate(0.05) // tried 0.001, 0.005, 0.01
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAM)
                .list()
                .layer(0, convInit("cnn1", channels, 10 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, conv3x3("cnn3", 40,1))
                .layer(3, maxPool("maxpool2", new int[]{2,2}))
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(512).dropOut(0.5).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);

        network.init();
        // Visualizing Network Training
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        network.setListeners((IterationListener) new StatsListener(statsStorage),new ScoreIterationListener(iterations));

        /**
         * Load data
         */
        log.info("Load data....");
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;


        log.info("Train model....");
        // Train without transformations
        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        preProcessor.fit(dataIter);
        dataIter.setPreProcessor(preProcessor);
        trainIter = new MultipleEpochsIterator(nEpochs, dataIter);
        network.fit(trainIter);


        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        preProcessor.fit(dataIter);
        dataIter.setPreProcessor(preProcessor);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));

        if (true) {
            log.info("Save model....");
            ModelSerializer.writeModel(network,  "letters.bin", true);
        }
        log.info("**************** Bird Classification finished ********************");

    }

    public static void test() throws Exception {
        File model = new File("letters.bin");
        File test = new File("dataset/2/41.jpg");

        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(model);


        NativeImageLoader loader = new NativeImageLoader(32, 32,1);
        INDArray image = loader.asMatrix(test);
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
        preProcessor.transform(image);

        long start = System.currentTimeMillis();
        INDArray output = network.output(image, false);
        System.out.println(System.currentTimeMillis() - start);
        System.out.println(output.getFloat(1) > 0.8);
    }
    public static void main(String[] args) {
        try {
            //new App().execute(args);

            new App().test();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
