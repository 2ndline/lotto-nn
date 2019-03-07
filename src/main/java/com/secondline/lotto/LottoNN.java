package com.secondline.lotto;

import java.util.ArrayList;
import java.util.Arrays;

import com.google.common.primitives.Doubles;
import java.util.Comparator;
import java.util.List;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

import com.secondline.lotto.util.DataUtil;

public class LottoNN {

	public static void main(String[] args) {
		List<String> dataRows = DataUtil.getRowsFromFile("src/main/resources/train/testtraindata.csv");

		DataSet dataSet = createDataSet(dataRows);
		dataSet.shuffle();
		
		//Normalizing data set
        Normalizer normalizer = new MaxNormalizer();
        normalizer.normalize(dataSet);
        
		DataSet[] sets = dataSet.createTrainingAndTestSubsets(60, 40);
		DataSet trainSet = sets[0];
		DataSet testSet = sets[1];

		int inputCount = 6; // 5 numbers + powerball
		int outputCount = 69 + 26; // all numbers + powerball options

		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, 12, 12, 20, outputCount);
		BackPropagation learningRule = neuralNet.getLearningRule();

		learningRule.setLearningRate(0.5);
		learningRule.setMaxError(0.001);
		learningRule.setMaxIterations(2500);

		// add learning listener in order to print out training info
		learningRule.addListener(new LearningEventListener() {

			public void handleLearningEvent(LearningEvent event) {
				BackPropagation bp = (BackPropagation) event.getSource();
				if (event.getEventType().equals(LearningEvent.Type.LEARNING_STOPPED)) {
					System.out.println();
					System.out.println("Training completed in " + bp.getCurrentIteration() + " iterations");
					System.out.println("With total error " + bp.getTotalNetworkError() + '\n');
				} else {
					System.out.println("Iteration: " + bp.getCurrentIteration() + " | Network error: "
							+ bp.getTotalNetworkError());
				}
			}

		});

		// train neural network
		neuralNet.learn(trainSet);

		// train the network with training set
		testNeuralNetwork(neuralNet, testSet);

	}

	private static DataSet createDataSet(List<String> textRows) {
		DataSet data = new DataSet(6, 95);

		// transform string to datasetrow
		for (String trainRow : textRows) {
			DataSetRow row = createDataSetRow(trainRow);
			if (row != null)
				data.addRow(row);
		}

		return data;
	}

	private static DataSetRow createDataSetRow(String trainRow) {
		String[] elements = trainRow.split("\\|");
		if (elements.length != 12)
			return null;

		double[] inputs = new double[6];
		double[] outputs= new double[95];

		//first 6 columns of row are input layer values
		for(int i = 0; i < 6; ++i){
			inputs[i] = Double.valueOf(elements[i]);
		}

		// next 5 columns are desired output numbers
		for(int i = 6; i < 11; ++i){
			//index of the output layer equals the lotto number
			outputs[Integer.valueOf(elements[i]) -1] = 1;
		}
		//last column is desired output powerball
		outputs[68+ Integer.valueOf(elements[11])] = 1;

		return new DataSetRow(inputs, outputs);
	}

	/**
	 * Prints network output for the each element from the specified training
	 * set.
	 *
	 * @param neuralNet
	 *            neural network
	 * @param testSet
	 *            test data set
	 */
	public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {

		System.out.println("--------------------------------------------------------------------");
		System.out.println("***********************TESTING NEURAL NETWORK***********************");
		for (DataSetRow testSetRow : testSet.getRows()) {
			neuralNet.setInput(testSetRow.getInput());
			neuralNet.calculate();

			int[] predictions = maxOutputs(neuralNet.getOutput());
			System.out.println(printArray(testSetRow.getInput()) + "----->" + printIntArray(predictions));
			System.out.println("");
		}
	}

	private static String printArray(double[] object){
		String result = "";
		for(double o : object){
			result += o+"-";
		}
		return result;
	}
	
	private static String printIntArray(int[] object){
		String result = "";
		for(int o : object){
			result += o+"-";
		}
		return result;
	}
	
	public static int[] maxOutputs(double[] array) {

		int[] result = new int[6];
		double[] sortedNumbers = array.clone();
		// sort thru number predictions, then powerball predictions
		Arrays.sort(sortedNumbers, 0, 69);
		Arrays.sort(sortedNumbers, 69, 95);

		List<Double> sorted = Doubles.asList(array);
		for (int i = 0; i < 5; ++i) {
			int index = sorted.indexOf(sortedNumbers[68 - i]);
			result[i] = index + 1;
		}
		result[5] = 95 - sorted.indexOf(sortedNumbers[94]);
		return result;
	}
}
