package com.secondline.lotto;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;

import com.secondline.lotto.util.DataUtil;

public class LottoNN {

	public static void main(String[] args) {
		DataSet dataSet = createDataSet();

		int inputCount = 6; // 5 numbers + powerball
		int outputCount = 69 + 26; // all numbers + powerball options
		int hiddenLayerCount = 4;

		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, hiddenLayerCount, outputCount);
		BackPropagation learningRule = neuralNet.getLearningRule();

		learningRule.setLearningRate(0.5);
		learningRule.setMaxError(0.001);
		learningRule.setMaxIterations(5000);

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
		neuralNet.learn(dataSet);

		// train the network with training set
		testNeuralNetwork(neuralNet, dataSet);

	}

	private static DataSet createDataSet() {
		DataSet data = new DataSet(6, 95);

		List<String> dataRows = DataUtil.getRowsFromFile("src/main/resources/train/testtraindata.csv");

		// TODO: only use random set
		List<String> trainRows = dataRows.subList(0, dataRows.size() / 2);

		// transform string to datasetrow
		for (String trainRow : trainRows) {
			DataSetRow row = createDataSetRow(trainRow);
			if (row != null)
				data.addRow(row);
		}

		return data;
	}

	private static DataSetRow createDataSetRow(String trainRow) {
		String[] elements = trainRow.split("|");
		if (elements.length != 12)
			return null;

		double[] inputs = new double[6];
		double[] outputs= new double[95];

		//first 6 columns of row are input layer values
		for(int i = 0; i < 6; ++i){
			inputs[i] = Double.valueOf(elements[i]);
		}

		// next 5 columns are desired output numbers
		for(int i = 7; i < 11; ++i){
			//index of the output layer equals the lotto number
			outputs[Integer.valueOf(elements[i])] = 1;
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
			System.out.println(testSetRow.getInput() + "----->" + predictions);
			System.out.println("");
		}
	}

	public static int[] maxOutputs(double[] array) {

		int[] result = { 0, 0, 0, 0, 0, 0 };
		double[] sortedNumbers = array.clone();
		// sort thru numbers then powerballs
		Arrays.sort(sortedNumbers, 0, 68);
		Arrays.sort(sortedNumbers, 69, 94);

		List sorted = Arrays.asList(sortedNumbers);
		for (int i = 0; i < 5; ++i) {
			result[i] = sorted.indexOf(sortedNumbers[68 - i]);
		}
		result[5] = sorted.indexOf(sortedNumbers[94]);
		return result;
	}

	static Comparator<Double> comp = new Comparator<Double>() {

		public int compare(Double o1, Double o2) {
			return o2.compareTo(o1);
		}

	};
}
