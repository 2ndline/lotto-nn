package com.secondline.lotto;

import java.util.Arrays;

import com.google.common.primitives.Doubles;
import java.util.List;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.JordanNetwork;
import org.neuroph.nnet.learning.BackPropagation;
import com.secondline.lotto.util.DataUtil;

public class LottoNN {

	public static void main(String[] args) {
		List<String> dataRows = DataUtil.getRowsFromFile("src/main/resources/train/testtraindata.csv");

		DataSet dataSet = createDataSet(dataRows);
		dataSet.shuffle();

		DataSet[] sets = dataSet.createTrainingAndTestSubsets(65, 35);
		DataSet trainSet = sets[0];
		trainSet.shuffle();
		DataSet testSet = sets[1];

		int inputCount = 6; // 5 numbers + powerball
		int outputCount = 69 + 26; // all numbers + powerball options

		JordanNetwork neuralNet = new JordanNetwork(inputCount, 36, 64, outputCount);
		BackPropagation learningRule = new BackPropagation();

		learningRule.setLearningRate(0.004);
		learningRule.setMaxError(0.05);
		learningRule.setMaxIterations(3000);

		// add learning listener in order to print out training info
		learningRule.addListener(new LearningEventListener() {

			public void handleLearningEvent(LearningEvent event) {
				BackPropagation bp = (BackPropagation) event.getSource();
				if (event.getEventType().equals(LearningEvent.Type.LEARNING_STOPPED)) {
					System.out.println();
					System.out.println("Training completed in " + bp.getCurrentIteration() + " iterations");
					System.out.println("With total error " + bp.getTotalNetworkError() + '\n');
				} else if((bp.getCurrentIteration() % 100) == 0){
					System.out.println("Iteration: " + bp.getCurrentIteration() + " | Network error: "
							+ bp.getTotalNetworkError());
				}
			}

		});
		neuralNet.setLearningRule(learningRule);

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
		double[] outputs = new double[95];

		// first 6 columns of row are input layer values
		for (int i = 0; i < 6; ++i) {
			inputs[i] = Double.valueOf(elements[i]);
		}

		// next 5 columns are desired output numbers
		for (int i = 6; i < 11; ++i) {
			// index of the output layer equals the lotto number
			outputs[Integer.valueOf(elements[i]) - 1] = 1;
		}
		// last column is desired output powerball
		outputs[68 + Integer.valueOf(elements[11])] = 1;

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
		int totalWinnings = 0, count = 0;
		for (DataSetRow testSetRow : testSet.getRows()) {
			neuralNet.setInput(testSetRow.getInput());
			neuralNet.calculate();

			int[] predictions = maxOutputs(neuralNet.getOutput());
			double[] expected = testSetRow.getDesiredOutput();
			int amountWon = getWinningAmount(predictions, expected);
			System.out.println(printArray(testSetRow.getInput()) + ", guessed " + printIntArray(predictions)+ ", Expected: " + printIntArray(expected(expected)) + ", wins $" + amountWon);
			System.out.println("");
			totalWinnings += amountWon;
			count++;
		}
		System.out.println("***********************DONE TESTING NEURAL NETWORK***********************");
		System.out.println("Total winnings: $"+totalWinnings+", Avg winnings: $" + (totalWinnings / count));
	}

	private static int getWinningAmount(int[] predictions, double[] expected) {
		final int[] winningNumbers = expected(expected);

		int matchingNumbers = 0;
		for (int i = 0; i < winningNumbers.length - 1; ++i) {
			int winningNumber = winningNumbers[i];
			for (int j = 0; j < predictions.length - 1; ++j) {
				int guessedNumber = predictions[j];
				if (guessedNumber == winningNumber)
					matchingNumbers++;
			}
		}
		boolean powerballCorrect = winningNumbers[5] == predictions[5];

		switch (matchingNumbers) {
		case 1:
			return powerballCorrect ? 4 : 0;
		case 2:
			return powerballCorrect ? 7 : 0;
		case 3:
			return powerballCorrect ? 100 : 7;
		case 4:
			return powerballCorrect ? 50000 : 100;
		case 5:
			return powerballCorrect ? 100000000 : 1000000;
		default:
			return powerballCorrect ? 4 : 0;

		}
	}

	private static String printArray(double[] object) {
		String result = "";
		for (double o : object) {
			result += o + "-";
		}
		return result;
	}

	private static String printIntArray(int[] object) {
		String result = "";
		for (int o : object) {
			result += o + "-";
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
	
	public static int[] expected(double[] array) {

		int[] result = new int[6];
		int count = 0;
		for(int i = 0; i < 69; ++i){
			if(array[i] > 0.1){
				result[count] = i+1;
				count++;
			}
		}
		for(int i = 69; i < 95; ++ i){
			if(array[i] > 0.1){
				result[5] = 95 - i;
				break;
			}
		}
		return result;
	}
}
