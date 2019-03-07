package com.secondline.lotto.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class DataUtil {

	public static List<String> getRowsFromFile(String path) {
		try {
			return Files.readAllLines(Paths.get(path));
		} catch (IOException e) {
			return null;
		}
	}

}
