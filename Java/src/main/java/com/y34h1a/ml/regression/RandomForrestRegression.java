package com.y34h1a.ml.regression;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import smile.classification.RandomForest;
import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;

/**
 * Created by y34h1a on 9/9/17.
 */

public class RandomForrestRegression {
    public static void main(String args[]) throws IOException, ParseException {
        String fileName = "/home/y34h1a/StudioProjects/Mechine Learning Examples/Data/weather.nominal.arff";
        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(4);
        AttributeDataset weather = arffParser.parse(new File(fileName));


        double[][] x = weather.toArray(new double[weather.size()][]);
        int[]y = weather.toArray(new int[weather.size()]);

        RandomForest randomForest = new RandomForest(weather.attributes(),x,y,300);

        for (int i = 0 ; i < x.length; i++) {
            double predictedY = randomForest.predict(x[i]);
            System.out.println( (i) + ": " + predictedY + " " + "(Actual: " + y[i]);
        }

    }
}
