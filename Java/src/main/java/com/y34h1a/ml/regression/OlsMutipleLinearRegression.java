package com.y34h1a.ml.regression;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;
import smile.regression.OLS;

/**
 * Created by y34h1a on 9/9/17.
 */

public class OlsMutipleLinearRegression {
    public static void main(String args[]) throws IOException, ParseException {
        String fileName = "/home/y34h1a/StudioProjects/Mechine Learning Examples/Data/abalone.arff";
        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(8);

        AttributeDataset weather = arffParser.parse(new File(fileName));
        double[][] x = weather.toArray(new double[weather.size()][]);
        double[] y = weather.toArray(new double[weather.size()]);

        OLS ols = new OLS(x,y);

        for (int i = 0; i < x.length; i++){
            System.out.println(ols.predict(x[i]) + " Actual(" + y[i]);
        }

        System.out.print(ols.toString());


    }
}
