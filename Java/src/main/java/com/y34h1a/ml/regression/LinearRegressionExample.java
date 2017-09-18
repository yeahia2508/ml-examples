package com.y34h1a.ml.regression;

import com.y34h1a.ml.utils.ExtractArray;
import com.y34h1a.ml.utils.LinearRegression;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import smile.data.Attribute;
import smile.data.AttributeDataset;
import smile.data.NominalAttribute;
import smile.data.NumericAttribute;
import smile.data.parser.DelimitedTextParser;

/**
 * Created by y34h1a on 9/18/17.
 */

public class LinearRegressionExample {
    public static void main(String args[]) throws IOException, ParseException {
        String fileName = "/home/tareq/StudioProjects/Mechine Learning Examples/Data/Position_Salaries.csv";
        DelimitedTextParser parser = new DelimitedTextParser();
        parser.setDelimiter(",");
        parser.setColumnNames(true);

        Attribute[] attributes = new Attribute[2];

        attributes[0] = new NominalAttribute("posittion");
        attributes[1] = new NominalAttribute("level");

        parser.setResponseIndex(new NumericAttribute("salary"), 2);

        AttributeDataset usps = parser.parse(attributes, new File(fileName));

        double[] x = ExtractArray.convertSingleColumn(usps.toArray(new double[usps.size()][]), 1);
        double[]y = usps.toArray(new double[usps.size()]);

        LinearRegression linearRegression = new LinearRegression(x,y);

        for (int i = 0; i < x.length; i++) {
            System.out.println(linearRegression.predict(x[i]) + " Original : " + y[i]);
        }
    }
}
