package com.y34h1a.ml.classification;

import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import smile.classification.DecisionTree;
import smile.data.Attribute;
import smile.data.AttributeDataset;
import smile.data.NominalAttribute;
import smile.data.NumericAttribute;
import smile.data.parser.DelimitedTextParser;
import smile.math.Math;

/**
 * Created by y34h1a on 9/9/17.
 */

public class DecisionTreeClassification {
    public static void main(String args[]) throws IOException, ParseException {
        String fileName = "/home/y34h1a/StudioProjects/Mechine Learning Examples/Data/train-0.1m.csv";
        DelimitedTextParser parser = new DelimitedTextParser();
        parser.setDelimiter(",");
        parser.setColumnNames(true);

        Attribute[] attributes = new Attribute[8];
        attributes[0] = new NominalAttribute("V0");
        attributes[1] = new NominalAttribute("V2");
        attributes[2] = new NominalAttribute("V3");
        attributes[3] = new NumericAttribute("V4");
        attributes[4] = new NominalAttribute("V5");
        attributes[5] = new NominalAttribute("V6");
        attributes[6] = new NominalAttribute("V7");
        attributes[7] = new NominalAttribute("V8");


        parser.setResponseIndex(new NominalAttribute("class"), 8);
        AttributeDataset data = parser.parse(attributes, new File(fileName));

        double[][] x = data.toArray(new double[data.size()][]);
        int[]y = data.toArray(new int[data.size()]);

        double[] maxX = Math.colMax(x);
        double[] minX = Math.colMin(x);


        NormalizedField normalizedField = new NormalizedField();
        normalizedField.setAction(NormalizationAction.Normalize);

        // Normalize X veriable
        for (int i = 0; i < x.length; i++) {

            for (int j = 0; j < x[i].length; j++) {
                double min = minX[j];
                double max = maxX[j];

                normalizedField.setActualHigh(max);
                normalizedField.setActualLow(min);

                x[i][j] = normalizedField.normalize(x[i][j]);
            }
        }

        DecisionTree ols = new DecisionTree(x,y,300);

        for (int i = 0; i < x.length; i++){
            System.out.println(ols.predict(x[i]) + " Actual(" + y[i]);
        }

    }
}
