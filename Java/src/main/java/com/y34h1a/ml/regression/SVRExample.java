package com.y34h1a.ml.regression;

import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import smile.data.Attribute;
import smile.data.AttributeDataset;
import smile.data.NominalAttribute;
import smile.data.NumericAttribute;
import smile.data.parser.DelimitedTextParser;
import smile.math.Math;
import smile.math.kernel.GaussianKernel;
import smile.regression.SVR;

/**
 * Created by y34h1a on 9/9/17.
 */

public class SVRExample {
    public static void main(String args[]) throws IOException, ParseException {
        String fileName = "/home/y34h1a/StudioProjects/Mechine Learning Examples/Data/Position_Salaries.csv";
        DelimitedTextParser parser = new DelimitedTextParser();
        parser.setDelimiter(",");
        parser.setColumnNames(true);

        Attribute[] attributes = new Attribute[2];

        attributes[0] = new NominalAttribute("posittion");
        attributes[1] = new NominalAttribute("level");

        parser.setResponseIndex(new NumericAttribute("salary"), 2);

        AttributeDataset usps = parser.parse(attributes, new File(fileName));

        double[][] x = usps.toArray(new double[usps.size()][]);
        double[]y = usps.toArray(new double[usps.size()]);

        double minY = Math.min(y);
        double maxY = Math.max(y);


        double[] maxX = Math.colMax(x);
        double[] minX = Math.colMin(x);

        //If You Need Independent Veriable to normalize then do it

//        NormalizedField normalizedField = new NormalizedField();
//        normalizedField.setAction(NormalizationAction.Ignore);
//        normalizedField.setNormalizedHigh(2.5);
//        normalizedField.setNormalizedLow(-1.5);
//        // Normalize X veriable
//        for (int i = 0; i < x.length; i++){
//
//            for (int j = 0; j < x[i].length; j++){
//                double min = minX[j];
//                double max = maxX[j];
//
//                normalizedField.setActualHigh(max);
//                normalizedField.setActualLow(min);
//
//                x[i][j] = normalizedField.normalize(x[i][j]);
//                System.out.print(x[i][j] + " ");
//            }
//
//            System.out.println();
//        }

        NormalizedField normalizedFieldY = new NormalizedField();
        normalizedFieldY.setActualHigh(maxY);
        normalizedFieldY.setActualLow(minY);
        normalizedFieldY.setAction(NormalizationAction.Normalize);
        normalizedFieldY.setNormalizedHigh(2.5);
        normalizedFieldY.setNormalizedLow(-0.7);

        //Normalize Y veriable
        for (int i = 0; i < y.length; i++){
            y[i] = normalizedFieldY.normalize(y[i]);
            System.out.println(y[i] + " ");

        }

        System.out.print("\n\n\n");


        SVR<double[]> svr = new SVR<>(x, y, new GaussianKernel(1.5), 0.03, 1.0);
        double predicted = normalizedFieldY.deNormalize(svr.predict(new double[]{6.5,6.5}));
        System.out.println(predicted + " ");
        System.out.print("\n\n");

        for (int i = 0 ; i < x.length; i++) {
            double predictedY = normalizedFieldY.deNormalize(svr.predict(x[i]));
            System.out.println( (i) + ": " + predictedY + " " + "(Actual: " + normalizedFieldY.deNormalize(y[i]));
        }

//        0: 38433.89706916765 (Actual Salary: 35000)
//        1: 46624.12744121588 (Actual Salary: 50000)
//        2: 63375.8725587841 (Actual Salary: 60000.0)
//        3: 83419.84871052955 (Actual Salary: 80000.0)
//        4: 90849.03640783756 (Actual Salary: 90000.0)
//        5: 103453.55143045585 (Actual Salary: 100000.0)
//        6: 146534.4743391068 (Actual Salary: 150000.0)
//        7: 203433.89706916767 (Actual Salary: 200000.0)
//        8: 296573.0213434293 (Actual Salary: 300000.0)
//        9: 317563.30864602653 (Actual Salary: 400000.0)
    }
}
