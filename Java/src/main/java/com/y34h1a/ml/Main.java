package com.y34h1a.ml;

import com.y34h1a.ml.utils.ExtractArray;

import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import smile.classification.LogisticRegression;
import smile.data.Attribute;
import smile.data.AttributeDataset;
import smile.data.NominalAttribute;
import smile.data.NumericAttribute;
import smile.data.parser.DelimitedTextParser;
import smile.math.Math;
import smile.validation.LOOCV;

/**
 * Created by y34h1a on 9/3/17.
 */

public class Main {
    public static void main(String args[]) throws IOException, ParseException {
        String fileName = "/home/y34h1a/test/Social_Network_Ads.csv";

        DelimitedTextParser parser = new DelimitedTextParser();
        parser.setDelimiter(",");
        parser.setColumnNames(true);

        Attribute[] attributes = new Attribute[4];
        attributes[0] = new NumericAttribute("V0");
        attributes[1] = new NominalAttribute("V2");
        attributes[2] = new NumericAttribute("V3");
        attributes[3] = new NumericAttribute("V4");

        parser.setResponseIndex(new NominalAttribute("class"), 4);

        AttributeDataset data = parser.parse(attributes, new File(fileName));

        double[][] x = data.toArray(new double[data.size()][]);
        int[]y = data.toArray(new int[data.size()]);

        double[][] trainX = ExtractArray.deleteColumn(x, 0);


        double[] maxX = Math.colMax(trainX);
        double[] minX = Math.colMin(trainX);

        NormalizedField normalizedField = new NormalizedField();
        normalizedField.setAction(NormalizationAction.Ignore);

        for (int i = 0; i < trainX.length; i++){

            for (int j = 0; j < trainX[i].length; j++){
                double min = minX[j];
                double max = maxX[j];

                normalizedField.setActualHigh(max);
                normalizedField.setActualLow(min);

                trainX[i][j] = normalizedField.normalize(trainX[i][j]);
            }
        }

        int n = trainX.length;
        LOOCV loocv = new LOOCV(n);

        int error = 0;
        for (int i = 0; i < n; i++) {
            double[][] trainx = Math.slice(trainX, loocv.train[i]);
            int[] trainy = Math.slice(y, loocv.train[i]);

            LogisticRegression lr = new LogisticRegression(trainx, trainy);
            if (y[loocv.test[i]] != lr.predict(trainX[loocv.test[i]]))
                error++;
        }

        System.out.println(error);

    }
}
